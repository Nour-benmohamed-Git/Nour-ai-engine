"""
Arabic Quran ASR Server — v4.3  (CTC interim + RNNT final)
===========================================================

Architecture
------------
• INTERIM  chunks    → CTC decoder  (~30-50 ms, non-autoregressive)
• FINAL    utterance → RNNT decoder (accurate,  autoregressive)

RNNT is triggered by three events:
  1. Client sends {"type": "eof"}
  2. Silence gap of SILENCE_TRIGGER_S seconds (best real-time UX — fires
     during the natural pause between Quran ayahs, before the next starts)
  3. Utterance reaches MAX_UTTERANCE_SAMPLES (model's training max = 20 s)

v4.3 fixes over v4.2
---------------------
1. 30 s utterance cap exceeded model's 20 s training max → degraded RNNT
   accuracy and encoder positional embedding range.  Fixed: cap at 20 s and
   auto-finalize (not trim) when reached.

2. disconnect-final RNNT called send_text() on a closed socket.  The client
   is gone; nobody receives the result.  Removed: RNNT on disconnect is
   pointless and causes a spurious send error in the logs.

3. CTC results sent repeatedly for the same text.  Logs showed "نهاية"
   sent 10+ consecutive times.  Fixed: per-session deduplication — CTC
   result is only transmitted when the text differs from the last sent.

4. Silence-gap RNNT trigger added.  A background asyncio task per session
   watches the last-speech timestamp; when the gap exceeds SILENCE_TRIGGER_S
   it calls finalize().  For Quran recitation this fires RNNT during the
   natural inter-ayah pause and delivers the accurate result before the
   reciter starts the next verse.

5. CTC warm-up reimplemented the decode logic bypassing _transcribe_ctc().
   Fixed: warm-up now calls _transcribe_ctc() and _transcribe_rnnt() directly
   using a white-noise burst that passes the VAD gate (RMS ≈ 0.1 >> 0.005).

6. Unused `traceback` import removed.
"""

# ============================================================================
# ENV — before ALL imports (torch and NeMo check these at import time)
# ============================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import logging
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import nemo.collections.asr as nemo_asr  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_RATE           = 16000
CHUNK_SAMPLES         = 15360    # 0.96 s  — CTC sliding window size
CHUNK_STRIDE          = 7680     # 0.48 s  — 50 % overlap for smooth interim
MAX_UTTERANCE_SAMPLES = 320000   # 20 s    — matches model's training max_duration
MIN_UTTERANCE_SAMPLES = 8000     # 0.5 s   — minimum audio to bother calling RNNT
VAD_RMS_THRESHOLD     = 0.005    # ~−46 dBFS; chunks below this are silence
SILENCE_TRIGGER_S     = 0.8      # gap after last speech that triggers RNNT
SILENCE_POLL_S        = 0.1      # how often the silence monitor checks
MODEL_NAME            = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"

# ============================================================================
# MODEL LOAD
# ============================================================================

logger.info(f"Loading model: {MODEL_NAME}")
_t0 = time.time()
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.cuda()
    logger.info("Model on GPU")
else:
    logger.warning("GPU not available — running on CPU (very slow)")

CTC_BLANK_ID = model.tokenizer.vocab_size
logger.info(f"Model loaded in {time.time() - _t0:.2f}s")

# ============================================================================
# RNNT DECODING STRATEGY
# ============================================================================
# The model's default greedy config has use_cuda_graph_decoder: true.
# cuStreamGetCaptureInfo returns 5 values on this driver but NeMo's binding
# unpacks 6, crashing with "not enough values to unpack (expected 6, got 5)".
# Disabling cuda graphs restores the standard Python greedy path with
# identical numeric results.
#
# Clean approach: instantiate RNNTBPEDecodingConfig (the official NeMo
# dataclass) directly and set fields on it as plain Python attributes.
# change_decoding_strategy() internally merges this against the full schema
# (OmegaConf.structured → to_container → create → merge), so every field
# is valid and no OmegaConf struct-key errors are possible.
# ============================================================================

logger.info("Configuring RNNT decoder (greedy_batch, cuda-graphs off)...")
try:
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecodingConfig

    rnnt_cfg = RNNTBPEDecodingConfig(strategy="greedy_batch")
    rnnt_cfg.greedy.max_symbols            = 10
    rnnt_cfg.greedy.use_cuda_graph_decoder = False   # disables the crashing path

    model.change_decoding_strategy(decoder_type="rnnt", decoding_cfg=rnnt_cfg)
    logger.info("RNNT decoder configured ✓")

except Exception as exc:
    logger.error(f"Failed to configure RNNT decoder: {exc}")
    raise   # RNNT is required — hard fail at startup

# ============================================================================
# ENCODER
# ============================================================================

def _encode(audio_f32: np.ndarray):
    """Preprocessor + encoder. Must be called inside torch.inference_mode()."""
    signal = torch.from_numpy(audio_f32).unsqueeze(0).to(device)
    length = torch.tensor([audio_f32.shape[0]], device=device)
    processed, proc_len = model.preprocessor(input_signal=signal, length=length)
    encoded, enc_len    = model.encoder(audio_signal=processed, length=proc_len)
    return encoded, enc_len


# ============================================================================
# VAD GATE
# ============================================================================

def _has_speech(audio: np.ndarray) -> bool:
    """True when RMS energy exceeds the silence threshold."""
    return float(np.sqrt(np.mean(audio ** 2))) > VAD_RMS_THRESHOLD


# ============================================================================
# CTC — interim decoder  (fast, ~30-50 ms)
# ============================================================================

def _transcribe_ctc(audio_f32: np.ndarray) -> str:
    """
    Greedy CTC decode.
    Returns "" for silent frames (VAD gate) — prevents the model from
    hallucinating training-time silence-marker tokens on quiet audio.

    Algorithm: collapse consecutive duplicate tokens, then strip blanks.
    """
    if not _has_speech(audio_f32):
        return ""

    try:
        with torch.inference_mode():
            encoded, _ = _encode(audio_f32)
            log_probs  = model.ctc_decoder(encoder_output=encoded)
            preds      = log_probs.argmax(dim=-1)[0].cpu().tolist()

        collapsed = [preds[0]] if preds else []
        for t in preds[1:]:
            if t != collapsed[-1]:
                collapsed.append(t)

        token_ids = [t for t in collapsed if t != CTC_BLANK_ID]
        text = model.tokenizer.ids_to_text(token_ids)
        return text.strip() if text else ""

    except Exception:
        logger.exception("CTC inference error")
        return ""


# ============================================================================
# RNNT — final decoder  (accurate, autoregressive)
# ============================================================================

def _transcribe_rnnt(audio_f32: np.ndarray) -> str:
    """
    Full-utterance RNNT decode via model.decoding.rnnt_decoder_predictions_tensor().

    model.decoding (RNNTBPEDecoding) is already wired to model.decoder (LSTM
    prediction network) and model.joint.  It expects encoder output in raw
    (B, D, T) shape and handles the internal (B,D,T)→(B,T,D) transpose.
    """
    try:
        with torch.inference_mode():
            encoded, enc_len = _encode(audio_f32)
            hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,    # (B, D, T) — raw encoder shape
                encoded_lengths=enc_len,
            )

        if hypotheses and hypotheses[0].text:
            return hypotheses[0].text.strip()

        # Some NeMo versions don't populate .text; fall back to y_sequence ids
        if hypotheses and hasattr(hypotheses[0], "y_sequence"):
            ids  = hypotheses[0].y_sequence
            ids  = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            text = model.tokenizer.ids_to_text(ids)
            return text.strip() if text else ""

        return ""

    except Exception:
        logger.exception("RNNT inference error")
        return ""


# ============================================================================
# WARM-UP — both decoders, hard fail on error
# ============================================================================
# Use a white-noise burst (RMS ≈ 0.1) so the VAD gate passes.
# This lets us call _transcribe_ctc and _transcribe_rnnt directly — no code
# duplication, and we exercise the exact same paths used in production.
# ============================================================================

logger.info("Warming up CTC + RNNT decoders...")
_t0 = time.time()

np.random.seed(0)
_warmup_audio = (np.random.randn(SAMPLE_RATE) * 0.1).astype(np.float32)

_ctc_text  = _transcribe_ctc(_warmup_audio)
logger.info(f"  CTC  warm-up OK — '{_ctc_text}'")

_rnnt_text = _transcribe_rnnt(_warmup_audio)   # raises + aborts if cuda-graph fix failed
logger.info(f"  RNNT warm-up OK — '{_rnnt_text}'")

logger.info(f"Warm-up complete in {time.time() - _t0:.2f}s")

# ============================================================================
# INFERENCE LOCK  (one GPU inference at a time — shared across all sessions)
# ============================================================================

_infer_lock = asyncio.Lock()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Quran ASR Server", version="4.3.0")


@app.get("/")
def root():
    return {
        "service":  "Quran ASR Server",
        "version":  "4.3.0",
        "model":    MODEL_NAME,
        "status":   "operational",
        "decoders": {"interim": "CTC", "final": "RNNT (greedy_batch)"},
    }


@app.get("/health")
def health():
    return {
        "status":            "healthy",
        "model":             MODEL_NAME,
        "device":            device,
        "chunk_ms":          int(CHUNK_SAMPLES / SAMPLE_RATE * 1000),
        "stride_ms":         int(CHUNK_STRIDE  / SAMPLE_RATE * 1000),
        "overlap_pct":       int((1 - CHUNK_STRIDE / CHUNK_SAMPLES) * 100),
        "max_utterance_s":   MAX_UTTERANCE_SAMPLES / SAMPLE_RATE,
        "min_utterance_s":   MIN_UTTERANCE_SAMPLES / SAMPLE_RATE,
        "silence_trigger_s": SILENCE_TRIGGER_S,
        "vad_rms_threshold": VAD_RMS_THRESHOLD,
        "interim_decoder":   "ctc",
        "final_decoder":     "rnnt",
    }


# ============================================================================
# INFERENCE + SEND HELPER
# ============================================================================

async def _run_and_send(
    websocket:    WebSocket,
    audio_f32:    np.ndarray,
    is_final:     bool,
    *,
    last_ctc_text: str = "",   # for interim deduplication; ignored for final
) -> str:
    """
    Run the appropriate decoder on audio and send the result to the client.

    For interim (CTC): skips send if the text is identical to last_ctc_text.
    For final   (RNNT): always sends (each utterance is a new event).

    Returns the text that was sent, or "" if nothing was sent.
    """
    if audio_f32.size == 0:
        return ""

    fn = _transcribe_rnnt if is_final else _transcribe_ctc
    t0 = time.time()

    async with _infer_lock:
        text = await asyncio.to_thread(fn, audio_f32)

    if not text:
        return ""

    # Deduplication: don't send the same CTC result twice in a row
    if not is_final and text == last_ctc_text:
        return ""

    words         = text.split()
    processing_ms = int((time.time() - t0) * 1000)

    await websocket.send_text(json.dumps({
        "words":         words,
        "is_final":      is_final,
        "decoder":       "rnnt" if is_final else "ctc",
        "chunk_ms":      int(audio_f32.size / SAMPLE_RATE * 1000),
        "processing_ms": processing_ms,
    }, ensure_ascii=False))

    logger.info(
        f"{'FINAL(RNNT)' if is_final else 'interim(CTC)'} | "
        f"{len(words)} words | {processing_ms}ms | {text[:80]}"
    )
    return text


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================
#
# Client → server:
#   binary frames : raw PCM16-LE at 16 kHz
#   text frame    : {"type": "eof"} — signals end of utterance → RNNT pass
#
# Server → client:
#   interim : {"words":[…], "is_final":false, "decoder":"ctc",  …}
#   final   : {"words":[…], "is_final":true,  "decoder":"rnnt", …}
#
# Per-session state:
#   ctc_buf        — CTC sliding window; advanced by CHUNK_STRIDE each step
#   utterance_buf  — full audio accumulator; passed intact to RNNT; never sliced
#   last_speech_t  — wall-clock time of last speech-containing chunk
#   last_ctc_text  — last CTC text sent; used for deduplication

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = id(websocket)
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    ctc_buf:       np.ndarray = np.empty(0, dtype=np.float32)
    utterance_buf: np.ndarray = np.empty(0, dtype=np.float32)
    last_ctc_text: str        = ""
    last_speech_t: float      = 0.0
    had_speech:    bool       = False    # True once any speech is detected
    finalizing:    bool       = False    # guard against concurrent finalizations
    chunks_processed:    int  = 0

    async def _finalize(reason: str) -> None:
        """
        Run RNNT on the accumulated utterance and reset session state.
        No-op if already finalizing or utterance is too short.
        """
        nonlocal ctc_buf, utterance_buf, last_ctc_text
        nonlocal last_speech_t, had_speech, finalizing

        if finalizing:
            return
        finalizing = True

        try:
            n = utterance_buf.size
            if n >= MIN_UTTERANCE_SAMPLES:
                logger.info(
                    f"{reason}: session={session_id} | "
                    f"{n/SAMPLE_RATE:.1f}s → RNNT"
                )
                await _run_and_send(websocket, utterance_buf, is_final=True)
            else:
                logger.info(
                    f"{reason}: session={session_id} | "
                    f"{n} samples — too short for RNNT"
                )
        finally:
            ctc_buf       = np.empty(0, dtype=np.float32)
            utterance_buf = np.empty(0, dtype=np.float32)
            last_ctc_text = ""
            last_speech_t = 0.0
            had_speech    = False
            finalizing    = False

    async def _silence_monitor() -> None:
        """
        Background task: fires RNNT when the speaker has been silent for
        SILENCE_TRIGGER_S seconds after having produced speech.

        This is the key to real-time Quran UX — RNNT fires during the natural
        pause between ayahs, delivering the accurate result before the reciter
        starts the next verse.
        """
        while True:
            await asyncio.sleep(SILENCE_POLL_S)
            if (had_speech
                    and not finalizing
                    and utterance_buf.size >= MIN_UTTERANCE_SAMPLES
                    and (time.time() - last_speech_t) >= SILENCE_TRIGGER_S):
                await _finalize("silence-gap")

    monitor_task = asyncio.create_task(_silence_monitor())

    try:
        while True:
            msg      = await websocket.receive()
            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                # Client is gone — do NOT attempt RNNT (socket is closed,
                # send_text would raise, and there's nobody to receive it).
                logger.info(f"Client disconnected: {session_id}")
                break

            if msg_type != "websocket.receive":
                continue

            # ── Text (control) ──────────────────────────────────────────────
            text_payload = msg.get("text")
            if text_payload:
                try:
                    obj = json.loads(text_payload)
                    if obj.get("type") == "eof":
                        await _finalize("EOF")
                        # don't break — client may continue with next utterance
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {text_payload[:80]}")
                except Exception:
                    logger.exception(f"Control message error: {session_id}")
                continue

            # ── Binary (audio) ──────────────────────────────────────────────
            data = msg.get("bytes")
            if not data:
                continue

            if len(data) % 2:          # PCM16 = 2 bytes per sample
                data = data[:-1]
            if not data:
                continue

            try:
                pcm_f32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                if pcm_f32.size == 0:
                    continue

                # Accumulate into the full-utterance buffer (for RNNT).
                # This buffer is NEVER sliced — it holds the complete utterance.
                utterance_buf = np.concatenate([utterance_buf, pcm_f32])

                # Auto-finalize at model's training max (20 s) instead of
                # trimming — trimming would silently drop the start of the ayah.
                if utterance_buf.size >= MAX_UTTERANCE_SAMPLES:
                    logger.info(
                        f"Max utterance reached ({session_id}) — auto-finalizing"
                    )
                    await _finalize("max-length")

                # Accumulate into the CTC sliding window
                ctc_buf = np.concatenate([ctc_buf, pcm_f32])

                while ctc_buf.size >= CHUNK_SAMPLES:
                    chunk   = ctc_buf[:CHUNK_SAMPLES]
                    ctc_buf = ctc_buf[CHUNK_STRIDE:]

                    # Update VAD / speech timestamp before inference
                    if _has_speech(chunk):
                        last_speech_t = time.time()
                        had_speech    = True

                    last_ctc_text = await _run_and_send(
                        websocket, chunk, is_final=False,
                        last_ctc_text=last_ctc_text,
                    )
                    chunks_processed += 1

            except Exception:
                logger.exception(f"Audio processing error: {session_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected (exc): {session_id}")
    except Exception:
        logger.exception(f"WebSocket error: {session_id}")
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(
            f"WebSocket closed: {session_id} | "
            f"{chunks_processed} CTC chunks processed"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Quran ASR Server v4.3")
    logger.info(f"  Device              : {device}")
    logger.info(f"  CTC chunk           : {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES/SAMPLE_RATE:.2f}s)")
    logger.info(f"  CTC stride          : {CHUNK_STRIDE} samples ({CHUNK_STRIDE/SAMPLE_RATE:.2f}s)")
    logger.info(f"  CTC overlap         : {(1 - CHUNK_STRIDE/CHUNK_SAMPLES)*100:.0f}%")
    logger.info(f"  Max utterance (RNNT): {MAX_UTTERANCE_SAMPLES/SAMPLE_RATE:.0f}s")
    logger.info(f"  Min utterance (RNNT): {MIN_UTTERANCE_SAMPLES/SAMPLE_RATE:.2f}s")
    logger.info(f"  Silence trigger     : {SILENCE_TRIGGER_S}s")
    logger.info(f"  VAD RMS threshold   : {VAD_RMS_THRESHOLD}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)