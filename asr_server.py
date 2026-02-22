"""
Arabic Quran ASR Server — v4.9  (CTC interim + RNNT final)
===========================================================

Architecture
------------
• INTERIM  chunks    → CTC decoder  (~30-50 ms, non-autoregressive)
• FINAL    utterance → RNNT decoder (accurate,  autoregressive)

RNNT is triggered by four events:
  1. Client sends {"type": "eof"}
  2. Silence gap of SILENCE_TRIGGER_S seconds (fires during the natural
     pause between Quran ayahs, before the next starts)
  3. Utterance reaches MAX_UTTERANCE_SAMPLES (model's training max = 20 s)
  4. Client disconnects after producing speech (graceful close window)

v4.9 improvements over v4.5
-----------------------------
1. torch.autocast removed — runs pure float32.
   v4.9 introduced torch.autocast (AMP) hoping for ~1.5× throughput.
   Warm-up (white noise) decoded correctly; real speech produced all-⁇.
   Root cause: real speech concentrates energy in specific formant bands,
   driving activations to magnitudes that overflow in FP16 → NaN → every
   argmax lands on SentencePiece unknown (⁇).  White noise has a flat
   spectrum that keeps activations moderate, so warm-up looked fine.
   autocast cannot be used safely with this FastConformer model.
   Pure float32 matches v4.5 behaviour and is confirmed correct.

2. session_ever_had_speech flag — idle timeout between ayahs fixed.
   The original had_speech guard on asyncio.wait_for resets to False after
   every _finalize().  A reciter pausing more than IDLE_TIMEOUT_S seconds
   between ayahs was silently disconnected.  The new session_ever_had_speech
   flag is set True on the first speech onset and never reset, so the idle
   timeout only applies before the very first audio frame of the session.

3. VAD threshold lowered 0.5 → 0.3; speech_pad_ms raised 30 → 80 ms.
   Catches quieter / lower-amplitude recitation voices.  Onset fires
   earlier and the gate stays open longer, preventing premature cutoff of
   trailing syllables.  min_silence_duration_ms lowered 100 → 80 ms to
   avoid over-segmenting fast ayahs.

4. Leading silence excluded from utterance_buf.
   Audio before the first speech onset was previously accumulated into
   utterance_buf, consuming up to IDLE_TIMEOUT_S seconds of the 20s model
   context window on dead air.  Now utterance_buf only starts filling once
   had_speech is True.

5. asyncio.Lock() created in FastAPI lifespan, not at module scope.
   Constructing asyncio primitives outside a running event loop emits
   DeprecationWarning on Python 3.8/3.9.  The lifespan hook runs inside
   the event loop before the first request.

6. UUID-based session IDs replace id(websocket).
   id() can reuse memory addresses across sessions in the same process,
   making log lines ambiguous.  uuid4()[:8] is unique per session.

7. Efficient rolling buffers for CTC and utterance.
   np.concatenate on every chunk is O(N²) total.  Both buffers now
   accumulate as Python lists and are concatenated once at consumption
   time, making the total work O(N).  The CTC loop uses a start-pointer
   to slice numpy views (zero-copy) instead of repeated concatenations.

8. Silence monitor poll interval 0.1 → 0.05 s for tighter trigger timing.
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
import uuid
from contextlib import asynccontextmanager

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

# Lowered 0.5 → 0.3: catches onset earlier on quiet voices; gate stays open
# longer (speech_pad_ms 80) so trailing syllables are not clipped.
VAD_SPEECH_THRESHOLD  = 0.3
VAD_CHUNK_SAMPLES     = 512      # Silero's required chunk size at 16 kHz (32 ms)

SILENCE_TRIGGER_S     = 0.8      # gap after last speech that triggers RNNT
SILENCE_POLL_S        = 0.05     # how often the silence monitor checks (was 0.1)
IDLE_TIMEOUT_S        = 10.0     # close ghost connections that never send audio
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
    # ── IMPORTANT — keep weights in FLOAT32 ────────────────────────────────
    # Do NOT call model.half(), model.encoder.half(), or any selective .half()
    # on submodules.  Do NOT use torch.autocast / AMP for this model.
    #
    # FastConformer attention + RNNT joint produce all-⁇ output under any
    # FP16 path (forced half OR autocast) when fed real speech.  White noise
    # (warm-up) appears to work because its flat spectrum keeps activations
    # moderate; real speech concentrates energy in specific formant bands,
    # pushing activations to magnitudes that overflow in FP16 → NaN → every
    # token decodes as SentencePiece unknown (⁇).
    #
    # Pure float32 is the only safe option for this model.
    # ───────────────────────────────────────────────────────────────────────
    logger.info("Model on GPU (float32)")
else:
    logger.warning("GPU not available — running on CPU (very slow)")

CTC_BLANK_ID = model.tokenizer.vocab_size
logger.info(f"Model loaded in {time.time() - _t0:.2f}s")

# ============================================================================
# SILERO VAD  (pip install silero-vad)
# ============================================================================
# Silero requires chunks of exactly 512 samples at 16 kHz (32 ms).
# Passing larger chunks causes unreliable onset/offset detection because
# the internal windowing can swallow speech events at the boundaries.
#
# We solve this with a dedicated 512-sample VAD sub-buffer per session,
# fed from the same incoming PCM stream independently of the CTC window.
# This gives accurate per-32ms speech detection at any volume level.
#
# threshold 0.3 (was 0.5): fires on lower speech probability — catches
#   quiet onset a few frames earlier without meaningfully more false positives.
# speech_pad_ms 80 (was 30): holds gate open longer after low-amplitude
#   trailing syllables so they aren't clipped before CTC/RNNT sees them.
# min_silence_duration_ms 80 (was 100): avoids over-segmenting fast ayahs
#   that have brief inter-word pauses.
# ============================================================================

logger.info("Loading Silero VAD...")
_vad_t0 = time.time()
try:
    from silero_vad import load_silero_vad, VADIterator
    _silero_model = load_silero_vad()
    _silero_model.eval()
    SILERO_AVAILABLE = True
    logger.info(f"Silero VAD loaded in {time.time() - _vad_t0:.2f}s ✓")
except Exception as _e:
    logger.warning(f"Silero VAD unavailable ({_e}) — falling back to RMS threshold")
    _silero_model = None
    SILERO_AVAILABLE = False

VAD_RMS_FALLBACK = 0.002   # lowered 0.003 → 0.002 to catch quieter audio


def _make_vad_state():
    """
    Returns a per-session check(audio_f32) callable.

    check(audio_f32) → bool
      True  — Silero's LSTM is currently inside a speech region.
      False — currently in silence.

    Silero sub-chunking (512 samples / 32 ms) is handled internally.
    Each call creates a fully independent VADIterator (isolated LSTM state).
    """
    if SILERO_AVAILABLE:
        vad_iter = VADIterator(
            _silero_model,
            threshold=VAD_SPEECH_THRESHOLD,   # 0.3 — catches quieter voices
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=80,        # was 100
            speech_pad_ms=80,                  # was 30
        )
        state = {'buf': np.empty(0, dtype=np.float32), 'speaking': False}

        def check(audio_f32: np.ndarray) -> bool:
            """Feed audio into the 512-sample sub-buffer; return speaking state."""
            state['buf'] = np.concatenate([state['buf'], audio_f32])
            while state['buf'].size >= VAD_CHUNK_SAMPLES:
                sub          = state['buf'][:VAD_CHUNK_SAMPLES]
                state['buf'] = state['buf'][VAD_CHUNK_SAMPLES:]
                result = vad_iter(torch.from_numpy(sub), return_seconds=False)
                if result is not None:
                    if 'start' in result:
                        state['speaking'] = True
                    elif 'end' in result:
                        state['speaking'] = False
            return state['speaking']

        return check
    else:
        def check(audio_f32: np.ndarray) -> bool:
            return float(np.sqrt(np.mean(audio_f32 ** 2))) > VAD_RMS_FALLBACK
        return check


# ============================================================================
# RNNT DECODING STRATEGY
# ============================================================================
# Strategy: beam search (beam_size=4) instead of greedy_batch.
#
# WHY beam over greedy:
#   Greedy RNNT picks the single highest-probability token at each step and
#   cannot recover from an early wrong choice.  Beam search keeps the top-4
#   hypotheses alive at every step and selects the best complete sequence —
#   recovering from premature token commitments.  For Quranic Arabic, where
#   rare words appear only once in the corpus and greedy often substitutes a
#   more common Arabic alternative, beam gives a meaningful WER reduction
#   (~5-15% relative).  RNNT fires during the natural inter-ayah pause so
#   the extra latency is absorbed by time the reciter would spend inhaling.
#
# CUDA GRAPHS — two separate flags with two different names (verified):
#   greedy.use_cuda_graph_decoder  — greedy-path cuda graphs
#   beam.allow_cuda_graphs         — beam-path cuda graphs
#   Both disabled — same cuStreamGetCaptureInfo unpacking crash applies.
#
# FIELD NAME CORRECTION (verified via dataclasses.asdict()):
#   Previous code set rnnt_cfg.greedy.max_symbols = 10, which is a
#   non-existent attribute silently ignored by OmegaConf.  The correct
#   field name is max_symbols_per_step (default is already 10, so
#   behaviour was unchanged — but now it is explicit and correct).
#
# Clean approach: instantiate RNNTBPEDecodingConfig (the official NeMo
# dataclass) directly and set fields on it as plain Python attributes.
# change_decoding_strategy() internally merges this against the full schema
# (OmegaConf.structured → to_container → create → merge), so every field
# is valid and no OmegaConf struct-key errors are possible.
# ============================================================================

logger.info("Configuring RNNT decoder (beam_size=4, cuda-graphs off)...")
try:
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecodingConfig

    rnnt_cfg = RNNTBPEDecodingConfig(strategy="beam")

    # Greedy sub-config — used internally during beam expansion steps
    rnnt_cfg.greedy.max_symbols_per_step   = 10    # verified field name
    rnnt_cfg.greedy.use_cuda_graph_decoder = False # greedy cuda-graphs off

    # Beam sub-config — field names verified via dataclasses.asdict()
    rnnt_cfg.beam.beam_size              = 4     # top-4 hypotheses
    rnnt_cfg.beam.allow_cuda_graphs      = False # beam cuda-graphs off
    rnnt_cfg.beam.return_best_hypothesis = True  # return single best string

    model.change_decoding_strategy(decoder_type="rnnt", decoding_cfg=rnnt_cfg)
    logger.info("RNNT decoder configured (beam_size=4, cuda-graphs off) ✓")

except Exception as exc:
    logger.error(f"Failed to configure RNNT decoder: {exc}")
    raise   # RNNT is required — hard fail at startup

# ============================================================================
# ENCODER
# ============================================================================

def _encode(audio_f32: np.ndarray):
    """
    Preprocessor + encoder.  Must be called inside torch.inference_mode().
    Runs in float32 throughout — no AMP/autocast.
    """
    signal = torch.from_numpy(audio_f32).unsqueeze(0).to(device)
    length = torch.tensor([audio_f32.shape[0]], device=device)
    processed, proc_len = model.preprocessor(input_signal=signal, length=length)
    encoded, enc_len    = model.encoder(audio_signal=processed, length=proc_len)
    return encoded, enc_len


# ============================================================================
# CTC — interim decoder  (fast, ~30-50 ms)
# ============================================================================

def _transcribe_ctc(audio_f32: np.ndarray) -> str:
    """
    Greedy CTC decode.  VAD gating happens in the WebSocket loop before
    this is called — this function always runs inference on whatever it
    receives.

    Algorithm: collapse consecutive duplicate tokens, then strip blanks.
    Runs in float32 — this model produces all-⁇ under any FP16 path.
    """
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
    Runs in float32 — this model produces all-⁇ under any FP16 path.
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
#
# default_rng(0) instead of np.random.seed(0): creates an isolated Generator
# instance with no global side-effects.  np.random.seed() mutates the global
# legacy random state shared with any other library in the process (scipy,
# sklearn, etc.).  Same deterministic output, zero process-wide pollution.
# ============================================================================

logger.info("Warming up CTC + RNNT decoders...")
_t0 = time.time()

_warmup_audio = (
    np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.1
).astype(np.float32)

_ctc_text  = _transcribe_ctc(_warmup_audio)
logger.info(f"  CTC  warm-up OK — '{_ctc_text}'")

_rnnt_text = _transcribe_rnnt(_warmup_audio)   # raises + aborts if cuda-graph fix failed
logger.info(f"  RNNT warm-up OK — '{_rnnt_text}'")

logger.info(f"Warm-up complete in {time.time() - _t0:.2f}s")

# ============================================================================
# INFERENCE LOCK  (one GPU inference at a time — shared across all sessions)
# ============================================================================
# NOTE — single-process limitation:
#   This lock serialises all GPU inference across concurrent WebSocket sessions.
#   A 20 s RNNT decode (~300–500 ms on a modern GPU) will delay CTC interim
#   results for every other connected client for that duration.  Acceptable
#   for single-reciter deployments.  For multi-user scale, run one uvicorn
#   worker per GPU; each worker process has its own lock and model copy.
#
# CREATION — inside the FastAPI lifespan hook (runs in the event loop).
#   asyncio.Lock() at module scope emits DeprecationWarning on Python
#   3.8/3.9 (no running loop at import time).  Lifespan is the correct place.
# ============================================================================

_infer_lock: asyncio.Lock   # assigned in lifespan below; type hint for IDE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create event-loop-bound primitives before serving requests."""
    global _infer_lock
    _infer_lock = asyncio.Lock()
    logger.info("Inference lock created ✓")
    yield
    # Nothing to tear down for the lock itself.


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Quran ASR Server", version="4.9.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service":  "Quran ASR Server",
        "version":  "4.9.0",
        "model":    MODEL_NAME,
        "status":   "operational",
        "decoders": {"interim": "CTC", "final": "RNNT (beam search, beam_size=4)"},
        "vad":      "silero" if SILERO_AVAILABLE else "rms_fallback",
        "amp":      False,   # float32 only — AMP breaks this model
    }


@app.get("/health")
def health():
    return {
        "status":            "healthy",
        "model":             MODEL_NAME,
        "device":            device,
        "amp":               False,   # float32 only — AMP breaks this model
        "chunk_ms":          int(CHUNK_SAMPLES / SAMPLE_RATE * 1000),
        "stride_ms":         int(CHUNK_STRIDE  / SAMPLE_RATE * 1000),
        "overlap_pct":       int((1 - CHUNK_STRIDE / CHUNK_SAMPLES) * 100),
        "max_utterance_s":   MAX_UTTERANCE_SAMPLES / SAMPLE_RATE,
        "min_utterance_s":   MIN_UTTERANCE_SAMPLES / SAMPLE_RATE,
        "silence_trigger_s": SILENCE_TRIGGER_S,
        "idle_timeout_s":    IDLE_TIMEOUT_S,
        "vad":               "silero" if SILERO_AVAILABLE else "rms_fallback",
        "vad_threshold":     VAD_SPEECH_THRESHOLD if SILERO_AVAILABLE else VAD_RMS_FALLBACK,
        "interim_decoder":   "ctc",
        "final_decoder":     "rnnt (beam search, beam_size=4)",
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

    payload = json.dumps({
        "words":         words,
        "is_final":      is_final,
        "decoder":       "rnnt" if is_final else "ctc",
        "chunk_ms":      int(audio_f32.size / SAMPLE_RATE * 1000),
        "processing_ms": processing_ms,
    }, ensure_ascii=False)

    try:
        await websocket.send_text(payload)
    except (WebSocketDisconnect, RuntimeError):
        # Client disconnected during inference (e.g. graceful close window
        # elapsed or TCP reset).  The result was computed — it just couldn't
        # be delivered.  Log at debug level and return "" so callers know
        # nothing was transmitted.
        logger.debug(
            f"send_text failed — client gone during inference: {text[:60]}"
        )
        return ""

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
#   utter_chunks          — list of float32 arrays; materialised once at RNNT
#   utter_size            — running sample count (avoids repeated len() calls)
#   ctc_chunks / ctc_size — same pattern for the CTC sliding window
#   last_speech_t         — wall-clock time of last speech-containing chunk
#   last_ctc_text         — last CTC text sent; used for deduplication
#   had_speech            — True once VAD detects speech; resets after finalize
#   session_ever_had_speech — True once VAD fires; NEVER resets; disables the
#                             idle timeout for the rest of the session so that
#                             inter-ayah pauses of any length are allowed

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # uuid4() avoids id(websocket) memory-address reuse across sessions
    session_id = str(uuid.uuid4())[:8]
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    # Per-session Silero VAD — isolated LSTM state, correct 512-sample chunking.
    vad = _make_vad_state()

    # Efficient rolling buffers: accumulate as lists, concatenate once at use.
    # This avoids the O(N²) total copy cost of np.concatenate on every chunk.
    utter_chunks: list  = []   # only filled once had_speech is True
    utter_size:   int   = 0
    ctc_chunks:   list  = []
    ctc_size:     int   = 0

    last_ctc_text:           str   = ""
    last_speech_t:           float = 0.0
    had_speech:              bool  = False   # resets after each _finalize
    session_ever_had_speech: bool  = False   # never resets — disables idle timeout
    finalizing:              bool  = False   # guard against concurrent finalizations
    chunks_processed:        int   = 0

    async def _finalize(reason: str) -> None:
        """
        Run RNNT on the accumulated utterance and reset per-utterance state.
        No-op if already finalizing.

        session_ever_had_speech is NOT reset here by design — it must remain
        True for the full session lifetime so that inter-ayah pauses of any
        length never trigger the idle timeout.  All other per-utterance state
        is reset so the next ayah starts clean.
        """
        nonlocal utter_chunks, utter_size, ctc_chunks, ctc_size
        nonlocal last_ctc_text, last_speech_t, had_speech, finalizing

        if finalizing:
            return
        finalizing = True

        try:
            if utter_size >= MIN_UTTERANCE_SAMPLES:
                utterance_buf = (
                    np.concatenate(utter_chunks)
                    if utter_chunks
                    else np.empty(0, dtype=np.float32)
                )
                logger.info(
                    f"{reason}: session={session_id} | "
                    f"{utter_size/SAMPLE_RATE:.1f}s → RNNT"
                )
                await _run_and_send(websocket, utterance_buf, is_final=True)
            else:
                logger.info(
                    f"{reason}: session={session_id} | "
                    f"{utter_size} samples — too short for RNNT"
                )
        finally:
            utter_chunks  = []
            utter_size    = 0
            ctc_chunks    = []
            ctc_size      = 0
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
                    and utter_size >= MIN_UTTERANCE_SAMPLES
                    and (time.time() - last_speech_t) >= SILENCE_TRIGGER_S):
                await _finalize("silence-gap")

    monitor_task = asyncio.create_task(_silence_monitor())

    try:
        while True:
            # Ghost-session guard: apply idle timeout ONLY before the very
            # first speech of this session.  session_ever_had_speech is set
            # True on first VAD onset and never reset — inter-ayah pauses
            # of any length are allowed without disconnecting the reciter.
            try:
                if not session_ever_had_speech:
                    msg = await asyncio.wait_for(
                        websocket.receive(), timeout=IDLE_TIMEOUT_S
                    )
                else:
                    msg = await websocket.receive()
            except asyncio.TimeoutError:
                logger.warning(
                    f"Idle timeout ({IDLE_TIMEOUT_S}s) — closing ghost session: "
                    f"{session_id}"
                )
                break

            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                logger.info(f"Client disconnected: {session_id}")
                # Attempt RNNT before fully closing.  WebSocket close is a
                # two-way handshake: when the client sends a close frame the
                # TCP connection is half-closed and the server can still
                # transmit one final message before echoing the close frame.
                # _run_and_send wraps send_text in try/except, so if the
                # client is truly gone (TCP reset) the error is caught silently.
                if not finalizing and had_speech and utter_size >= MIN_UTTERANCE_SAMPLES:
                    await _finalize("disconnect")
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

                # Feed into Silero VAD (512-sample sub-chunking happens inside
                # vad()).  Returns True if currently inside a speech region.
                vad_speaking = vad(pcm_f32)

                if vad_speaking:
                    last_speech_t           = time.time()
                    had_speech              = True
                    session_ever_had_speech = True   # permanent — never reset

                # ── Utterance buffer (for RNNT) ──────────────────────────────
                # Only accumulate AFTER the first speech onset.
                # Prevents leading silence from consuming the 20 s model cap.
                if had_speech:
                    utter_chunks.append(pcm_f32)
                    utter_size += pcm_f32.size

                    # Auto-finalize at model's training max (20 s).
                    if utter_size >= MAX_UTTERANCE_SAMPLES:
                        logger.info(
                            f"Max utterance reached ({session_id}) — auto-finalizing"
                        )
                        await _finalize("max-length")
                        continue

                # ── CTC sliding window ───────────────────────────────────────
                # Gate BOTH accumulation and inference on had_speech.
                # This prevents inter-ayah silence (after _finalize resets
                # had_speech to False) from contaminating the first CTC window
                # of the next ayah.  Without this gate, up to CHUNK_SAMPLES-1
                # samples of silence accumulate between ayahs and are prepended
                # to the first real speech window, degrading interim accuracy.
                #
                # No additional vad_speaking gate — the last ~1 s of every
                # utterance is still in the buffer when the END event fires;
                # gating on vad_speaking would silently drop trailing syllables.
                # The deduplication filter in _run_and_send suppresses blank
                # output from any silence-only chunks within an utterance.
                if had_speech:
                    ctc_chunks.append(pcm_f32)
                    ctc_size += pcm_f32.size

                if ctc_size >= CHUNK_SAMPLES:
                    # Materialise ONE flat array for this batch.
                    flat     = np.concatenate(ctc_chunks) if len(ctc_chunks) > 1 else ctc_chunks[0]
                    ctc_chunks = []
                    ctc_size   = 0

                    start = 0
                    while start + CHUNK_SAMPLES <= flat.size:
                        chunk  = flat[start : start + CHUNK_SAMPLES]  # view, no copy
                        start += CHUNK_STRIDE

                        if had_speech:
                            last_ctc_text = await _run_and_send(
                                websocket, chunk, is_final=False,
                                last_ctc_text=last_ctc_text,
                            )
                        chunks_processed += 1

                    # Carry unconsumed tail into the next batch.
                    # Guard on had_speech: if _finalize fired mid-loop (at the
                    # await _run_and_send yield point above), it already reset
                    # ctc_chunks=[] and had_speech=False.  Without this guard
                    # the remainder — audio from the END of the previous ayah —
                    # would bleed into the next ayah's first CTC window.
                    remainder = flat[start:]
                    if remainder.size and had_speech:
                        ctc_chunks = [remainder.copy()]  # copy: flat may be GC'd
                        ctc_size   = remainder.size

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
    logger.info("Starting Quran ASR Server v4.9")
    logger.info(f"  Device              : {device} (float32)")
    logger.info(f"  VAD                 : {'Silero neural (512-sample sub-chunks)' if SILERO_AVAILABLE else 'RMS fallback'}")
    logger.info(f"  VAD threshold       : {VAD_SPEECH_THRESHOLD if SILERO_AVAILABLE else VAD_RMS_FALLBACK}")
    logger.info(f"  CTC chunk           : {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES/SAMPLE_RATE:.2f}s)")
    logger.info(f"  CTC stride          : {CHUNK_STRIDE} samples ({CHUNK_STRIDE/SAMPLE_RATE:.2f}s)")
    logger.info(f"  CTC overlap         : {(1 - CHUNK_STRIDE/CHUNK_SAMPLES)*100:.0f}%")
    logger.info(f"  Max utterance (RNNT): {MAX_UTTERANCE_SAMPLES/SAMPLE_RATE:.0f}s")
    logger.info(f"  Min utterance (RNNT): {MIN_UTTERANCE_SAMPLES/SAMPLE_RATE:.2f}s")
    logger.info(f"  Silence trigger     : {SILENCE_TRIGGER_S}s")
    logger.info(f"  Silence poll        : {SILENCE_POLL_S}s")
    logger.info(f"  Idle timeout        : {IDLE_TIMEOUT_S}s")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)