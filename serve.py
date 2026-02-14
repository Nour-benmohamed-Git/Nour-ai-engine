import modal

app = modal.App("nour-engine")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
WHISPER_MODEL = "large-v3-turbo"
WHISPER_VAD_PARAMS = dict(
    threshold=0.3,
    min_speech_duration_ms=100,
    min_silence_duration_ms=300,
)

# Periodic transcription for real-time feedback (Tarteel-like)
PERIODIC_TRANSCRIBE_S = 0.5     # transcribe every 0.5s for real-time word reveal
MIN_AUDIO_S = 0.3               # minimum audio before first transcription

# Known Whisper hallucination patterns for Arabic
HALLUCINATION_PATTERNS = {
    "اشتركوا في القناة", "شكرا", "شكرا لكم", "السلام عليكم",
    "تابعونا", "اشترك", "لا تنسوا الاشتراك", "مشاهدة ممتعة",
    "تالي", "يكا",
}


def download_model():
    """Pre-download models during image build so they're baked into the layer."""
    from faster_whisper import WhisperModel
    WhisperModel(WHISPER_MODEL, device="cpu", compute_type="auto")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "fastapi[standard]",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:"
                           "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
    })
    .run_function(download_model)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pcm_to_float(pcm: bytes) -> "np.ndarray":
    """Convert raw 16-bit PCM bytes to float32 numpy array (no file I/O)."""
    import numpy as np
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _is_hallucination(text: str) -> bool:
    """Check if transcribed text is a known Whisper hallucination."""
    stripped = text.strip()
    if stripped in HALLUCINATION_PATTERNS:
        return True
    # Detect repetition: same phrase repeated 3+ times
    words = stripped.split()
    if len(words) >= 6:
        half = len(words) // 2
        first_half = " ".join(words[:half])
        second_half = " ".join(words[half:2 * half])
        if first_half == second_half:
            return True
    return False


def _transcribe_pcm(
    model, pcm: bytes, sr: int = 16000, initial_prompt: str | None = None
) -> list[str]:
    """Transcribe raw PCM audio → list of word strings. Zero file I/O."""
    if len(pcm) < sr:  # less than 0.25s of audio (16-bit = 2 bytes/sample)
        return []
    audio = _pcm_to_float(pcm)
    segs, info = model.transcribe(
        audio,
        language="ar",
        beam_size=1,                # greedy decoding — 3-5x faster than beam_size=5
        temperature=0.0,
        vad_filter=True,
        vad_parameters=WHISPER_VAD_PARAMS,
        word_timestamps=False,      # skip timestamp overhead
        without_timestamps=True,    # skip timestamp token generation
        condition_on_previous_text=False,
        # --- Anti-hallucination (built-in faster-whisper flags) ---
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        suppress_blank=True,
        suppress_tokens=[-1],
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        # --- Context biasing ---
        initial_prompt=initial_prompt,
    )
    words: list[str] = []
    for s in segs:
        if s.no_speech_prob > 0.5:
            continue
        text = s.text.strip()
        if text:
            words.extend(w for w in text.split() if w)
    # Filter hallucinations
    joined = " ".join(words)
    if _is_hallucination(joined):
        print(f"[ws] filtered hallucination: {joined}")
        return []
    return words


# ---------------------------------------------------------------------------
# Real-time WebSocket ASGI app
#
# Protocol (all messages are JSON text):
#   → Client sends config:  {"type":"config","sample_rate":16000}
#   → Client streams audio: {"type":"audio","data":"<base64_pcm_16bit_mono>"}
#   → Client sends finish:  {"type":"eof"}
#
#   ← Server sends periodically + on silence:
#     {"words":["بسم","الله","الرحمن"], "is_final":false}
#   ← Server sends final:
#     {"words":[...], "is_final":true}
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    image=image,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def nour_realtime():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from faster_whisper import WhisperModel
    import json, traceback, base64, asyncio, time

    web_app = FastAPI()
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8_float16")

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": WHISPER_MODEL}

    @web_app.websocket("/ws")
    async def ws_transcribe(ws: WebSocket):
        await ws.accept()
        try:
            # 1) Config — client sends expected_text for initial_prompt
            config = json.loads(await ws.receive_text())
            sr = config.get("sample_rate", 16000)
            expected_text = config.get("expected_text", None)

            # Use a generic Quranic prompt to bias Whisper toward Quranic Arabic
            # WITHOUT revealing the specific verse (which causes premature word reveal)
            prompt = "بسم الله الرحمن الرحيم. القرآن الكريم."

            # 2) State — full-buffer transcription for maximum accuracy
            min_audio_bytes = int(MIN_AUDIO_S * sr * 2)
            pcm = bytearray()
            last_transcribe_time = 0.0
            transcribe_task: asyncio.Task | None = None
            ws_open = True

            async def _do_periodic_transcribe(
                snapshot: bytes, is_final: bool
            ):
                """Transcribe full accumulated audio for best context."""
                nonlocal last_transcribe_time
                try:
                    words = await asyncio.to_thread(
                        _transcribe_pcm, whisper, snapshot, sr, prompt
                    )
                    if ws_open and words:
                        await ws.send_json({"words": words, "is_final": is_final})
                    last_transcribe_time = time.time()
                except Exception as e:
                    print(f"[ws] transcribe error: {e}")

            # 3) Receive audio, transcribe full buffer periodically
            while True:
                raw_msg = await ws.receive_text()
                msg = json.loads(raw_msg)

                if msg.get("type") == "eof":
                    break
                if msg.get("type") != "audio":
                    continue

                chunk = base64.b64decode(msg["data"])
                pcm.extend(chunk)

                now = time.time()
                prev_done = transcribe_task is None or transcribe_task.done()
                enough_audio = len(pcm) >= min_audio_bytes
                enough_time = now - last_transcribe_time >= PERIODIC_TRANSCRIBE_S

                if prev_done and enough_audio and enough_time:
                    transcribe_task = asyncio.create_task(
                        _do_periodic_transcribe(bytes(pcm), False)
                    )

            # 4) Wait for any in-flight periodic transcription to finish
            if transcribe_task and not transcribe_task.done():
                try:
                    await transcribe_task
                except Exception:
                    pass

            # 5) Final transcription on full audio
            words = await asyncio.to_thread(
                _transcribe_pcm, whisper, bytes(pcm), sr, prompt
            ) if pcm else []
            await ws.send_json({"words": words, "is_final": True})

        except WebSocketDisconnect:
            pass
        except Exception:
            traceback.print_exc()
            try:
                await ws.send_json({"error": traceback.format_exc()})
            except Exception:
                pass
        finally:
            ws_open = False
            try:
                await ws.close()
            except Exception:
                pass

    return web_app
