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
PERIODIC_TRANSCRIBE_S = 1.0     # transcribe every 1s for real-time word reveal
MIN_AUDIO_S = 1.0               # minimum audio before first transcription (avoid noise hallucinations)

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
def _save_temp_audio(data: bytes, ext: str = "wav") -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(data)
        return f.name


def _build_wav(pcm: bytes, sr: int = 16000, ch: int = 1, bits: int = 16) -> bytes:
    import struct
    n = len(pcm)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + n, b'WAVE',
        b'fmt ', 16, 1, ch, sr, sr * ch * bits // 8, ch * bits // 8, bits,
        b'data', n,
    )
    return header + pcm


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
    """Transcribe raw PCM audio → list of word strings."""
    import os
    if len(pcm) < sr * 2:  # less than 0.5s of audio
        return []
    wav = _build_wav(pcm, sr)
    path = _save_temp_audio(wav)
    try:
        segs, info = model.transcribe(
            path,
            language="ar",
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=WHISPER_VAD_PARAMS,
            word_timestamps=True,
            condition_on_previous_text=False,
            # --- Anti-hallucination (built-in faster-whisper flags) ---
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=1.0,
            suppress_blank=True,
            suppress_tokens=[-1],
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            # --- Context biasing ---
            initial_prompt=initial_prompt,
            hotwords=initial_prompt,
        )
        words: list[str] = []
        for s in segs:
            if s.no_speech_prob > 0.5:
                continue
            if s.words:
                words.extend(w.word.strip() for w in s.words if w.word.strip())
            elif s.text.strip():
                words.append(s.text.strip())
        # Filter hallucinations
        text = " ".join(words)
        if _is_hallucination(text):
            print(f"[ws] filtered hallucination: {text}")
            return []
        return words
    finally:
        os.remove(path)


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

            # Use expected verse as initial_prompt to bias Whisper toward Quran
            prompt = expected_text if expected_text else None

            # 2) State — sliding window: only transcribe NEW audio
            min_audio_bytes = int(MIN_AUDIO_S * sr * 2)
            pcm = bytearray()           # full accumulated audio
            transcribed_up_to = 0       # byte offset already transcribed
            confirmed_words: list[str] = []  # words from previous transcriptions
            last_transcribe_time = 0.0
            transcribe_task: asyncio.Task | None = None
            ws_open = True

            async def _do_periodic_transcribe(
                new_audio: bytes, snapshot_end: int, is_final: bool
            ):
                """Transcribe only new audio and prepend confirmed words."""
                nonlocal last_transcribe_time, transcribed_up_to, confirmed_words
                try:
                    new_words = await asyncio.to_thread(
                        _transcribe_pcm, whisper, new_audio, sr, prompt
                    )
                    all_words = confirmed_words + new_words
                    if ws_open and all_words:
                        await ws.send_json({"words": all_words, "is_final": is_final})
                    # If we got words, confirm them and advance the window
                    if new_words:
                        confirmed_words = all_words
                        transcribed_up_to = snapshot_end
                    last_transcribe_time = time.time()
                except Exception as e:
                    print(f"[ws] transcribe error: {e}")

            # 3) Receive audio, transcribe periodically (non-blocking)
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
                new_audio_len = len(pcm) - transcribed_up_to
                enough_audio = new_audio_len >= min_audio_bytes
                enough_time = now - last_transcribe_time >= PERIODIC_TRANSCRIBE_S

                if prev_done and enough_audio and enough_time:
                    snapshot_end = len(pcm)
                    new_audio = bytes(pcm[transcribed_up_to:snapshot_end])
                    transcribe_task = asyncio.create_task(
                        _do_periodic_transcribe(new_audio, snapshot_end, False)
                    )

            # 4) Wait for any in-flight periodic transcription to finish
            if transcribe_task and not transcribe_task.done():
                try:
                    await transcribe_task
                except Exception:
                    pass

            # 5) Final transcription on remaining new audio
            remaining = bytes(pcm[transcribed_up_to:])
            if remaining and len(remaining) >= sr * 2:
                final_words = await asyncio.to_thread(
                    _transcribe_pcm, whisper, remaining, sr, prompt
                )
                all_words = confirmed_words + final_words
            else:
                all_words = confirmed_words
            await ws.send_json({"words": all_words, "is_final": True})

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
