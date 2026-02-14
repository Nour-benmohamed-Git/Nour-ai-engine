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
MIN_AUDIO_S = 0.5               # minimum audio before first transcription


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


def _transcribe_pcm(model, pcm: bytes, sr: int = 16000) -> list[str]:
    """Transcribe raw PCM audio → list of word strings."""
    import os
    if len(pcm) < 3200:
        return []
    wav = _build_wav(pcm, sr)
    path = _save_temp_audio(wav)
    try:
        segs, _ = model.transcribe(
            path, language="ar", beam_size=5,
            vad_filter=True, vad_parameters=WHISPER_VAD_PARAMS,
            word_timestamps=True, condition_on_previous_text=False,
        )
        words: list[str] = []
        found = False
        for s in segs:
            found = True
            if s.words:
                words.extend(w.word.strip() for w in s.words if w.word.strip())
            elif s.text.strip():
                words.append(s.text.strip())
        if not found or not words:
            segs2, _ = model.transcribe(
                path, language="ar", beam_size=5,
                vad_filter=False, word_timestamps=True,
                condition_on_previous_text=False,
            )
            words = []
            for s in segs2:
                if s.words:
                    words.extend(w.word.strip() for w in s.words if w.word.strip())
                elif s.text.strip():
                    words.append(s.text.strip())
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
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": WHISPER_MODEL}

    async def transcribe_async(pcm_bytes: bytes, sr: int) -> list[str]:
        """Run transcription in thread pool to avoid blocking audio receive."""
        return await asyncio.to_thread(_transcribe_pcm, whisper, pcm_bytes, sr)

    @web_app.websocket("/ws")
    async def ws_transcribe(ws: WebSocket):
        await ws.accept()
        try:
            # 1) Config
            config = json.loads(await ws.receive_text())
            sr = config.get("sample_rate", 16000)

            # 2) State
            min_audio_bytes = int(MIN_AUDIO_S * sr * 2)
            pcm = bytearray()
            last_transcribe_time = 0.0
            transcribe_task: asyncio.Task | None = None
            ws_open = True

            async def _do_periodic_transcribe(snapshot: bytes, is_final: bool):
                """Run transcription in background and send result back."""
                nonlocal last_transcribe_time
                try:
                    words = await transcribe_async(snapshot, sr)
                    if ws_open:
                        await ws.send_json({"words": words, "is_final": is_final})
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

            # 5) Final transcription on all accumulated audio
            words = await transcribe_async(bytes(pcm), sr) if pcm else []
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
