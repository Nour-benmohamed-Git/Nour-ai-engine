import modal

app = modal.App("nour-engine")

# ---------------------------------------------------------------------------
# Shared transcription parameters
# ---------------------------------------------------------------------------
WHISPER_MODEL = "large-v3-turbo"
VAD_PARAMS = dict(
    threshold=0.3,              # lower than default 0.5 — catch quieter speech
    min_speech_duration_ms=100,  # lower than default 250 — keep short utterances
    min_silence_duration_ms=300, # lower than default 2000 — don't merge across pauses
)


def download_model():
    """Pre-download model during image build → baked into the image layer.
    Eliminates the ~60s model download on every cold start."""
    from faster_whisper import WhisperModel

    WhisperModel(WHISPER_MODEL, device="cpu", compute_type="auto")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # robust audio decoding for all formats
    .pip_install(
        "faster-whisper",
        "nvidia-cublas-cu12",   # provides libcublas.so.12
        "nvidia-cudnn-cu12",    # provides libcudnn for CTranslate2 GPU backend
        "fastapi[standard]",
        "websockets",
    )
    .env({
        # pip installs .so files into site-packages, but ctranslate2 uses dlopen()
        # which only searches system paths + LD_LIBRARY_PATH
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:"
                           "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
    })
    .run_function(download_model)  # bake model weights into the image
)


# ---------------------------------------------------------------------------
# Helper: write audio bytes to a temp file and return the path
# ---------------------------------------------------------------------------
def _save_temp_audio(audio_bytes: bytes, ext: str = "m4a") -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(audio_bytes)
        return f.name


# ---------------------------------------------------------------------------
# NourEngine — GPU-accelerated Whisper on Modal
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",             # 2× faster FP16 than T4, 24 GB VRAM
    image=image,
    scaledown_window=300,   # containers linger 5 min after last request before scaling to 0
)
@modal.concurrent(max_inputs=4)
class NourEngine:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float16",  # pure FP16, A10G has native FP16 Tensor Cores
        )

    # ------------------------------------------------------------------
    # 1) Original POST endpoint — backward compatible, full text at once
    # ------------------------------------------------------------------
    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, data: dict):
        import base64, os, traceback
        from fastapi.responses import JSONResponse

        try:
            audio_bytes = base64.b64decode(data["audio_base64"])
            ext = data.get("format", "m4a")
            path = _save_temp_audio(audio_bytes, ext)

            try:
                segments, info = self.model.transcribe(
                    path,
                    language="ar",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=VAD_PARAMS,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments)

                # Fallback: if VAD filtered everything out, retry without VAD
                if not text.strip():
                    segments, info = self.model.transcribe(
                        path,
                        language="ar",
                        beam_size=5,
                        vad_filter=False,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                    )
                    text = " ".join(s.text.strip() for s in segments)

            finally:
                os.remove(path)

            return {"text": text, "language": info.language, "duration": info.duration}

        except Exception:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": traceback.format_exc()},
            )

    # ------------------------------------------------------------------
    # 2) SSE streaming POST — words arrive one-by-one as they decode
    #    Client sends full audio (base64), receives an event stream of
    #    individual words so the frontend can render them instantly.
    #
    #    Event format (text/event-stream):
    #      data: {"word":"كلمة","start":0.42,"end":0.78,"segment_idx":0}
    #      ...
    #      data: {"done":true,"full_text":"...","language":"ar","duration":12.3}
    # ------------------------------------------------------------------
    @modal.fastapi_endpoint(method="POST")
    def transcribe_stream(self, data: dict):
        import base64, os, json, traceback
        from fastapi.responses import StreamingResponse, JSONResponse

        try:
            audio_bytes = base64.b64decode(data["audio_base64"])
            ext = data.get("format", "m4a")
            path = _save_temp_audio(audio_bytes, ext)
        except Exception:
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={"error": traceback.format_exc()},
            )

        def word_generator():
            try:
                segments, info = self.model.transcribe(
                    path,
                    language="ar",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=VAD_PARAMS,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                )

                all_words = []
                seg_idx = 0
                found_speech = False
                for segment in segments:
                    found_speech = True
                    if segment.words:
                        for w in segment.words:
                            word_text = w.word.strip()
                            if word_text:
                                all_words.append(word_text)
                                payload = json.dumps({
                                    "word": word_text,
                                    "start": round(w.start, 3),
                                    "end": round(w.end, 3),
                                    "segment_idx": seg_idx,
                                }, ensure_ascii=False)
                                yield f"data: {payload}\n\n"
                    else:
                        # segment without word-level detail — emit whole text
                        text = segment.text.strip()
                        if text:
                            all_words.append(text)
                            payload = json.dumps({
                                "word": text,
                                "start": round(segment.start, 3),
                                "end": round(segment.end, 3),
                                "segment_idx": seg_idx,
                            }, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    seg_idx += 1

                # Fallback: if VAD filtered everything out, retry without VAD
                if not found_speech or not all_words:
                    segments2, info = self.model.transcribe(
                        path,
                        language="ar",
                        beam_size=5,
                        vad_filter=False,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                    )
                    all_words = []
                    seg_idx = 0
                    for segment in segments2:
                        if segment.words:
                            for w in segment.words:
                                word_text = w.word.strip()
                                if word_text:
                                    all_words.append(word_text)
                                    payload = json.dumps({
                                        "word": word_text,
                                        "start": round(w.start, 3),
                                        "end": round(w.end, 3),
                                        "segment_idx": seg_idx,
                                    }, ensure_ascii=False)
                                    yield f"data: {payload}\n\n"
                        else:
                            text = segment.text.strip()
                            if text:
                                all_words.append(text)
                                payload = json.dumps({
                                    "word": text,
                                    "start": round(segment.start, 3),
                                    "end": round(segment.end, 3),
                                    "segment_idx": seg_idx,
                                }, ensure_ascii=False)
                                yield f"data: {payload}\n\n"
                        seg_idx += 1

                # Final summary event
                done_payload = json.dumps({
                    "done": True,
                    "full_text": " ".join(all_words),
                    "language": info.language,
                    "duration": round(info.duration, 2),
                }, ensure_ascii=False)
                yield f"data: {done_payload}\n\n"

            except Exception:
                traceback.print_exc()
                err_payload = json.dumps({"error": traceback.format_exc()})
                yield f"data: {err_payload}\n\n"
            finally:
                os.remove(path)

        return StreamingResponse(
            word_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    # ------------------------------------------------------------------
    # 3) Health / warm-up endpoint
    #    GET /health — lightweight ping so the client can pre-warm the
    #    container and check readiness before the user starts reciting.
    # ------------------------------------------------------------------
    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok", "model": WHISPER_MODEL}
