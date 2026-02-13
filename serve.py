import modal
from fastapi.responses import JSONResponse

app = modal.App("nour-engine")


def download_model():
    """Pre-download model during image build → baked into the image layer.
    Eliminates the ~60s model download on every cold start."""
    from faster_whisper import WhisperModel

    WhisperModel("large-v3-turbo", device="cpu", compute_type="auto")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "faster-whisper",
        "nvidia-cublas-cu12",   # fixes libcublas.so.12 not found
        "nvidia-cudnn-cu12",    # cuDNN for CTranslate2 GPU backend
        "fastapi[standard]",
    )
    .run_function(download_model)  # bake model weights into the image
)


@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=300,   # containers linger 5 min after last request before scaling to 0
)
@modal.concurrent(max_inputs=4)
class NourEngine:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",  # pure FP16, zero quantization loss, T4 has native FP16 Tensor Cores
        )

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, data: dict):
        import base64, os, tempfile, traceback

        try:
            audio_bytes = base64.b64decode(data["audio_base64"])

            # Use the actual audio format from the client (default m4a = expo-audio output)
            ext = data.get("format", "m4a")
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                f.write(audio_bytes)
                path = f.name

            try:
                # Try with relaxed VAD first (tuned for short Quran recitation clips)
                segments, info = self.model.transcribe(
                    path,
                    language="ar",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.3,              # lower than default 0.5 — catch quieter speech
                        min_speech_duration_ms=100,  # lower than default 250 — keep short utterances
                        min_silence_duration_ms=300,  # lower than default 2000 — don't merge across pauses
                    ),
                    without_timestamps=True,
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
                        without_timestamps=True,
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
