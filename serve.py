import modal

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
        "nvidia-cublas-cu12",   # ← fixes libcublas.so.12 not found
        "nvidia-cudnn-cu12",    # ← cuDNN for CTranslate2 GPU backend
        "fastapi[standard]",
    )
    .run_function(download_model)  # bake model weights into the image
)


@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=300,   # ← containers linger 5 min after last request before scaling to 0
)
@modal.concurrent(max_inputs=4)
class NourEngine:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",  # pure FP16 → zero quantization loss, T4 has native FP16 Tensor Cores
        )

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, data: dict):
        import base64, os, tempfile

        audio_bytes = base64.b64decode(data["audio_base64"])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name

        try:
            segments, info = self.model.transcribe(
                path,
                language="ar",
                beam_size=5,                        # beam search → best accuracy for Quran diacritics
                vad_filter=True,
                without_timestamps=True,            # skip timestamp alignment
                condition_on_previous_text=False,    # prevent hallucination loops
            )
            text = " ".join(s.text.strip() for s in segments)
        finally:
            os.remove(path)

        return {"text": text, "language": info.language, "duration": info.duration}
