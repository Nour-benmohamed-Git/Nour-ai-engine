import modal

app = modal.App("nour-engine")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("faster-whisper")
    .pip_install("fastapi[standard]")
)

@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=30,
)
@modal.concurrent(max_inputs=4)
class NourEngine:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="int8",
        )

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, data: dict):
        import base64, os
        audio_bytes = base64.b64decode(data["audio_base64"])
        path = "/tmp/audio.wav"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        segments, info = self.model.transcribe(
            path, language="ar", beam_size=5,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments)
        os.remove(path)
        return {"text": text, "language": info.language, "duration": info.duration}