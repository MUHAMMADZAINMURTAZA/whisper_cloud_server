from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from starlette.concurrency import run_in_threadpool
import tempfile, os

app = FastAPI()

model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_bytes = await file.read()
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        def run_model():
            segments, _ = model.transcribe(path, beam_size=1)
            return " ".join([s.text for s in segments]).strip()

        text = await run_in_threadpool(run_model)

        return {"text": text}

    finally:
        if os.path.exists(path):
            os.remove(path)