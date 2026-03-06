from fastapi import FastAPI, UploadFile, File, HTTPException
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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        def do_transcribe():
            segments, _ = model.transcribe(path, vad_filter=True, beam_size=1)
            return "".join(s.text for s in segments).strip()
        text = await run_in_threadpool(do_transcribe)
        return {"text": text}
    finally:
        try:
            os.remove(path)
        except:
            pass
