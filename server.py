from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

model = WhisperModel("base", device="cpu")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    segments, info = model.transcribe(temp_path)

    text = ""
    for segment in segments:
        text += segment.text

    return {"text": text}