import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictor import Translator

translator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator
    translator = Translator(checkpoint=os.getenv("MODEL_CHECKPOINT"), device=os.getenv("DEVICE"))
    yield


app = FastAPI(title="Transformer Translator", version="1.0.0", lifespan=lifespan)


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class TranslationResponse(BaseModel):
    translation: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": translator is not None}


@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
        return TranslationResponse(translation=translator.translate(request.text))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
