from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.detect_api import detect_router
from backend.api.speech_api import speech_router

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(detect_router, prefix="/api")
app.include_router(speech_router, prefix="/api")