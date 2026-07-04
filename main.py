from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.detect_api import detect_router

app = FastAPI()


app.include_router(detect_router, prefix="/api")