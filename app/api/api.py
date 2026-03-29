from fastapi import APIRouter

from app.api.endpoints.add_kb import ingest_router
from app.api.endpoints.chat import chat_router

api_router = APIRouter()

api_router.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
api_router.include_router(chat_router, prefix="/chat", tags=["Chat"])
