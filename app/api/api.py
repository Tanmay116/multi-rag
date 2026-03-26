from fastapi import APIRouter

from app.api.endpoints.add_kb import ingest_router

api_router = APIRouter()

api_router.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
