from fastapi import FastAPI

from app.api.api import api_router
from app.core.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="My Rag Application",
    description="This RAG application intends to showcase my various skills in RAG",
)


app.include_router(api_router, prefix="/api")

@app.get("health")
async def health_check():
    return {"status": "healthy"}


if __name__=="__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)