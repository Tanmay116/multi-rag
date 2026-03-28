from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.services.ingestion.pdf import ingest_document_flow
from app.services.ingestion.web import ingest_webpage_flow
from app.core.logger import get_logger

logger = get_logger("add_kb_router")

ingest_router = APIRouter()


@ingest_router.post("/pdf")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
):
    """Ingest a PDF or TXT document into the knowledge base.

    Accepts an uploaded file and a human-readable description of the source.
    Only ``.pdf`` and ``.txt`` files are accepted.

    Args:
        file: The uploaded file object provided by FastAPI.
        description: A short description of the document, passed via the
            ``X-Description`` HTTP header.

    Returns:
        A JSON object with a ``"status"`` key containing the ingestion result.

    Raises:
        HTTPException 400: If no filename is detected or the file type is not
            supported.
        HTTPException 500: If the ingestion pipeline raises an unexpected error.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename detected.")

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    logger.info("PDF ingestion request received", extra={"filename": filename})

    try:
        result = await ingest_document_flow(description, file)
    except Exception:
        logger.error(
            "PDF ingestion failed",
            extra={"filename": filename},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during document ingestion. Check server logs.",
        )

    logger.info("PDF ingestion succeeded", extra={"filename": filename})
    return {"status": result}


@ingest_router.post("/web")
async def upload_webpage(
    webpage: str,
    description: str = Header(..., alias="X-Description"),
):
    """Crawl a webpage and ingest its content into the knowledge base.

    Crawls the given URL (and up to ``max_page`` linked pages) using Crawl4AI,
    chunks the markdown output, and stores the resulting vectors in the FAISS
    index.

    Args:
        webpage: The root URL to crawl.
        description: A short description of the web source, passed via the
            ``X-Description`` HTTP header.

    Returns:
        A JSON object with a ``"status"`` key containing a summary dict with
        ``"ingested"`` and ``"failed"`` page counts.

    Raises:
        HTTPException 400: If the ``webpage`` parameter is empty.
        HTTPException 500: If the ingestion pipeline raises an unexpected error.
    """
    if not webpage:
        raise HTTPException(status_code=400, detail="No web URL provided.")

    logger.info("Web ingestion request received", extra={"url": webpage})

    try:
        result = await ingest_webpage_flow(webpage, description)
    except RuntimeError as exc:
        logger.error(
            "Web ingestion failed",
            extra={"url": webpage, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        logger.error(
            "Unexpected error during web ingestion",
            extra={"url": webpage},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during web ingestion. Check server logs.",
        )

    logger.info(
        "Web ingestion complete",
        extra={"url": webpage, "result": result},
    )
    return {"status": result}
