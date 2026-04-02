from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from pydantic import AnyHttpUrl

from app.api.schemas.ingest_schemas import PDFIngestionResponse, WebIngestionResponse
from app.core.logger import get_logger
from app.services.ingestion.pdf import ingest_document_flow
from app.services.ingestion.web import ingest_webpage_flow

logger = get_logger("add_kb_router")

ingest_router = APIRouter()


@ingest_router.post("/pdf", response_model=PDFIngestionResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
) -> PDFIngestionResponse:
    """Ingest a PDF or TXT document into the knowledge base.

    Accepts an uploaded file and a human-readable description of the source.
    Only ``.pdf`` and ``.txt`` files are accepted.

    Args:
        file: The uploaded file object provided by FastAPI.
        description: A short description of the document, passed via the
            ``X-Description`` HTTP header.

    Returns:
        A :class:`~app.api.schemas.ingest_schemas.PDFIngestionResponse`
        containing the overall ingestion status and a detailed breakdown
        of the processed file.

    Raises:
        HTTPException 400: If no filename is detected or the file type is not
            supported.
        HTTPException 422: If the document could not be parsed.
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

    logger.info("PDF ingestion request received", extra={"file": filename})

    try:
        result = await ingest_document_flow(description, file)
    except HTTPException:
        raise
    except Exception:
        logger.error(
            "PDF ingestion failed",
            extra={"file": filename},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during document ingestion. Check server logs.",
        )

    logger.info("PDF ingestion succeeded", extra={"file": filename})
    return PDFIngestionResponse(**result)


@ingest_router.post("/web", response_model=WebIngestionResponse)
async def upload_webpage(
    webpage: AnyHttpUrl = Query(..., description="The root URL to crawl and ingest."),
    description: str = Header(..., alias="X-Description"),
) -> WebIngestionResponse:
    """Crawl a webpage and ingest its content into the knowledge base.

    Crawls the given URL (and up to ``max_page`` linked pages) using Crawl4AI,
    chunks the markdown output, and stores the resulting vectors in the FAISS
    index.

    Args:
        webpage: The root URL to crawl, provided as a query parameter.
            Validated as a proper HTTP/HTTPS URL by Pydantic.
        description: A short description of the web source, passed via the
            ``X-Description`` HTTP header.

    Returns:
        A :class:`~app.api.schemas.ingest_schemas.WebIngestionResponse`
        containing the overall status and a detailed breakdown
        of ingested versus failed pages.

    Raises:
        HTTPException 422: If ``webpage`` is not a valid URL (Pydantic validation).
        HTTPException 500: If the ingestion pipeline raises an unexpected error.
    """
    webpage = str(webpage)  # type: ignore

    logger.info("Web ingestion request received", extra={"url": webpage})

    try:
        result = await ingest_webpage_flow(webpage, description)  # type: ignore
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
    return WebIngestionResponse(**result)
