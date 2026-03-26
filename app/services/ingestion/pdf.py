import os
import re
import tempfile
from datetime import datetime

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from app.core.logger import get_logger
from app.db.vector_store import update_faiss_index

logger = get_logger("document_ingestion")


def get_safe_temp_path(file: UploadFile) -> tuple[str, str]:
    """Saves file to disk and returns (physical_path, sanitized_name)."""

    temp_path: str | None = None

    if not file.filename:
        logger.error("ingestion_failed_missing_filename")
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    filename: str = file.filename
    raw_name, ext = os.path.splitext(filename)
    clean_name = re.sub(r"[^a-z0-9]", "_", raw_name.lower()) + ext.lower()

    try:
        # Create physical temp file
        fd, temp_path = tempfile.mkstemp(suffix=ext.lower())
        logger.info(
            "creating_temp_file", extra={"upload_name": filename, "temp_path": temp_path}
        )

        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file.file.read())

        return temp_path, clean_name

    except Exception as e:
        logger.exception(
            "temp_file_creation_failed",
            extra={"upload_name": filename, "error": str(e)},
        )

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        raise HTTPException(
            status_code=500, detail="Internal server error during file processing."
        )


def parse_to_documents(
    file_path: str, clean_name: str, description: str
) -> list[Document]:
    """Dispatches the correct loader and injects metadata with error handling."""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        logger.info("parsing_started", extra={"file": clean_name, "type": ext})

        if ext == ".pdf":
            loader = PyMuPDF4LLMLoader(file_path=file_path, mode="page")
        else:
            loader = TextLoader(file_path=file_path, encoding="utf-8")

        docs = loader.load()

        if not docs:
            logger.warning("parsing_yielded_no_content", extra={"file": clean_name})
            return []

        for doc in docs:
            doc.metadata.update(
                {
                    "source": clean_name,
                    "description": description,
                    "file_type": ext,
                    "ingested_at": datetime.now(),
                }
            )

        logger.info(
            "parsing_completed", extra={"chunks": len(docs), "file": clean_name}
        )
        return docs

    except Exception as e:
        logger.error(
            "document_parsing_failed", extra={"file": clean_name, "error": str(e)}
        )
        raise HTTPException(
            status_code=422, detail=f"Could not parse document: {str(e)}"
        )


async def ingest_document_flow(description: str, file: UploadFile):
    """Orchestrates the ingestion flow with global cleanup."""
    temp_path = None
    try:
        # Step 1: Save
        temp_path, clean_name = get_safe_temp_path(file)

        # Step 2: Parse
        documents = parse_to_documents(temp_path, clean_name, description)

        if not documents:
            return "No content found in document."

        # Step 3: Store
        status = update_faiss_index(documents)

        logger.info(
            "ingestion_flow_complete", extra={"file": clean_name, "status": status}
        )
        return status

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("unhandled_ingestion_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    finally:
        # Step 4: Atomic Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug("temp_file_removed", extra={"path": temp_path})
            except OSError as e:
                logger.error(
                    "failed_to_delete_temp_file",
                    extra={"path": temp_path, "error": str(e)},
                )
