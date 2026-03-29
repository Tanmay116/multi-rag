"""Pydantic schemas for the knowledge-base ingestion endpoints.

All request bodies and response models used by ``add_kb`` routes are
defined here so that the endpoint module stays focused on request
handling and keeps schema logic in one canonical location.
"""

from typing import Any, Dict, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PDFIngestionResponse(BaseModel):
    """Successful response from the ``POST /pdf`` ingestion endpoint.

    Attributes:
        status: The ingestion result returned by the pipeline.  The exact
            shape depends on ``ingest_document_flow`` but is typically a
            string summary or a structured dict.
    """

    status: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Ingestion result returned by the document pipeline.",
    )


class WebIngestionResult(BaseModel):
    """Breakdown of a completed web-crawl ingestion run.

    Attributes:
        ingested: Number of pages successfully ingested.
        failed: Number of pages that could not be ingested.
    """

    ingested: int = Field(..., description="Number of pages successfully ingested.")
    failed: int = Field(..., description="Number of pages that failed ingestion.")


class WebIngestionResponse(BaseModel):
    """Successful response from the ``POST /web`` ingestion endpoint.

    Attributes:
        status: A summary dict containing ``ingested`` and ``failed`` page counts.
    """

    status: Union[WebIngestionResult, Dict[str, Any]] = Field(
        ...,
        description="Ingestion summary returned by the web crawl pipeline.",
    )
