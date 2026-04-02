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


class PDFIngestionResult(BaseModel):
    """Detailed breakdown of a successful PDF/TXT ingestion.

    Attributes:
        filename: Sanitized name of the uploaded file.
        chunks: Number of chunks (pages/lines) successfully added to the index.
        description: The user-provided description of the document.
    """

    filename: str = Field(..., description="Sanitized name of the uploaded file.")
    chunks: int = Field(..., description="Number of document chunks created.")
    description: str = Field(..., description="The user-provided description of the document.")
    message: str | None = Field(None, description="Optional extra information or warnings.")


class PDFIngestionResponse(BaseModel):
    """Successful response from the ``POST /pdf`` ingestion endpoint.

    Attributes:
        status: Overall status message.
        result: Detailed breakdown of the ingestion.
    """

    status: str = Field(
        ...,
        description="Overall ingestion status (e.g., 'success').",
    )
    result: PDFIngestionResult = Field(
        ...,
        description="Detailed breakdown of the ingestion process.",
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
        status: Overall status message.
        result: Breakdown of ingested and failed pages.
    """

    status: str = Field(
        ...,
        description="Overall ingestion status (e.g., 'success').",
    )
    result: WebIngestionResult = Field(
        ...,
        description="Detailed breakdown of page ingestion counts.",
    )
