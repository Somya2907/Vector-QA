"""
routes.py

FastAPI router defining the public REST API endpoints.
All business logic is delegated to the QAPipeline; routes handle only
request/response shaping, authentication, and error translation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Payload for the /ask endpoint."""

    query: str = Field(..., min_length=1, max_length=2048, description="User's question.")
    user_id: str = Field(..., description="Authenticated user identifier.")
    # Additional user attributes used for access-control decisions
    user_attributes: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response returned by the /ask endpoint."""

    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    permitted_chunk_count: int
    total_candidate_count: int


class IngestRequest(BaseModel):
    """Payload for the /ingest endpoint (single document)."""

    doc_id: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    permission_tags: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse, summary="Answer a permission-aware question")
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Run the QA pipeline for an authenticated user's question.

    The pipeline embeds the query, retrieves permitted context, and generates
    a grounded answer. Only chunks the user is authorised to see are used.
    """
    # pipeline: QAPipeline = request.app.state.pipeline
    # qa_request = QARequest(query=body.query, user_context={...})
    # response = pipeline.run(qa_request)
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED, summary="Ingest a new document")
async def ingest(body: IngestRequest, request: Request) -> dict[str, str]:
    """Load, chunk, embed, and store a single document with its permission tags.

    Accepted asynchronously — returns immediately; processing happens in the
    background.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a document from the vector store",
)
async def delete_document(doc_id: str, request: Request) -> None:
    """Delete all chunks associated with a document ID."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")
