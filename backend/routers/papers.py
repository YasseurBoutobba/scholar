import uuid

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.exceptions import NotFoundError, ValidationError
from backend.models.base import get_session
from backend.models.schemas import (
    IngestArxivRequest,
    PaperResponse,
    PapersListResponse,
)
from backend.repositories.graph_repository import GraphRepository
from backend.repositories.postgres_repository import PostgresRepository
from backend.repositories.qdrant_repository import QdrantRepository
from backend.services.ingestion.embedding_service import EmbeddingService
from backend.services.ingestion.orchestrator import IngestionOrchestrator

router = APIRouter(tags=["papers"])

_embedding = EmbeddingService()
_qdrant = QdrantRepository()


def _get_postgres(session: AsyncSession = Depends(get_session)) -> PostgresRepository:
    return PostgresRepository(session)


def _get_graph_repo(session: AsyncSession = Depends(get_session)) -> GraphRepository:
    return GraphRepository(session)


def _get_orchestrator(
    postgres: PostgresRepository = Depends(_get_postgres),
    graph_repo: GraphRepository = Depends(_get_graph_repo),
) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        postgres=postgres, qdrant=_qdrant, embedding=_embedding, graph_repo=graph_repo
    )


@router.get("/papers", response_model=PapersListResponse)
async def list_papers(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    postgres: PostgresRepository = Depends(_get_postgres),
):
    papers = await postgres.list_papers(limit=limit, offset=offset)
    return PapersListResponse(
        papers=[PaperResponse.model_validate(p, from_attributes=True) for p in papers]
    )


@router.get("/papers/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: str,
    postgres: PostgresRepository = Depends(_get_postgres),
):
    try:
        uid = uuid.UUID(paper_id)
    except ValueError:
        raise ValidationError(message=f"Invalid paper ID: {paper_id}")

    paper = await postgres.get_paper(uid)
    if paper is None:
        raise NotFoundError(message=f"Paper {paper_id} not found")
    return PaperResponse.model_validate(paper, from_attributes=True)


@router.post("/papers", response_model=PaperResponse)
async def ingest_paper_upload(
    request: Request,
    file: UploadFile = File(...),
    orchestrator: IngestionOrchestrator = Depends(_get_orchestrator),
):
    filename = file.filename or "paper.pdf"
    content_type = (file.content_type or "").lower()
    if not filename.lower().endswith(".pdf") or (
        content_type and content_type != "application/pdf"
    ):
        raise ValidationError(message="Only PDF files are supported.")

    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > settings.max_upload_bytes:
        raise ValidationError(message=f"File exceeds the {settings.MAX_UPLOAD_MB}MB upload limit.")

    content = await file.read()
    result = await orchestrator.ingest_pdf(content, filename)
    return PaperResponse.model_validate(result.paper, from_attributes=True)


@router.post("/papers/arxiv", response_model=PaperResponse)
async def ingest_paper_arxiv(
    body: IngestArxivRequest,
    orchestrator: IngestionOrchestrator = Depends(_get_orchestrator),
):
    result = await orchestrator.ingest_arxiv(body.url)
    return PaperResponse.model_validate(result.paper, from_attributes=True)
