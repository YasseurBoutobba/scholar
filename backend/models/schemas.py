from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class IngestArxivRequest(BaseModel):
    url: str


class PaperResponse(BaseModel):
    id: str
    source: str
    arxiv_id: str | None = None
    title: str
    authors: list[str] = []
    abstract: str = ""
    year: int | None = None
    ingestion_status: str
    filename: str | None = None
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _uuid_to_str(cls, value: UUID | str) -> str:
        return str(value)


class PapersListResponse(BaseModel):
    papers: list[PaperResponse]


class Source(BaseModel):
    title: str
    page: int
    paper_id: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=6, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    paper_ids: list[str] = []


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


class ContradictionResponse(BaseModel):
    description: str
    papers_in_conflict: list[str]


class ReportSection(BaseModel):
    title: str
    content: str
    source_chunk_ids: list[str] = []


class ReportResponse(BaseModel):
    run_id: str
    executive_summary: str
    sections: list[ReportSection]
    contradictions: list[ContradictionResponse]
    confidence_scores: dict[str, float]
    source_list: list[Source]

    @field_validator("run_id", mode="before")
    @classmethod
    def _uuid_to_str(cls, value: UUID | str) -> str:
        return str(value)


class RunResponse(BaseModel):
    trace_id: str
    question: str
    classification: str
    status: str
    created_at: datetime | None = None
    completed_at: datetime | None = None


class AuditEntryResponse(BaseModel):
    phase: str
    event: str
    payload: dict
    timestamp: datetime | None = None


class RunDetailResponse(BaseModel):
    run: RunResponse
    audit_log: list[AuditEntryResponse]
    report: ReportResponse | None = None


class RunsListResponse(BaseModel):
    runs: list[RunResponse]


class ServiceHealth(BaseModel):
    status: str
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, ServiceHealth]
