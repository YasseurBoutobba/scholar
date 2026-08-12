from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.exceptions import NotFoundError
from backend.models.base import get_session
from backend.models.schemas import (
    AuditEntryResponse,
    ContradictionResponse,
    ReportResponse,
    ReportSection,
    RunDetailResponse,
    RunResponse,
    RunsListResponse,
    Source,
)
from backend.repositories.postgres_repository import PostgresRepository

router = APIRouter(tags=["runs"])


def _get_postgres(session: AsyncSession = Depends(get_session)) -> PostgresRepository:
    return PostgresRepository(session)


@router.get("/runs", response_model=RunsListResponse)
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    postgres: PostgresRepository = Depends(_get_postgres),
):
    runs = await postgres.list_runs(limit=limit, offset=offset)
    return RunsListResponse(
        runs=[
            RunResponse(
                trace_id=run.trace_id,
                question=run.question,
                classification=run.classification,
                status=run.status,
                created_at=run.created_at,
                completed_at=run.completed_at,
            )
            for run in runs
        ]
    )


@router.get("/runs/{trace_id}", response_model=RunDetailResponse)
async def get_run(
    trace_id: str,
    postgres: PostgresRepository = Depends(_get_postgres),
):
    run = await postgres.get_run_by_trace_id(trace_id)
    if run is None:
        raise NotFoundError(message=f"Run {trace_id} not found")

    audit_logs = await postgres.get_audit_log(trace_id)
    report_row = await postgres.get_report_by_run_id(run.id)

    report = None
    if report_row is not None:
        report = ReportResponse(
            run_id=str(run.id),
            executive_summary=report_row.executive_summary,
            sections=[ReportSection.model_validate(s) for s in report_row.sections],
            contradictions=[
                ContradictionResponse.model_validate(c) for c in report_row.contradictions
            ],
            confidence_scores=report_row.confidence_scores,
            source_list=[Source.model_validate(s) for s in report_row.source_list],
        )

    return RunDetailResponse(
        run=RunResponse(
            trace_id=trace_id,
            question=run.question,
            classification=run.classification,
            status=run.status,
            created_at=run.created_at,
            completed_at=run.completed_at,
        ),
        audit_log=[
            AuditEntryResponse(
                phase=log.phase,
                event=log.event,
                payload=log.payload,
                timestamp=log.timestamp,
            )
            for log in audit_logs
        ],
        report=report,
    )
