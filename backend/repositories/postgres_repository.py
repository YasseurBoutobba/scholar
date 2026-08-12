import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.orm import (
    AuditLog,
    Chunk,
    Contradiction,
    Paper,
    Report,
    ResearchRun,
    RetrievedChunk,
    SubAnswer,
    SubQuestion,
)


class PostgresRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_paper(self, **kwargs) -> Paper:
        paper = Paper(**kwargs)
        self._session.add(paper)
        await self._session.commit()
        await self._session.refresh(paper)
        return paper

    async def get_paper(self, paper_id: uuid.UUID) -> Paper | None:
        return await self._session.get(Paper, paper_id)

    async def list_papers(self, limit: int = 50, offset: int = 0) -> list[Paper]:
        result = await self._session.scalars(
            select(Paper).order_by(Paper.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result)

    async def update_paper_status(
        self,
        paper: Paper,
        status: str,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> Paper:
        paper.ingestion_status = status
        if error is not None:
            paper.error = error
        elif status == "ready":
            paper.error = None
        if chunk_count is not None:
            paper.chunk_count = chunk_count
        await self._session.commit()
        await self._session.refresh(paper)
        return paper

    async def find_paper_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        result = await self._session.scalars(select(Paper).where(Paper.arxiv_id == arxiv_id))
        return result.first()

    async def log_audit(
        self,
        trace_id: str,
        phase: str,
        event: str,
        payload: dict | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            trace_id=trace_id,
            phase=phase,
            event=event,
            payload=payload or {},
            timestamp=datetime.now(timezone.utc),
        )
        self._session.add(entry)
        await self._session.commit()
        return entry

    async def get_audit_log(self, trace_id: str) -> list[AuditLog]:
        result = await self._session.scalars(
            select(AuditLog).where(AuditLog.trace_id == trace_id).order_by(AuditLog.timestamp.asc())
        )
        return list(result)

    async def create_chunks(
        self,
        paper_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
        texts: list[str],
        section_labels: list[str],
        token_counts: list[int],
    ) -> None:
        for chunk_id, text, section_label, token_count in zip(
            chunk_ids, texts, section_labels, token_counts, strict=True
        ):
            self._session.add(
                Chunk(
                    id=chunk_id,
                    paper_id=paper_id,
                    section_label=section_label,
                    text=text,
                    embedding_id=str(chunk_id),
                    token_count=token_count,
                )
            )
        await self._session.commit()

    async def create_run(self, trace_id: str, question: str) -> ResearchRun:
        run = ResearchRun(trace_id=trace_id, question=question, status="running")
        self._session.add(run)
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def update_run_status(
        self,
        run: ResearchRun,
        status: str,
        classification: str | None = None,
    ) -> ResearchRun:
        run.status = status
        if classification is not None:
            run.classification = classification
        if status in ("completed", "failed"):
            run.completed_at = datetime.now(timezone.utc)
        await self._session.commit()
        await self._session.refresh(run)
        return run

    async def get_run_by_trace_id(self, trace_id: str) -> ResearchRun | None:
        result = await self._session.scalars(
            select(ResearchRun).where(ResearchRun.trace_id == trace_id)
        )
        return result.first()

    async def list_runs(self, limit: int = 50, offset: int = 0) -> list[ResearchRun]:
        result = await self._session.scalars(
            select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result)

    async def get_report_by_run_id(self, run_id: uuid.UUID) -> Report | None:
        result = await self._session.scalars(select(Report).where(Report.run_id == run_id))
        return result.first()

    async def create_sub_question(self, run_id: uuid.UUID, text: str) -> SubQuestion:
        sq = SubQuestion(run_id=run_id, text=text, status="pending")
        self._session.add(sq)
        await self._session.commit()
        await self._session.refresh(sq)
        return sq

    async def update_sub_question(
        self,
        sub_question: SubQuestion,
        hyde_answer: str | None = None,
        status: str | None = None,
        retry_count: int | None = None,
    ) -> SubQuestion:
        if hyde_answer is not None:
            sub_question.hyde_answer = hyde_answer
        if status is not None:
            sub_question.status = status
        if retry_count is not None:
            sub_question.retry_count = retry_count
        await self._session.commit()
        await self._session.refresh(sub_question)
        return sub_question

    async def create_retrieved_chunks(self, sub_question_id: uuid.UUID, chunks: list) -> None:
        for chunk in chunks:
            self._session.add(
                RetrievedChunk(
                    sub_question_id=sub_question_id,
                    chunk_id=uuid.UUID(chunk.chunk_id),
                    retrieval_strategy=chunk.retrieval_strategy,
                    rrf_score=chunk.score,
                )
            )
        await self._session.commit()

    async def create_sub_answer(self, sub_question_id: uuid.UUID, answer) -> SubAnswer:
        row = SubAnswer(
            sub_question_id=sub_question_id,
            answer_text=answer.answer_text,
            source_chunk_ids=answer.source_chunk_ids,
            source_paper_ids=answer.source_paper_ids,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def update_sub_answer_scores(self, sub_answer: SubAnswer, scores) -> SubAnswer:
        sub_answer.faithfulness_score = scores.faithfulness
        sub_answer.relevance_score = scores.relevance
        sub_answer.context_precision_score = scores.context_precision
        await self._session.commit()
        await self._session.refresh(sub_answer)
        return sub_answer

    async def create_contradiction(self, run_id: uuid.UUID, contradiction) -> Contradiction:
        row = Contradiction(
            run_id=run_id,
            sub_question_ids=contradiction.sub_question_ids,
            description=contradiction.description,
            papers_in_conflict=contradiction.papers_in_conflict,
        )
        self._session.add(row)
        await self._session.commit()
        return row

    async def create_report(self, run_id: uuid.UUID, report) -> Report:
        row = Report(
            run_id=run_id,
            executive_summary=report.executive_summary,
            sections=[
                {
                    "title": s.title,
                    "content": s.content,
                    "source_chunk_ids": s.source_chunk_ids,
                }
                for s in report.sections
            ],
            contradictions=[
                {
                    "description": c.description,
                    "papers_in_conflict": c.papers_in_conflict,
                }
                for c in report.contradictions
            ],
            confidence_scores=report.confidence_scores,
            source_list=report.source_list,
        )
        self._session.add(row)
        await self._session.commit()
        return row
