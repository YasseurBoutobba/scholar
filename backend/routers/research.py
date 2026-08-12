import asyncio
import json

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from backend.config import settings
from backend.models.base import get_session
from backend.models.schemas import ResearchRequest
from backend.repositories.postgres_repository import PostgresRepository
from backend.services.agents.contradiction_service import ContradictionService
from backend.services.agents.evaluation_service import EvaluationService
from backend.services.agents.planner_service import PlannerService
from backend.services.agents.reasoning_service import ReasoningService
from backend.services.agents.report_writer_service import ReportWriterService
from backend.services.agents.retrieval_service import RetrievalService, RetrievedChunk
from backend.services.llm.llm_service import NETWORK_ISSUE_MESSAGE, LLMService, is_network_issue
from backend.utils.logging import generate_trace_id, get_logger, trace_id_var

logger = get_logger(__name__)

router = APIRouter(tags=["research"])

NETWORK_ISSUE_SKIP_THRESHOLD = 2


def _get_postgres(session: AsyncSession = Depends(get_session)) -> PostgresRepository:
    return PostgresRepository(session)


@router.post("/research")
async def research_stream(
    body: ResearchRequest,
    postgres: PostgresRepository = Depends(_get_postgres),
):
    trace_id = generate_trace_id()
    trace_id_var.set(trace_id)

    async def event_generator():
        llm = LLMService()
        retrieval = RetrievalService()
        planner = PlannerService(llm)
        reasoning = ReasoningService(llm)
        contradiction_svc = ContradictionService(llm)
        evaluation = EvaluationService(llm)
        report_writer = ReportWriterService(llm)

        run = None
        try:
            run = await postgres.create_run(trace_id, body.question)
            await postgres.log_audit(
                trace_id, "intake", "question_received", {"question": body.question}
            )

            yield {"event": "phase", "data": json.dumps({"phase": "classify", "status": "running"})}
            lower = body.question.lower()
            if any(w in lower for w in ["compare", "difference", "versus"]):
                classification = "comparative"
            elif any(w in lower for w in ["why", "how", "explain"]):
                classification = "analytical"
            else:
                classification = "factual"
            await postgres.update_run_status(run, "running", classification=classification)
            await postgres.log_audit(trace_id, "classify", "classified", {"type": classification})
            yield {
                "event": "phase",
                "data": json.dumps(
                    {"phase": "classify", "status": "complete", "result": classification}
                ),
            }

            yield {"event": "phase", "data": json.dumps({"phase": "plan", "status": "running"})}
            sub_questions = await asyncio.to_thread(planner.decompose, body.question)
            sub_question_rows = {}
            for sq in sub_questions:
                sq_row = await postgres.create_sub_question(run.id, sq.text)
                sub_question_rows[sq.index] = sq_row
                yield {
                    "event": "sub_question",
                    "data": json.dumps({"index": sq.index, "text": sq.text}),
                }
            await postgres.log_audit(trace_id, "plan", "decomposed", {"count": len(sub_questions)})
            yield {"event": "phase", "data": json.dumps({"phase": "plan", "status": "complete"})}

            yield {"event": "phase", "data": json.dumps({"phase": "retrieve", "status": "running"})}
            all_chunks: dict[int, list[RetrievedChunk]] = {}
            chunk_lookup: dict[str, RetrievedChunk] = {}
            for sq in sub_questions:
                await asyncio.to_thread(planner.generate_hyde, sq)
                await postgres.update_sub_question(
                    sub_question_rows[sq.index], hyde_answer=sq.hyde_answer, status="retrieving"
                )
                query_text = sq.hyde_answer if sq.hyde_answer else sq.text
                chunks = await asyncio.to_thread(retrieval.retrieve, query_text, 10)
                all_chunks[sq.index] = chunks
                for c in chunks:
                    chunk_lookup[c.chunk_id] = c
                await postgres.create_retrieved_chunks(sub_question_rows[sq.index].id, chunks)
            await postgres.log_audit(trace_id, "retrieve", "all_complete", {})
            yield {
                "event": "phase",
                "data": json.dumps({"phase": "retrieve", "status": "complete"}),
            }

            yield {"event": "phase", "data": json.dumps({"phase": "reason", "status": "running"})}
            sub_answers = []
            eval_scores = []
            network_issue_count = 0
            for sq in sub_questions:
                sq_row = sub_question_rows[sq.index]
                chunks = all_chunks.get(sq.index, [])

                if network_issue_count >= NETWORK_ISSUE_SKIP_THRESHOLD:
                    await postgres.update_sub_question(sq_row, status="failed")
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "scope": "sub_question",
                                "index": sq.index,
                                "error": NETWORK_ISSUE_MESSAGE,
                            }
                        ),
                    }
                    continue

                try:
                    await postgres.update_sub_question(sq_row, status="reasoning")

                    best_answer = None
                    best_scores = None
                    attempts = 0
                    while True:
                        answer = await asyncio.to_thread(
                            reasoning.reason, sq.text, chunks, sq.index
                        )
                        scores = await asyncio.to_thread(
                            evaluation.evaluate, sq.text, answer, chunks
                        )
                        if best_scores is None or scores.faithfulness > best_scores.faithfulness:
                            best_answer, best_scores = answer, scores
                        attempts += 1
                        if (
                            not evaluation.should_retry(scores)
                            or attempts >= settings.RETRY_MAX_ATTEMPTS
                        ):
                            break

                    assert best_answer is not None and best_scores is not None
                    answer, scores = best_answer, best_scores

                    await postgres.update_sub_question(
                        sq_row, status="completed", retry_count=attempts - 1
                    )
                    sub_answer_row = await postgres.create_sub_answer(sq_row.id, answer)
                    await postgres.update_sub_answer_scores(sub_answer_row, scores)

                    sub_answers.append(answer)
                    eval_scores.append(scores)

                    yield {
                        "event": "sub_answer",
                        "data": json.dumps(
                            {
                                "index": sq.index,
                                "answer": answer.answer_text[:500],
                                "sources": answer.source_chunk_ids[:5],
                            }
                        ),
                    }
                    yield {
                        "event": "confidence",
                        "data": json.dumps(
                            {
                                "section": sq.index,
                                "faithfulness": round(scores.faithfulness, 2),
                                "relevance": round(scores.relevance, 2),
                                "context_precision": round(scores.context_precision, 2),
                            }
                        ),
                    }

                except Exception as e:
                    logger.exception(f"[{trace_id}] Sub-question {sq.index} failed")
                    if is_network_issue(e):
                        network_issue_count += 1
                        error_message = NETWORK_ISSUE_MESSAGE
                    else:
                        error_message = str(e)
                    await postgres.update_sub_question(sq_row, status="failed")
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "scope": "sub_question",
                                "index": sq.index,
                                "error": error_message,
                            }
                        ),
                    }
                    continue

            await postgres.log_audit(
                trace_id, "reason", "all_complete", {"count": len(sub_answers)}
            )
            yield {"event": "phase", "data": json.dumps({"phase": "reason", "status": "complete"})}
            yield {
                "event": "phase",
                "data": json.dumps({"phase": "evaluate", "status": "complete"}),
            }

            yield {
                "event": "phase",
                "data": json.dumps({"phase": "contradiction", "status": "running"}),
            }
            contradictions = await asyncio.to_thread(contradiction_svc.detect, sub_answers)
            for c in contradictions:
                await postgres.create_contradiction(run.id, c)
                yield {
                    "event": "contradiction",
                    "data": json.dumps(
                        {
                            "description": c.description,
                            "papers": c.papers_in_conflict[:5],
                        }
                    ),
                }
            await postgres.log_audit(
                trace_id, "contradiction", "complete", {"count": len(contradictions)}
            )
            yield {
                "event": "phase",
                "data": json.dumps({"phase": "contradiction", "status": "complete"}),
            }

            yield {"event": "phase", "data": json.dumps({"phase": "report", "status": "running"})}
            sub_question_texts = {sq.index: sq.text for sq in sub_questions}
            report = await asyncio.to_thread(
                report_writer.generate,
                body.question,
                sub_answers,
                contradictions,
                eval_scores,
                chunk_lookup,
                sub_question_texts,
            )
            await postgres.create_report(run.id, report)

            yield {
                "event": "report",
                "data": json.dumps(
                    {
                        "run_id": trace_id,
                        "executive_summary": report.executive_summary,
                        "sections": [
                            {
                                "title": s.title,
                                "content": s.content,
                                "source_chunk_ids": s.source_chunk_ids,
                            }
                            for s in report.sections
                        ],
                        "confidence_scores": report.confidence_scores,
                        "source_list": report.source_list,
                    }
                ),
            }
            await postgres.log_audit(trace_id, "report", "generated", {})
            yield {"event": "phase", "data": json.dumps({"phase": "report", "status": "complete"})}

            await postgres.update_run_status(run, "completed")

            yield {"event": "done", "data": json.dumps({"trace_id": trace_id})}

        except Exception as e:
            logger.exception(f"Research pipeline failed for {trace_id}")
            if run is not None:
                await postgres.update_run_status(run, "failed")
            error_message = NETWORK_ISSUE_MESSAGE if is_network_issue(e) else str(e)
            yield {"event": "error", "data": json.dumps({"scope": "run", "error": error_message})}

    return EventSourceResponse(event_generator())
