import asyncio

from fastapi import APIRouter

from backend.models.schemas import QueryRequest, QueryResponse, Source
from backend.services.agents.reasoning_service import ReasoningService
from backend.services.agents.retrieval_service import RetrievalService
from backend.services.llm.llm_service import LLMService

router = APIRouter(tags=["search"])

_retrieval = RetrievalService()
_reasoning = ReasoningService(LLMService())


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    chunks = await asyncio.to_thread(_retrieval.retrieve, body.query, body.limit)
    answer = await asyncio.to_thread(_reasoning.reason, body.query, chunks, 0)

    sources = [
        Source(
            title=c.title,
            page=c.page,
            paper_id=c.paper_id,
        )
        for c in chunks
    ]

    return QueryResponse(
        answer=answer.answer_text,
        sources=sources,
        paper_ids=list({s.paper_id for s in sources if s.paper_id}),
    )
