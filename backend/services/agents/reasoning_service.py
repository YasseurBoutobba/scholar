from dataclasses import dataclass, field

from backend.services.agents.retrieval_service import RetrievedChunk
from backend.services.llm.llm_service import LLMService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubAnswer:
    sub_question_index: int
    answer_text: str
    source_chunk_ids: list[str]
    source_paper_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0


class ReasoningService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    def reason(self, sub_question: str, chunks: list[RetrievedChunk], sub_index: int) -> SubAnswer:
        if not chunks:
            return SubAnswer(
                sub_question_index=sub_index,
                answer_text="No relevant information found in the document library.",
                source_chunk_ids=[],
                source_paper_ids=[],
            )

        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[CHUNK_{i}] (paper_id={chunk.paper_id}, page={chunk.page})\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""You are a research assistant answering questions based on provided paper excerpts.

Rules:
- Answer ONLY using the excerpts below. If the answer is not in them, say "No relevant information found."
- Every claim MUST reference its source using [CHUNK_N] notation.
- Use chain-of-thought reasoning: show your reasoning step by step.
- Keep the answer concise but thorough (3-8 sentences).

Excerpts:
{context}

Question: {sub_question}

Answer (with [CHUNK_N] citations for every claim):"""

        response = self._llm.complete(
            prompt, model=self._llm.strong_model, temperature=0.3, max_tokens=2000
        )

        answer_text = response.content.strip()
        chunk_ids, paper_ids = self._extract_source_ids(answer_text, chunks)

        return SubAnswer(
            sub_question_index=sub_index,
            answer_text=answer_text,
            source_chunk_ids=chunk_ids,
            source_paper_ids=paper_ids,
        )

    def _extract_source_ids(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[str], list[str]]:
        import re

        pattern = re.compile(r"\[CHUNK_(\d+)\]")
        matches = pattern.findall(answer)

        chunk_ids: list[str] = []
        paper_ids: list[str] = []
        for m in matches:
            idx = int(m)
            if 0 <= idx < len(chunks):
                chunk_ids.append(chunks[idx].chunk_id)
                paper_ids.append(chunks[idx].paper_id)

        return list(dict.fromkeys(chunk_ids)), list(dict.fromkeys(paper_ids))
