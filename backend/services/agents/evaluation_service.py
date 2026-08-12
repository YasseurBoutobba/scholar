from dataclasses import dataclass

from backend.config import settings
from backend.services.agents.reasoning_service import SubAnswer
from backend.services.agents.retrieval_service import RetrievedChunk
from backend.services.llm.llm_service import LLMService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalScores:
    faithfulness: float
    relevance: float
    context_precision: float


class EvaluationService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    def evaluate(
        self, sub_question: str, answer: SubAnswer, chunks: list[RetrievedChunk]
    ) -> EvalScores:
        faithfulness = self._score_faithfulness(answer, chunks)
        relevance = self._score_relevance(sub_question, answer)
        context_precision = self._score_context_precision(sub_question, chunks)

        return EvalScores(
            faithfulness=faithfulness,
            relevance=relevance,
            context_precision=context_precision,
        )

    def _score_faithfulness(self, answer: SubAnswer, chunks: list[RetrievedChunk]) -> float:
        if not chunks or not answer.answer_text:
            return 0.0

        context = "\n\n".join(c.text for c in chunks[:10])

        prompt = f"""Evaluate the faithfulness of this answer to the provided sources.
A faithful answer only makes claims that are directly supported by the sources.

Sources:
{context}

Answer: {answer.answer_text}

Score from 0.0 to 1.0:
- 1.0 = every claim is directly supported by sources
- 0.5 = most claims are supported but some are not
- 0.0 = most claims are unsupported

Return ONLY a number:"""

        response = self._llm.complete(
            prompt, model=self._llm.fast_model, temperature=0.1, max_tokens=10
        )
        try:
            return max(0.0, min(1.0, float(response.content.strip())))
        except ValueError:
            logger.warning(f"Malformed score from LLM: {response.content!r}, defaulting to 0.5")
            return 0.5

    def _score_relevance(self, sub_question: str, answer: SubAnswer) -> float:
        if not answer.answer_text:
            return 0.0

        prompt = f"""Evaluate how well this answer addresses the question.
Score from 0.0 to 1.0:
- 1.0 = directly and completely answers the question
- 0.5 = partially answers the question
- 0.0 = does not address the question at all

Question: {sub_question}
Answer: {answer.answer_text}

Return ONLY a number:"""

        response = self._llm.complete(
            prompt, model=self._llm.fast_model, temperature=0.1, max_tokens=10
        )
        try:
            return max(0.0, min(1.0, float(response.content.strip())))
        except ValueError:
            logger.warning(f"Malformed score from LLM: {response.content!r}, defaulting to 0.5")
            return 0.5

    def _score_context_precision(self, sub_question: str, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0

        texts = [c.text[:300] for c in chunks[:5]]
        context = "\n---\n".join(texts)

        prompt = f"""Evaluate how relevant these retrieved text chunks are to the question.
Score from 0.0 to 1.0:
- 1.0 = most chunks are highly relevant
- 0.5 = some chunks are relevant
- 0.0 = chunks are mostly irrelevant

Question: {sub_question}
Chunks:
{context}

Return ONLY a number:"""

        response = self._llm.complete(
            prompt, model=self._llm.fast_model, temperature=0.1, max_tokens=10
        )
        try:
            return max(0.0, min(1.0, float(response.content.strip())))
        except ValueError:
            logger.warning(f"Malformed score from LLM: {response.content!r}, defaulting to 0.5")
            return 0.5

    def should_retry(self, scores: EvalScores) -> bool:
        return scores.faithfulness < settings.FAITHFULNESS_THRESHOLD
