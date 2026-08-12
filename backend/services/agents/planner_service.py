import json
from dataclasses import dataclass

from backend.services.llm.llm_service import LLMService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubQuestion:
    text: str
    hyde_answer: str = ""
    index: int = 0


class PlannerService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    def decompose(self, question: str) -> list[SubQuestion]:
        prompt = f"""You are a research question decomposition expert.
Given a complex research question, break it down into 3-6 atomic, self-contained sub-questions.
Each sub-question should be answerable independently from a document retrieval system.

Return a JSON object with a "sub_questions" key containing a list of strings.

Question: {question}

Return ONLY valid JSON:"""

        result = self._llm.complete_json(prompt, model=self._llm.fast_model, temperature=0.3)

        raw = result.get("sub_questions", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []

        sub_questions = [
            SubQuestion(text=q, index=i)
            for i, q in enumerate(raw[:6])
            if isinstance(q, str) and q.strip()
        ]

        if not sub_questions:
            sub_questions = [SubQuestion(text=question, index=0)]

        logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
        return sub_questions

    def generate_hyde(self, sub_question: SubQuestion) -> SubQuestion:
        prompt = f"""You are a research assistant. Write a hypothetical, ideal answer to the following
research question as if you had access to the relevant academic papers.
This answer will be used to find similar content in a document store.

Write 2-4 sentences that would be the perfect answer to this question.

Question: {sub_question.text}

Hypothetical answer:"""

        response = self._llm.complete(
            prompt, model=self._llm.fast_model, temperature=0.5, max_tokens=300
        )
        sub_question.hyde_answer = response.content.strip()
        return sub_question

    def rewrite_query(self, sub_question: SubQuestion) -> str:
        prompt = f"""Rewrite the following research question to optimize it for document retrieval.
Make it more specific, use key technical terms, and remove ambiguity.
Keep it under 100 words.

Original: {sub_question.text}

Rewritten:"""

        response = self._llm.complete(
            prompt, model=self._llm.fast_model, temperature=0.3, max_tokens=150
        )
        return response.content.strip()
