from dataclasses import dataclass

from backend.services.agents.reasoning_service import SubAnswer
from backend.services.llm.llm_service import LLMService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Contradiction:
    sub_question_ids: list[str]
    description: str
    papers_in_conflict: list[str]


class ContradictionService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    def detect(self, sub_answers: list[SubAnswer]) -> list[Contradiction]:
        if len(sub_answers) < 2:
            return []

        answer_descriptions = []
        for i, sa in enumerate(sub_answers):
            answer_descriptions.append(
                f"Answer {i}: (sources: {', '.join(sa.source_paper_ids[:3])})\n{sa.answer_text}"
            )

        answers_block = "\n\n".join(answer_descriptions)

        prompt = f"""You are a research contradiction detector. Compare the following answers to related
sub-questions and identify any contradictions, disagreements, or conflicting claims.

Rules:
- Only flag genuine contradictions (two sources making incompatible claims about the same thing).
- Do NOT flag differences in scope, perspective, or emphasis — only true factual conflicts.
- Each contradiction must specify which answers are in conflict and what the specific conflict is.

Answers to compare:
{answers_block}

Return a JSON object with:
{{
  "contradictions": [
    {{
      "answer_indices": [0, 1],
      "description": "description of the contradiction",
      "conflict_points": ["specific point of disagreement"]
    }}
  ]
}}

If no contradictions found, return: {{"contradictions": []}}

Return ONLY valid JSON:"""

        try:
            result = self._llm.complete_json(prompt, model=self._llm.strong_model, temperature=0.2)

            contradictions: list[Contradiction] = []
            for c in result.get("contradictions", []):
                indices = c.get("answer_indices", [])
                if len(indices) >= 2:
                    papers = []
                    for idx in indices:
                        if 0 <= idx < len(sub_answers):
                            papers.extend(sub_answers[idx].source_paper_ids[:2])

                    contradictions.append(
                        Contradiction(
                            sub_question_ids=[str(i) for i in indices],
                            description=c.get("description", ""),
                            papers_in_conflict=list(set(papers)),
                        )
                    )

            if contradictions:
                logger.info(f"Detected {len(contradictions)} contradictions")
            return contradictions

        except Exception as e:
            logger.warning(f"Contradiction detection failed: {e}")
            return []
