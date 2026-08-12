import re
from dataclasses import dataclass

from backend.services.agents.contradiction_service import Contradiction
from backend.services.agents.evaluation_service import EvalScores
from backend.services.agents.reasoning_service import SubAnswer
from backend.services.agents.retrieval_service import RetrievedChunk
from backend.services.llm.llm_service import LLMService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReportSection:
    title: str
    content: str
    source_chunk_ids: list[str]


@dataclass
class Report:
    executive_summary: str
    sections: list[ReportSection]
    contradictions: list[Contradiction]
    confidence_scores: dict[str, float]
    source_list: list[dict]


class ReportWriterService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    def generate(
        self,
        question: str,
        sub_answers: list[SubAnswer],
        contradictions: list[Contradiction],
        eval_scores: list[EvalScores],
        chunk_lookup: dict[str, RetrievedChunk] | None = None,
        sub_question_texts: dict[int, str] | None = None,
    ) -> Report:
        chunk_lookup = chunk_lookup or {}
        sub_question_texts = sub_question_texts or {}

        executive_summary = self._synthesize_answer(question, sub_answers)

        sections = [
            ReportSection(
                title=self._finding_title(i, sa, sub_question_texts),
                content=sa.answer_text,
                source_chunk_ids=sa.source_chunk_ids,
            )
            for i, sa in enumerate(sub_answers)
        ]

        confidence_scores = {}
        for i, scores in enumerate(eval_scores):
            confidence_scores[f"section_{i + 1}"] = round(
                (scores.faithfulness + scores.relevance + scores.context_precision) / 3, 2
            )

        seen_chunk_ids: set[str] = set()
        source_list: list[dict] = []
        for sa in sub_answers:
            for chunk_id in sa.source_chunk_ids:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunk = chunk_lookup.get(chunk_id)
                if chunk is not None:
                    source_list.append(
                        {"title": chunk.title, "page": chunk.page, "paper_id": chunk.paper_id}
                    )

        return Report(
            executive_summary=executive_summary,
            sections=sections,
            contradictions=contradictions,
            confidence_scores=confidence_scores,
            source_list=source_list,
        )

    @staticmethod
    def _finding_title(index: int, sa: SubAnswer, sub_question_texts: dict[int, str]) -> str:
        sub_question = sub_question_texts.get(sa.sub_question_index)
        if sub_question:
            return f"Finding {index + 1}: {sub_question}"
        return f"Finding {index + 1}"

    def _synthesize_answer(self, question: str, sub_answers: list[SubAnswer]) -> str:
        if not sub_answers:
            return f"No relevant information was found in the library for: {question}"

        context = "\n\n".join(
            f"- {self._CITATION_RE.sub('', sa.answer_text).strip()}" for sa in sub_answers
        )

        prompt = f"""You answer research questions using only the notes provided below, each
already grounded in cited academic sources.

Write ONE paragraph of 3-6 plain sentences that directly answers the question, as if
you were simply telling a colleague the answer. Do not restate the question, do not
describe your process, and do not label, number, or title any part of your answer —
no "Sentence 1:", no headings, no bullet points. Only state what the notes below
actually support; if they don't answer the question, say so in plain prose.

Example of the expected shape (do not copy its content, only its plain-paragraph form):
"Self-attention computes a representation of a sequence by relating each position to
every other position through learned query, key, and value vectors. It replaces the
sequential processing of recurrent networks, which lets it be trained far more in
parallel and lets it connect distant positions in a constant number of steps."

Question: {question}

Notes:
{context}

Answer, as one plain paragraph:"""

        try:
            response = self._llm.complete(
                prompt, model=self._llm.strong_model, temperature=0.3, max_tokens=500
            )
            answer = response.content.strip()
            if answer and not self._looks_malformed(answer):
                return answer
            logger.warning(
                f"Answer synthesis returned malformed output, using fallback: {answer!r}"
            )
        except Exception as e:
            logger.warning(f"Answer synthesis failed, falling back to extractive summary: {e}")

        return self._extractive_summary(question, sub_answers)

    _MALFORMED_PREFIX_RE = re.compile(r"^(sentence|step|note|finding)\s*\d", re.IGNORECASE)

    @classmethod
    def _looks_malformed(cls, answer: str) -> bool:
        return bool(cls._MALFORMED_PREFIX_RE.match(answer)) or len(answer) < 40

    def _extractive_summary(self, question: str, sub_answers: list[SubAnswer]) -> str:
        highlights = []
        for sa in sub_answers[:3]:
            highlight = self._first_substantive_sentence(sa.answer_text)
            if highlight:
                highlights.append(highlight)

        summary = f"Research on: {question}\n\nKey findings:\n"
        for i, h in enumerate(highlights):
            summary += f"{i + 1}. {h}.\n"

        return summary.strip()

    _CITATION_RE = re.compile(r"\[CHUNK_\d+\]")
    _META_PREFIX_RE = re.compile(
        r"^(?:step-by-step reasoning|reasoning|answer|summary)\s*:?\s*$", re.IGNORECASE
    )
    _SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

    @classmethod
    def _first_substantive_sentence(cls, answer_text: str) -> str:
        cleaned = re.sub(r"[ \t]+", " ", cls._CITATION_RE.sub("", answer_text))
        for raw_line in cleaned.splitlines():
            line = raw_line.strip().lstrip("-*#").strip()
            line = re.sub(r"^\d+[.)]\s*", "", line).strip("* ").strip()
            if not line or cls._META_PREFIX_RE.match(line):
                continue
            for sentence in cls._SENTENCE_END_RE.split(line):
                sentence = sentence.strip().rstrip(".").strip()
                if len(sentence) >= 25:
                    return sentence
        flat = cls._CITATION_RE.sub("", answer_text).strip()
        return (flat[:200] + "…") if len(flat) > 200 else flat
