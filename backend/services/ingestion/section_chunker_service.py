import re
from dataclasses import dataclass

from backend.utils.math_aware_splitter import MathAwareSplitter

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)

_TABLE_GARBAGE_RE = re.compile(
    r"(?:^\|[\s\-:]+\|$\n?)"
    r"|(?:^\|$)"
    r"|(?:^\|[^\n]*$)",
    re.MULTILINE,
)

_SECTION_LABELS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "methods": "Methodology",
    "methodology": "Methodology",
    "approach": "Methodology",
    "model": "Methodology",
    "experiments": "Experiments",
    "results": "Results",
    "evaluation": "Evaluation",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "appendix": "Appendix",
    "references": "References",
    "bibliography": "References",
}


@dataclass
class ChunkCandidate:
    text: str
    page: int
    section_label: str
    token_count: int = 0
    title: str = ""

    def __post_init__(self):
        self.token_count = len(self.text.split())


@dataclass
class ParseResult:
    pages: list[dict]
    title: str = ""
    references_text: str = ""


class SectionChunkerService:
    def __init__(self) -> None:
        self._splitter = MathAwareSplitter()

    def chunk(self, pages: list[dict], filename: str | None = None) -> list[ChunkCandidate]:
        if not pages:
            return []

        all_text = "\n".join(p.get("text", "") for p in pages)
        title = self._extract_title(all_text, filename)

        chunks: list[ChunkCandidate] = []
        hit_references = False
        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page", 1)

            if not text.strip():
                continue

            section_label = self._detect_section_label(text)

            if section_label == "References":
                hit_references = True
            if hit_references:
                continue

            text = self._clean_table_garbage(text)

            for piece in self._split_into_chunks(text):
                chunks.append(
                    ChunkCandidate(
                        text=piece,
                        page=page_num,
                        section_label=section_label,
                        title=title,
                    )
                )

        return chunks

    def _extract_title(self, text: str, filename: str | None = None) -> str:
        for line in text.splitlines()[:50]:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped.lstrip("#").strip().strip("*").strip()
        return filename or "Untitled"

    def _detect_section_label(self, text: str) -> str:
        for match in _HEADING_RE.finditer(text):
            heading = match.group(1).strip().lower()
            for keyword, label in _SECTION_LABELS.items():
                if keyword in heading:
                    return label
        return "Unknown"

    def _clean_table_garbage(self, text: str) -> str:
        lines = text.split("\n")
        cleaned: list[str] = []
        consecutive_table_lines = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                consecutive_table_lines += 1
                if consecutive_table_lines <= 2:
                    cleaned.append(line)
                elif re.match(r"^\|[^|]+\|", stripped) and not re.match(
                    r"^\|[\s\-:]+\|$", stripped
                ):
                    cleaned.append(line)
                else:
                    pass
            else:
                consecutive_table_lines = 0
                cleaned.append(line)

        return "\n".join(cleaned)

    def _split_into_chunks(self, text: str) -> list[str]:
        if len(text) <= CHUNK_SIZE:
            return [text.strip()] if text.strip() else []

        sentences = self._splitter.split(text)

        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= CHUNK_SIZE:
                current = f"{current} {sentence}".strip() if current else sentence
            else:
                if current:
                    pieces.append(current)
                if len(sentence) > CHUNK_SIZE:
                    pieces.append(sentence[:CHUNK_SIZE])
                    current = ""
                else:
                    current = sentence

        if current:
            pieces.append(current)

        if CHUNK_OVERLAP > 0 and len(pieces) > 1:
            overlapped = [pieces[0]]
            for i in range(1, len(pieces)):
                prev_tail = pieces[i - 1][-CHUNK_OVERLAP:]
                overlapped.append(f"{prev_tail} {pieces[i]}".strip())
            return overlapped

        return pieces
