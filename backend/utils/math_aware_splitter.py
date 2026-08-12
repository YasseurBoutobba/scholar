import re


class MathAwareSplitter:
    _MATH_PATTERN = re.compile(r"(\$\$.*?\$\$|\$[^$]+\$|\\\(.*?\\\)|\\\[.*?\\\])")
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'(])")
    _PLACEHOLDER = "\x00MATH{}\x00"

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []

        math_blocks: list[str] = []

        def _replace(match: re.Match) -> str:
            math_blocks.append(match.group(0))
            return f"{self._PLACEHOLDER}{len(math_blocks) - 1}"

        masked = self._MATH_PATTERN.sub(_replace, text)

        raw_sentences = self._SENTENCE_END.split(masked)

        restored: list[str] = []
        for sent in raw_sentences:
            for i, block in enumerate(math_blocks):
                sent = sent.replace(f"{self._PLACEHOLDER}{i}", block)
            restored.append(sent.strip())

        return [s for s in restored if s]
