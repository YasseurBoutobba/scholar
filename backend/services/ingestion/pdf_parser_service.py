from pathlib import Path

import pymupdf4llm

from backend.exceptions import ValidationError
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class PDFParserService:
    def parse(self, pdf_path: str) -> list[dict]:
        path = Path(pdf_path)
        if not path.exists():
            raise ValidationError(message=f"PDF file not found: {pdf_path}")

        try:
            pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        except Exception as e:
            logger.warning(f"pymupdf4llm failed to parse {pdf_path}: {e}")
            raise ValidationError(message=f"PDF parsing failed: {e}")

        return [
            {"page": p["metadata"]["page_number"], "text": p["text"]}
            for p in pages
            if p.get("text", "").strip()
        ]
