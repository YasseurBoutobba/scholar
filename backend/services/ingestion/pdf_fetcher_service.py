import re
import tempfile
from pathlib import Path

import requests

from backend.exceptions import ExternalServiceError, NotFoundError, ValidationError

ARXIV_API = "https://export.arxiv.org/api/query"
_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


class PdfFetcherService:
    def parse_arxiv_id(self, user_input: str) -> str:
        match = _ID_PATTERN.search(user_input.strip())
        if not match:
            raise ValidationError(
                message="Invalid arXiv input. Expected an ID like 2301.12345 or an arxiv.org URL."
            )
        return match.group(1)

    def fetch(self, arxiv_id: str) -> tuple[dict, str]:
        metadata = self._fetch_metadata(arxiv_id)
        pdf_path = self._download_pdf(arxiv_id)
        return metadata, pdf_path

    def _fetch_metadata(self, arxiv_id: str) -> dict:
        try:
            response = requests.get(ARXIV_API, params={"id_list": arxiv_id}, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ExternalServiceError(
                message="Failed to reach the arXiv API, please try again."
            ) from exc

        import feedparser

        feed = feedparser.parse(response.text)
        if not feed.entries:
            raise NotFoundError(message=f"No arXiv paper found for ID '{arxiv_id}'.")

        entry = feed.entries[0]
        return {
            "title": entry.title,
            "authors": [a.name for a in entry.authors],
            "abstract": entry.summary,
            "year": int(entry.published[:4]),
            "arxiv_id": arxiv_id,
        }

    def _download_pdf(self, arxiv_id: str) -> str:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        tmp_dir = Path(tempfile.gettempdir()) / "scholar"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = tmp_dir / f"{arxiv_id}.pdf"

        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ExternalServiceError(
                message=f"Failed to download PDF for arXiv ID '{arxiv_id}'."
            ) from exc

        pdf_path.write_bytes(response.content)
        return str(pdf_path)

    @staticmethod
    def write_temp_pdf(content: bytes, suffix: str = ".pdf") -> str:
        tmp_dir = Path(tempfile.gettempdir()) / "scholar"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        import uuid

        path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(content)
        return str(path)

    @staticmethod
    def cleanup_temp(path: str) -> None:
        try:
            import os

            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
