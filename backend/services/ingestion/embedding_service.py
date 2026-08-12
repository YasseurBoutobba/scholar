import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    def encode_dense(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        return [list(map(float, v)) for v in vectors]  # type: ignore[arg-type]

    def encode_single(self, text: str) -> list[float]:
        return self.encode_dense([text])[0]

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
