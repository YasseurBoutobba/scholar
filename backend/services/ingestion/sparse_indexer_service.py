from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from backend.config import settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class SparseIndexerService:
    def __init__(self) -> None:
        self._model: SparseTextEmbedding | None = None

    @property
    def model(self) -> SparseTextEmbedding:
        if self._model is None:
            logger.info(f"Loading sparse embedding model: {settings.SPARSE_EMBEDDING_MODEL}")
            self._model = SparseTextEmbedding(settings.SPARSE_EMBEDDING_MODEL)
        return self._model

    def encode_sparse(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        return [
            SparseVector(
                indices=list(map(int, e.indices)),
                values=list(map(float, e.values)),
            )
            for e in self.model.embed(texts)
        ]

    def encode_single(self, text: str) -> SparseVector:
        return self.encode_sparse([text])[0]
