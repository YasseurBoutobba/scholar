from dataclasses import dataclass

from backend.repositories.qdrant_repository import QdrantRepository
from backend.services.agents.reranker_service import RerankerService
from backend.services.ingestion.embedding_service import EmbeddingService
from backend.services.ingestion.sparse_indexer_service import SparseIndexerService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

_RERANK_CANDIDATE_MULTIPLIER = 3
_RERANK_CANDIDATE_CAP = 40


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    section_label: str
    score: float
    retrieval_strategy: str


class RetrievalService:
    def __init__(
        self,
        qdrant: QdrantRepository | None = None,
        embedding: EmbeddingService | None = None,
        sparse: SparseIndexerService | None = None,
        reranker: RerankerService | None = None,
    ) -> None:
        self._qdrant = qdrant or QdrantRepository()
        self._embedding = embedding or EmbeddingService()
        self._sparse = sparse or SparseIndexerService()
        self._reranker = reranker or RerankerService()

    def retrieve(self, query: str, limit: int = 10) -> list[RetrievedChunk]:
        dense_vector = self._embedding.encode_single(query)
        sparse_vector = self._sparse.encode_single(query)

        candidate_limit = min(limit * _RERANK_CANDIDATE_MULTIPLIER, _RERANK_CANDIDATE_CAP)
        results = self._qdrant.search_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=candidate_limit,
        )

        chunks: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for point in results:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            if text in seen_texts:
                continue
            seen_texts.add(text)
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    paper_id=payload.get("paper_id", ""),
                    title=payload.get("title", "Unknown"),
                    page=int(payload.get("page", 1)),
                    text=text,
                    section_label=payload.get("section_label", "Unknown"),
                    score=point.score,
                    retrieval_strategy="hybrid",
                )
            )

        reranked = self._reranker.rerank(query, chunks, top_k=limit)
        logger.info(
            f"Retrieved {len(chunks)} candidates, reranked to {len(reranked)} chunks for query"
        )
        return reranked
