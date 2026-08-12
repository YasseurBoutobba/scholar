from __future__ import annotations

from typing import TYPE_CHECKING

from backend.config import settings
from backend.utils.logging import get_logger

if TYPE_CHECKING:
    import cohere
    from sentence_transformers import CrossEncoder

    from backend.services.agents.retrieval_service import RetrievedChunk

logger = get_logger(__name__)


class RerankerService:
    def __init__(self) -> None:
        self._cohere_client: cohere.Client | None = None
        self._local_model: CrossEncoder | None = None

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 6
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        if not settings.RERANK_ENABLED:
            return chunks[:top_k]

        if settings.COHERE_API_KEY:
            try:
                return self._rerank_cohere(query, chunks, top_k)
            except Exception as e:
                logger.warning(f"Cohere rerank failed, falling back to local: {e}")

        return self._rerank_local(query, chunks, top_k)

    def _rerank_cohere(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        import cohere

        if self._cohere_client is None:
            self._cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
        documents = [c.text for c in chunks]

        response = self._cohere_client.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model=settings.COHERE_RERANK_MODEL,
        )

        reranked: list[RetrievedChunk] = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.score = result.relevance_score
            reranked.append(chunk)

        logger.info(f"Cohere reranked to {len(reranked)} chunks")
        return reranked

    def _rerank_local(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        try:
            from sentence_transformers import CrossEncoder

            if self._local_model is None:
                self._local_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            pairs = [(query, c.text) for c in chunks]
            scores = self._local_model.predict(pairs)

            scored_chunks = list(zip(chunks, scores, strict=False))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            reranked: list[RetrievedChunk] = []
            for chunk, score in scored_chunks[:top_k]:
                chunk.score = float(score)
                reranked.append(chunk)

            logger.info(f"Local cross-encoder reranked to {len(reranked)} chunks")
            return reranked

        except Exception as e:
            logger.warning(f"Local rerank failed, returning top_k by original score: {e}")
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:top_k]
