import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from backend.config import settings


class QdrantRepository:
    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                cloud_inference=False,
                timeout=30,
            )
        return self._client

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(settings.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=settings.DENSE_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse_vector": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        self.client.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="paper_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def upsert_points(self, points: list[PointStruct]) -> None:
        self.client.upsert(collection_name=settings.COLLECTION_NAME, points=points)

    def delete_by_paper_id(self, paper_id: str | uuid.UUID) -> None:
        points, _ = self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_id", match=models.MatchValue(value=str(paper_id))
                    )
                ]
            ),
            limit=1000,
            with_payload=False,
        )
        if points:
            self.client.delete(
                collection_name=settings.COLLECTION_NAME,
                points_selector=models.PointIdsList(points=[p.id for p in points]),
            )

    def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        limit: int = 6,
    ) -> list[models.ScoredPoint]:
        result = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense_vector", limit=limit * 3),
                models.Prefetch(query=sparse_vector, using="bm25_sparse_vector", limit=limit * 3),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return result.points

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
