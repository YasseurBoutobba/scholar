import asyncio
import uuid
from dataclasses import dataclass

from qdrant_client.models import PointStruct

from backend.config import settings
from backend.exceptions import ConflictError, ValidationError
from backend.models.orm import Paper
from backend.repositories.graph_repository import GraphRepository
from backend.repositories.postgres_repository import PostgresRepository
from backend.repositories.qdrant_repository import QdrantRepository
from backend.repositories.redis_repository import RedisRepository
from backend.services.graph.knowledge_graph_service import KnowledgeGraphService
from backend.services.ingestion.embedding_service import EmbeddingService
from backend.services.ingestion.entity_extraction_service import EntityExtractionService
from backend.services.ingestion.pdf_fetcher_service import PdfFetcherService
from backend.services.ingestion.pdf_parser_service import PDFParserService
from backend.services.ingestion.section_chunker_service import ChunkCandidate, SectionChunkerService
from backend.services.ingestion.sparse_indexer_service import SparseIndexerService
from backend.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_ENTITY_EXTRACTION_CHUNKS = 15


@dataclass
class IngestionResult:
    paper: Paper
    chunks: list[ChunkCandidate]


class IngestionOrchestrator:
    def __init__(
        self,
        postgres: PostgresRepository,
        qdrant: QdrantRepository,
        embedding: EmbeddingService,
        redis: RedisRepository | None = None,
        sparse: SparseIndexerService | None = None,
        pdf_fetcher: PdfFetcherService | None = None,
        pdf_parser: PDFParserService | None = None,
        chunker: SectionChunkerService | None = None,
        entity_extraction: EntityExtractionService | None = None,
        knowledge_graph: KnowledgeGraphService | None = None,
        graph_repo: GraphRepository | None = None,
    ) -> None:
        self._postgres = postgres
        self._qdrant = qdrant
        self._embedding = embedding
        self._redis = redis or RedisRepository()
        self._sparse = sparse or SparseIndexerService()
        self._pdf_fetcher = pdf_fetcher or PdfFetcherService()
        self._pdf_parser = pdf_parser or PDFParserService()
        self._chunker = chunker or SectionChunkerService()
        self._entity_extraction = entity_extraction or EntityExtractionService()
        self._knowledge_graph = knowledge_graph or KnowledgeGraphService()
        self._graph_repo = graph_repo

    async def ingest_pdf(self, content: bytes, filename: str) -> IngestionResult:
        if not content:
            raise ValidationError(message="The uploaded file is empty.")
        if not filename.lower().endswith(".pdf"):
            raise ValidationError(message="Only PDF files are supported.")
        if len(content) > settings.max_upload_bytes:
            raise ValidationError(
                message=f"File exceeds the {settings.MAX_UPLOAD_MB}MB upload limit."
            )

        paper = await self._postgres.create_paper(
            source="upload",
            filename=filename,
            ingestion_status="processing",
        )

        tmp_path = None
        try:
            tmp_path = PdfFetcherService.write_temp_pdf(content)

            pages = await self._to_thread(self._pdf_parser.parse, tmp_path)
            chunks = await self._to_thread(self._chunker.chunk, pages, filename)
            if not chunks:
                raise ValidationError(message="PDF parsing produced no content")

            await self._index_chunks(paper, chunks)
            await self._update_knowledge_graph(paper, chunks)

            paper = await self._postgres.update_paper_status(
                paper, "ready", chunk_count=len(chunks)
            )
            return IngestionResult(paper=paper, chunks=chunks)

        except Exception as exc:
            logger.exception(f"PDF ingestion failed for {filename}")
            await self._postgres.update_paper_status(paper, "failed", error=str(exc))
            raise
        finally:
            if tmp_path:
                await self._to_thread(PdfFetcherService.cleanup_temp, tmp_path)

    async def ingest_arxiv(self, url: str) -> IngestionResult:
        arxiv_id = self._pdf_fetcher.parse_arxiv_id(url)

        existing = await self._postgres.find_paper_by_arxiv_id(arxiv_id)
        if existing is not None and existing.ingestion_status == "processing":
            raise ConflictError(message=f"arXiv paper {arxiv_id} is already being ingested.")
        if existing is not None and existing.ingestion_status == "ready":
            paper = existing
            paper = await self._postgres.update_paper_status(paper, "processing")
        elif existing is None:
            paper = await self._postgres.create_paper(
                source="arxiv_url",
                arxiv_id=arxiv_id,
                ingestion_status="processing",
            )
        else:
            paper = existing
            paper = await self._postgres.update_paper_status(paper, "processing")

        pdf_path = None
        try:
            await self._to_thread(self._qdrant.delete_by_paper_id, paper.id)

            metadata, pdf_path = await self._to_thread(self._pdf_fetcher.fetch, arxiv_id)

            paper.title = metadata["title"]
            paper.authors = metadata["authors"]
            paper.abstract = metadata["abstract"]
            paper.year = metadata["year"]
            await self._postgres._session.commit()

            pages = await self._to_thread(self._pdf_parser.parse, pdf_path)
            chunks = await self._to_thread(self._chunker.chunk, pages)
            if not chunks:
                raise ValidationError(message="PDF parsing produced no content")

            await self._index_chunks(paper, chunks)
            await self._update_knowledge_graph(paper, chunks)

            paper = await self._postgres.update_paper_status(
                paper, "ready", chunk_count=len(chunks)
            )
            return IngestionResult(paper=paper, chunks=chunks)

        except Exception as exc:
            logger.exception(f"arXiv ingestion failed for {arxiv_id}")
            await self._postgres.update_paper_status(paper, "failed", error=str(exc))
            raise
        finally:
            if pdf_path:
                await self._to_thread(PdfFetcherService.cleanup_temp, pdf_path)

    async def _index_chunks(self, paper: Paper, chunks: list[ChunkCandidate]) -> None:
        texts = [c.text for c in chunks]
        chunk_ids = [uuid.uuid4() for _ in chunks]

        dense_vectors = await self._encode_dense_cached(texts)
        sparse_vectors = await self._to_thread(self._sparse.encode_sparse, texts)

        points = [
            PointStruct(
                id=chunk_ids[i],
                vector={
                    "dense_vector": dense_vectors[i],
                    "bm25_sparse_vector": sparse_vectors[i],
                },
                payload={
                    "paper_id": str(paper.id),
                    "title": chunks[i].title,
                    "page": chunks[i].page,
                    "text": chunks[i].text,
                    "section_label": chunks[i].section_label,
                },
            )
            for i in range(len(chunks))
        ]
        await self._to_thread(self._qdrant.upsert_points, points)

        await self._postgres.create_chunks(
            paper_id=paper.id,
            chunk_ids=chunk_ids,
            texts=texts,
            section_labels=[c.section_label for c in chunks],
            token_counts=[c.token_count for c in chunks],
        )

        logger.info(f"Upserted {len(points)} chunks (dense+sparse) for paper {paper.id}")

    async def _encode_dense_cached(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        hashes = [EmbeddingService.text_hash(t) for t in texts]
        cached = await asyncio.gather(*(self._redis.get_cached_embedding(h) for h in hashes))

        miss_indices = [i for i, v in enumerate(cached) if v is None]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            miss_vectors = await self._to_thread(self._embedding.encode_dense, miss_texts)
            await asyncio.gather(
                *(
                    self._redis.cache_embedding(hashes[i], vec)
                    for i, vec in zip(miss_indices, miss_vectors, strict=True)
                )
            )
            for i, vec in zip(miss_indices, miss_vectors, strict=True):
                cached[i] = vec

        return [v for v in cached if v is not None]

    async def _update_knowledge_graph(self, paper: Paper, chunks: list[ChunkCandidate]) -> None:
        if self._graph_repo is None:
            return

        try:
            sample = chunks[:_MAX_ENTITY_EXTRACTION_CHUNKS]
            results = await asyncio.gather(
                *(self._to_thread(self._entity_extraction.extract, c.text) for c in sample)
            )

            existing = await self._graph_repo.load_graph()
            if existing:
                self._knowledge_graph.load(existing)

            for entities, relations in results:
                if entities or relations:
                    self._knowledge_graph.add_entities(str(paper.id), entities, relations)

            await self._graph_repo.save_graph(self._knowledge_graph.save())

        except Exception:
            logger.exception(f"Knowledge graph update failed for paper {paper.id} (non-fatal)")

    @staticmethod
    async def _to_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
