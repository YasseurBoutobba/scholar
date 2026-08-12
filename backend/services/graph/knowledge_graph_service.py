import networkx as nx

from backend.services.ingestion.entity_extraction_service import Entity, EntityRelation
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeGraphService:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_entities(
        self, paper_id: str, entities: list[Entity], relations: list[EntityRelation]
    ) -> None:
        for entity in entities:
            node_id = self._normalize_name(entity.name)
            if self._graph.has_node(node_id):
                self._graph.nodes[node_id]["papers"] = list(
                    set(self._graph.nodes[node_id].get("papers", []) + [paper_id])
                )
            else:
                self._graph.add_node(
                    node_id,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    papers=[paper_id],
                )

        for rel in relations:
            src = self._normalize_name(rel.source)
            tgt = self._normalize_name(rel.target)
            if not self._graph.has_node(src):
                self._graph.add_node(src, entity_type="unknown", papers=[paper_id])
            if not self._graph.has_node(tgt):
                self._graph.add_node(tgt, entity_type="unknown", papers=[paper_id])
            self._graph.add_edge(src, tgt, relation=rel.relation, paper_id=paper_id)

        logger.info(
            f"Added {len(entities)} entities, {len(relations)} relations for paper {paper_id}"
        )

    def save(self) -> dict:
        return nx.node_link_data(self._graph)

    def load(self, data: dict) -> None:
        self._graph = nx.node_link_graph(data)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()
