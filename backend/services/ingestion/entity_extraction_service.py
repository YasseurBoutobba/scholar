from dataclasses import dataclass

from backend.services.llm.llm_service import LLMService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str = ""


@dataclass
class EntityRelation:
    source: str
    target: str
    relation: str


class EntityExtractionService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or LLMService()

    def extract(self, text: str) -> tuple[list[Entity], list[EntityRelation]]:
        if len(text.strip()) < 50:
            return [], []

        prompt = f"""Extract the key entities and their relationships from this academic text.
Focus on: methods, datasets, concepts, metrics, and people mentioned.

Return a JSON object with:
- "entities": list of {{"name": str, "type": str, "description": str}}
- "relations": list of {{"source": str, "target": str, "relation": str}}

Types can be: method, dataset, concept, person, metric, model
Relations can be: uses, proves, compares, extends, related_to, authored_by

Text:
{text[:2000]}

Return ONLY valid JSON:"""

        try:
            result = self._llm.complete_json(prompt, model=self._llm.fast_model, temperature=0.2)

            entities = [
                Entity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "concept"),
                    description=e.get("description", ""),
                )
                for e in result.get("entities", [])
                if isinstance(e, dict) and e.get("name")
            ]

            relations = [
                EntityRelation(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relation=r.get("relation", "related_to"),
                )
                for r in result.get("relations", [])
                if isinstance(r, dict) and r.get("source") and r.get("target")
            ]

            return entities, relations

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return [], []
