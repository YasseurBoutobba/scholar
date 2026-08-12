from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.orm import GraphStore

_GRAPH_NAME = "main"


class GraphRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_graph(self, graph_data: dict) -> GraphStore:
        row = await self._get_row()
        if row is None:
            row = GraphStore(name=_GRAPH_NAME, graph_data=graph_data)
            self._session.add(row)
        else:
            row.graph_data = graph_data
        await self._session.commit()
        return row

    async def load_graph(self) -> dict | None:
        row = await self._get_row()
        return row.graph_data if row is not None else None

    async def _get_row(self) -> GraphStore | None:
        result = await self._session.scalars(
            select(GraphStore).where(GraphStore.name == _GRAPH_NAME)
        )
        return result.first()
