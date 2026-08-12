"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-09

Hand-written to mirror backend/models/orm.py (no live Postgres was reachable in
the environment this was authored in to run --autogenerate). Verify with
`alembic upgrade head` against a real database before relying on it in prod.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="upload"),
        sa.Column("arxiv_id", sa.String(64), nullable=True),
        sa.Column("title", sa.Text(), nullable=False, server_default="Untitled paper"),
        sa.Column("authors", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("abstract", sa.Text(), nullable=False, server_default=""),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("ingestion_status", sa.String(16), nullable=False, server_default="processing"),
        sa.Column("filename", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "paper_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("section_label", sa.String(64), nullable=False, server_default="Unknown"),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("embedding_id", sa.String(64), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_paper_id", "chunks", ["paper_id"])

    op.create_table(
        "research_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False, unique=True),
        sa.Column("question", sa.Text(), nullable=False, server_default=""),
        sa.Column("classification", sa.String(32), nullable=False, server_default="factual"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_research_runs_trace_id", "research_runs", ["trace_id"])

    op.create_table(
        "sub_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("hyde_answer", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_sub_questions_run_id", "sub_questions", ["run_id"])

    op.create_table(
        "retrieved_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "sub_question_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sub_questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("retrieval_strategy", sa.String(16), nullable=False, server_default="dense"),
        sa.Column("rrf_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("rerank_score", sa.Float(), nullable=True),
    )
    op.create_index("ix_retrieved_chunks_sub_question_id", "retrieved_chunks", ["sub_question_id"])
    op.create_index("ix_retrieved_chunks_chunk_id", "retrieved_chunks", ["chunk_id"])

    op.create_table(
        "sub_answers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "sub_question_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sub_questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("answer_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("source_paper_ids", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("context_precision_score", sa.Float(), nullable=True),
    )
    op.create_index("ix_sub_answers_sub_question_id", "sub_answers", ["sub_question_id"])

    op.create_table(
        "contradictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sub_question_ids", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("papers_in_conflict", postgresql.JSONB(), nullable=False, server_default="[]"),
    )
    op.create_index("ix_contradictions_run_id", "contradictions", ["run_id"])

    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("research_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("executive_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("sections", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("contradictions", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("confidence_scores", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("source_list", postgresql.JSONB(), nullable=False, server_default="[]"),
    )
    op.create_index("ix_reports_run_id", "reports", ["run_id"])

    op.create_table(
        "graph_store",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True, server_default="main"),
        sa.Column("graph_data", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("phase", sa.String(32), nullable=False, server_default=""),
        sa.Column("event", sa.String(64), nullable=False, server_default=""),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_logs_trace_id", "audit_logs", ["trace_id"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("graph_store")
    op.drop_table("reports")
    op.drop_table("contradictions")
    op.drop_table("sub_answers")
    op.drop_table("retrieved_chunks")
    op.drop_table("sub_questions")
    op.drop_table("research_runs")
    op.drop_table("chunks")
    op.drop_table("papers")
