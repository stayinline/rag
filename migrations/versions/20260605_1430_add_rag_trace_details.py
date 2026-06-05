"""add rag trace details

Revision ID: 20260605_1430
Revises: 20260604_1751
Create Date: 2026-06-05 14:30:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260605_1430"
down_revision = "20260604_1751"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_trace_details",
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", sa.String(length=100), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("answer_preview", sa.Text(), nullable=False),
        sa.Column("answer_length", sa.Integer(), nullable=False),
        sa.Column("kb_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("steps", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("prompt_version", sa.String(length=50), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trace_id"),
    )
    op.create_index(op.f("ix_rag_trace_details_conversation_id"), "rag_trace_details", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_rag_trace_details_message_id"), "rag_trace_details", ["message_id"], unique=False)
    op.create_index(op.f("ix_rag_trace_details_org_id"), "rag_trace_details", ["org_id"], unique=False)
    op.create_index(op.f("ix_rag_trace_details_trace_id"), "rag_trace_details", ["trace_id"], unique=False)
    op.create_index(op.f("ix_rag_trace_details_user_id"), "rag_trace_details", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_rag_trace_details_user_id"), table_name="rag_trace_details")
    op.drop_index(op.f("ix_rag_trace_details_trace_id"), table_name="rag_trace_details")
    op.drop_index(op.f("ix_rag_trace_details_org_id"), table_name="rag_trace_details")
    op.drop_index(op.f("ix_rag_trace_details_message_id"), table_name="rag_trace_details")
    op.drop_index(op.f("ix_rag_trace_details_conversation_id"), table_name="rag_trace_details")
    op.drop_table("rag_trace_details")
