"""Add annotated video metadata.

Revision ID: 0002_video_annotation_fields
Revises: 0001_initial
Create Date: 2026-09-08
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_video_annotation_fields"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "videos", sa.Column("annotated_video_path", sa.String(length=1024), nullable=True)
    )
    op.add_column("videos", sa.Column("annotated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("videos", "annotated_at")
    op.drop_column("videos", "annotated_video_path")
