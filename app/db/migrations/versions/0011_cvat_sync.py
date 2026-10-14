"""Development CVAT task synchronization.

Revision ID: 0011_cvat_sync
Revises: 0010_manual_species
"""

import sqlalchemy as sa
from alembic import op

from app.db.types import GUID

revision = "0011_cvat_sync"
down_revision = "0010_manual_species"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cvat_syncs",
        sa.Column("video_id", GUID(), sa.ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("imported_job_id", GUID(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("cvat_syncs")
