"""Explicit CVAT review bounds and track species decisions.

Revision ID: 0012_review_regions
Revises: 0011_cvat_sync
"""

import sqlalchemy as sa
from alembic import op

from app.db.types import GUID

revision = "0012_review_regions"
down_revision = "0011_cvat_sync"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "review_regions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("video_id", GUID(), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("boxes_complete", sa.Boolean(), nullable=False),
        sa.Column("ids_complete", sa.Boolean(), nullable=False),
        sa.Column("species_complete", sa.Boolean(), nullable=False),
        sa.Column("annotation_sha256", sa.String(64), nullable=False),
        sa.Column("review_minutes", sa.Float(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("start_frame >= 0 AND end_frame >= start_frame", name="ck_review_region_frames"),
        sa.UniqueConstraint("video_id", "start_frame", "end_frame", name="uq_review_region_bounds"),
    )
    op.create_index("ix_review_regions_video_id", "review_regions", ["video_id"])
    op.create_table(
        "reviewed_species",
        sa.Column("video_id", GUID(), sa.ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("cvat_track_id", sa.Integer(), primary_key=True),
        sa.Column("species", sa.String(256), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("reviewed_species")
    op.drop_index("ix_review_regions_video_id", table_name="review_regions")
    op.drop_table("review_regions")
