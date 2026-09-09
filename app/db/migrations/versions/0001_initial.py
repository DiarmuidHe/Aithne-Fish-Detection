"""Initial fish monitoring schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-08
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _uuid_type():
    bind = op.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.CHAR(32)


def upgrade() -> None:
    uuid_type = _uuid_type()
    op.create_table(
        "videos",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("camera_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processing_status", sa.String(length=32), nullable=False),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("codec", sa.String(length=128), nullable=True),
        sa.Column("viame_version", sa.String(length=128), nullable=True),
        sa.Column("model_name", sa.String(length=256), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=True),
        sa.Column("pipeline_name", sa.String(length=512), nullable=False),
        sa.Column("confidence_threshold", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_path"),
    )
    op.create_index("ix_videos_camera_id", "videos", ["camera_id"])
    op.create_index("ix_videos_processing_status", "videos", ["processing_status"])

    op.create_table(
        "processing_jobs",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column("video_id", uuid_type, nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("stdout_log_path", sa.String(length=1024), nullable=True),
        sa.Column("stderr_log_path", sa.String(length=1024), nullable=True),
        sa.Column("output_csv_path", sa.String(length=1024), nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_processing_jobs_video_id", "processing_jobs", ["video_id"])
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"])
    op.create_index("ix_processing_jobs_status_created", "processing_jobs", ["status", "created_at"])

    op.create_table(
        "fish_tracks",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column("video_id", uuid_type, nullable=False),
        sa.Column("viame_track_id", sa.String(length=128), nullable=False),
        sa.Column("first_frame", sa.Integer(), nullable=False),
        sa.Column("last_frame", sa.Integer(), nullable=False),
        sa.Column("first_timestamp_seconds", sa.Float(), nullable=True),
        sa.Column("last_timestamp_seconds", sa.Float(), nullable=True),
        sa.Column("detection_count", sa.Integer(), nullable=False),
        sa.Column("mean_confidence", sa.Float(), nullable=False),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("species", sa.String(length=256), nullable=True),
        sa.Column("species_confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id", "viame_track_id", name="uq_fish_track_video_viame_id"),
    )
    op.create_index("ix_fish_tracks_video_id", "fish_tracks", ["video_id"])
    op.create_index("ix_fish_tracks_video_confidence", "fish_tracks", ["video_id", "max_confidence"])

    op.create_table(
        "fish_detections",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column("fish_track_id", uuid_type, nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_seconds", sa.Float(), nullable=True),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("class_name", sa.String(length=256), nullable=True),
        sa.Column("class_confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["fish_track_id"], ["fish_tracks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fish_detections_fish_track_id", "fish_detections", ["fish_track_id"])
    op.create_index("ix_fish_detections_frame_number", "fish_detections", ["frame_number"])
    op.create_index(
        "ix_fish_detections_track_frame", "fish_detections", ["fish_track_id", "frame_number"]
    )


def downgrade() -> None:
    op.drop_index("ix_fish_detections_track_frame", table_name="fish_detections")
    op.drop_index("ix_fish_detections_frame_number", table_name="fish_detections")
    op.drop_index("ix_fish_detections_fish_track_id", table_name="fish_detections")
    op.drop_table("fish_detections")

    op.drop_index("ix_fish_tracks_video_confidence", table_name="fish_tracks")
    op.drop_index("ix_fish_tracks_video_id", table_name="fish_tracks")
    op.drop_table("fish_tracks")

    op.drop_index("ix_processing_jobs_status_created", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_video_id", table_name="processing_jobs")
    op.drop_table("processing_jobs")

    op.drop_index("ix_videos_processing_status", table_name="videos")
    op.drop_index("ix_videos_camera_id", table_name="videos")
    op.drop_table("videos")

