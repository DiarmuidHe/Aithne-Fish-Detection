"""Persistent live monitoring sessions, fish histories, and observations."""
import sqlalchemy as sa
from alembic import op
from app.db.types import GUID

revision = "0005_live_monitor"
down_revision = "0004_track_review"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "live_monitor_sessions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stop_requested", sa.Boolean(), nullable=False),
        *[sa.Column(name, sa.DateTime(timezone=True), nullable=name != "created_at")
          for name in ("created_at", "started_at", "stopped_at", "heartbeat_at", "last_frame_at")],
        sa.Column("worker_id", sa.String(128)),
        sa.Column("error_message", sa.Text()),
        sa.Column("snapshot_path", sa.String(1024)),
        *[sa.Column(name, sa.Integer(), nullable=False)
          for name in ("frames_processed", "dropped_segments", "reconnect_count")],
    )
    predicate = sa.text("status IN ('queued','starting','running','reconnecting','stopping')")
    op.create_index("uq_live_monitor_open_source", "live_monitor_sessions", ["source_key"],
                    unique=True, postgresql_where=predicate, sqlite_where=predicate)
    op.create_table(
        "live_fish_tracks",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("session_id", GUID(), sa.ForeignKey("live_monitor_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finalized_at", sa.DateTime(timezone=True)),
        sa.Column("finalization_reason", sa.String(32)),
        sa.Column("detection_count", sa.Integer(), nullable=False),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("mean_confidence", sa.Float(), nullable=False),
        sa.Column("species", sa.String(256)),
        *[sa.Column(name, sa.Float(), nullable=False) for name in ("x1", "y1", "x2", "y2")],
        sa.Column("crop_path", sa.String(1024)),
        sa.Column("clip_path", sa.String(1024)),
        sa.Column("media_error", sa.Text()),
    )
    op.create_index("ix_live_fish_tracks_session_id", "live_fish_tracks", ["session_id"])
    op.create_index("ix_live_tracks_session_status", "live_fish_tracks", ["session_id", "status"])
    op.create_table(
        "live_fish_detections",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("track_id", GUID(), sa.ForeignKey("live_fish_tracks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        *[sa.Column(name, sa.Float(), nullable=False) for name in ("x1", "y1", "x2", "y2")],
        sa.Column("species", sa.String(256)),
    )
    op.create_index("ix_live_fish_detections_track_id", "live_fish_detections", ["track_id"])
    op.create_index("ix_live_fish_detections_observed_at", "live_fish_detections", ["observed_at"])


def downgrade():
    op.drop_table("live_fish_detections")
    op.drop_table("live_fish_tracks")
    op.drop_table("live_monitor_sessions")
