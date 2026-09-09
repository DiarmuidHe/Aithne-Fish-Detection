"""Add job uniqueness, configuration snapshots, and worker heartbeats.

Revision ID: 0003_job_reliability
Revises: 0002_video_annotation_fields
Create Date: 2026-09-09
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_job_reliability"
down_revision = "0002_video_annotation_fields"
branch_labels = None
depends_on = None


def _uuid_type():
    bind = op.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.UUID(as_uuid=True)
    return sa.CHAR(32)


def upgrade() -> None:
    op.add_column("videos", sa.Column("size_bytes", sa.BigInteger(), nullable=True))
    op.add_column("videos", sa.Column("content_sha256", sa.String(length=64), nullable=True))

    op.add_column(
        "processing_jobs",
        sa.Column("worker_mode", sa.String(length=16), server_default="mock", nullable=False),
    )
    op.add_column(
        "processing_jobs", sa.Column("worker_id", sa.String(length=128), nullable=True)
    )
    op.add_column(
        "processing_jobs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "processing_jobs",
        sa.Column("configuration_json", sa.Text(), server_default="{}", nullable=False),
    )
    op.create_index("ix_processing_jobs_worker_mode", "processing_jobs", ["worker_mode"])
    op.create_index("ix_processing_jobs_worker_id", "processing_jobs", ["worker_id"])

    # Preserve the newest active job if an older deployment admitted a race.
    op.execute(
        """
        UPDATE processing_jobs
        SET status = 'failed',
            finished_at = CURRENT_TIMESTAMP,
            error_message = 'Superseded while enforcing one active job per video'
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY video_id ORDER BY created_at DESC, id DESC
                       ) AS active_rank
                FROM processing_jobs
                WHERE status IN ('queued', 'processing')
            ) ranked
            WHERE active_rank > 1
        )
        """
    )
    op.create_index(
        "uq_processing_jobs_one_active_per_video",
        "processing_jobs",
        ["video_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'processing')"),
        sqlite_where=sa.text("status IN ('queued', 'processing')"),
    )

    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_job_id", _uuid_type(), nullable=True),
        sa.PrimaryKeyConstraint("worker_id"),
    )
    op.create_index("ix_worker_heartbeats_mode", "worker_heartbeats", ["mode"])
    op.create_index("ix_worker_heartbeats_last_seen_at", "worker_heartbeats", ["last_seen_at"])


def downgrade() -> None:
    op.drop_index("ix_worker_heartbeats_last_seen_at", table_name="worker_heartbeats")
    op.drop_index("ix_worker_heartbeats_mode", table_name="worker_heartbeats")
    op.drop_table("worker_heartbeats")

    op.drop_index("uq_processing_jobs_one_active_per_video", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_worker_id", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_worker_mode", table_name="processing_jobs")
    op.drop_column("processing_jobs", "configuration_json")
    op.drop_column("processing_jobs", "heartbeat_at")
    op.drop_column("processing_jobs", "worker_id")
    op.drop_column("processing_jobs", "worker_mode")

    op.drop_column("videos", "content_sha256")
    op.drop_column("videos", "size_bytes")
