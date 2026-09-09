"""Persist review decisions and the job that produced each track."""

import sqlalchemy as sa
from alembic import op

revision = "0004_track_review"
down_revision = "0003_job_reliability"
branch_labels = None
depends_on = None


def upgrade():
    from app.db.types import GUID

    with op.batch_alter_table("fish_tracks") as batch:
        batch.add_column(sa.Column("processing_job_id", GUID(), nullable=True))
        batch.add_column(sa.Column("review_state", sa.String(32), nullable=False,
                                   server_default="unreviewed"))
        batch.add_column(sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_foreign_key("fk_tracks_processing_job", "processing_jobs",
                                 ["processing_job_id"], ["id"])
        batch.create_index("ix_fish_tracks_processing_job_id", ["processing_job_id"])
        batch.create_check_constraint("ck_fish_tracks_review_state",
            "review_state IN ('unreviewed', 'reviewed', 'accepted', 'rejected', 'needs-review')")
    # Results are replaced only by successful processing. A newer failed/queued
    # attempt must never be attributed as the source of existing observations.
    op.execute("""
        UPDATE fish_tracks SET processing_job_id = (
            SELECT id FROM processing_jobs
            WHERE processing_jobs.video_id = fish_tracks.video_id AND status = 'completed'
            ORDER BY finished_at DESC, created_at DESC, id DESC LIMIT 1
        )
    """)


def downgrade():
    with op.batch_alter_table("fish_tracks") as batch:
        batch.drop_constraint("ck_fish_tracks_review_state", type_="check")
        batch.drop_index("ix_fish_tracks_processing_job_id")
        batch.drop_constraint("fk_tracks_processing_job", type_="foreignkey")
        batch.drop_column("reviewed_at")
        batch.drop_column("review_state")
        batch.drop_column("processing_job_id")
