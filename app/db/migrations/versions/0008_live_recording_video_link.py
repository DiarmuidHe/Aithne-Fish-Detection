"""Register a completed live session's recording as an ordinary video."""

import sqlalchemy as sa
from alembic import op

revision = "0008_live_recording_video_link"
down_revision = "0007_live_species_quality"
branch_labels = None
depends_on = None


def upgrade():
    from app.db.types import GUID

    with op.batch_alter_table("videos") as batch:
        batch.add_column(sa.Column("is_live_recording", sa.Boolean(), nullable=False,
                                   server_default=sa.text("false")))
        batch.add_column(sa.Column("source_session_id", GUID(), nullable=True))
        # A deleted session must not take the reviewed recording with it: the video,
        # its tracks and its annotations stand on their own once converted.
        batch.create_foreign_key("fk_videos_source_session", "live_monitor_sessions",
                                 ["source_session_id"], ["id"], ondelete="SET NULL")
        batch.create_index("ix_videos_source_session_id", ["source_session_id"])


def downgrade():
    with op.batch_alter_table("videos") as batch:
        batch.drop_index("ix_videos_source_session_id")
        batch.drop_constraint("fk_videos_source_session", type_="foreignkey")
        batch.drop_column("source_session_id")
        batch.drop_column("is_live_recording")
