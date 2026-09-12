"""Operator-requested species identification on one chosen fish.

Library tracks gain the same identification columns a live track already has, so
one service can identify either. The session gains a separate counter for calls an
operator asked for, keeping manual spend out of the automatic pass's budget.
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_requested_species_id"
down_revision = "0008_live_recording_video_link"
branch_labels = None
depends_on = None

COLUMNS = (
    ("fishial_state", sa.String(16), {"nullable": False, "server_default": "none"}),
    ("fishial_species", sa.String(256), {"nullable": True}),
    ("fishial_species_confidence", sa.Float(), {"nullable": True}),
    ("fishial_frames_used", sa.Integer(), {"nullable": False, "server_default": "0"}),
    ("fishial_votes_json", sa.Text(), {"nullable": True}),
    ("fishial_requested_at", sa.DateTime(timezone=True), {"nullable": True}),
    ("fishial_completed_at", sa.DateTime(timezone=True), {"nullable": True}),
    ("fishial_quality_score", sa.Float(), {"nullable": True}),
)


def upgrade():
    with op.batch_alter_table("fish_tracks") as batch:
        for name, kind, options in COLUMNS:
            batch.add_column(sa.Column(name, kind, **options))
        batch.create_check_constraint(
            "ck_fish_tracks_fishial_state",
            "fishial_state IN ('none','submitted','identified','review_required','error')",
        )
    with op.batch_alter_table("live_fish_tracks") as batch:
        # A live track already carries the rest; only the claim timestamp is new.
        batch.add_column(sa.Column("fishial_requested_at", sa.DateTime(timezone=True),
                                   nullable=True))
    with op.batch_alter_table("live_monitor_sessions") as batch:
        batch.add_column(sa.Column("species_id_manual_api_calls", sa.Integer(),
                                   nullable=False, server_default="0"))


def downgrade():
    with op.batch_alter_table("live_monitor_sessions") as batch:
        batch.drop_column("species_id_manual_api_calls")
    with op.batch_alter_table("live_fish_tracks") as batch:
        batch.drop_column("fishial_requested_at")
    with op.batch_alter_table("fish_tracks") as batch:
        batch.drop_constraint("ck_fish_tracks_fishial_state", type_="check")
        for name, _, _ in reversed(COLUMNS):
            batch.drop_column(name)
