"""Optional, budgeted live species identification."""
import sqlalchemy as sa
from alembic import op

revision = "0006_live_species_id"
down_revision = "0005_live_monitor"
branch_labels = None
depends_on = None

SESSION_COLUMNS = (
    "species_id_enabled", "species_id_fish_target", "species_id_frames_per_fish",
    "species_id_fish_enrolled", "species_id_api_calls",
)
TRACK_COLUMNS = (
    "fishial_state", "fishial_species", "fishial_species_confidence",
    "fishial_frames_used", "fishial_votes_json", "fishial_completed_at",
)


def upgrade():
    with op.batch_alter_table("live_monitor_sessions") as batch:
        batch.add_column(sa.Column("species_id_enabled", sa.Boolean(), nullable=False,
                                   server_default=sa.false()))
        for name in SESSION_COLUMNS[1:]:
            batch.add_column(sa.Column(name, sa.Integer(), nullable=False,
                server_default="5" if name == "species_id_frames_per_fish" else "0"))
    with op.batch_alter_table("live_fish_tracks") as batch:
        batch.add_column(sa.Column("fishial_state", sa.String(16), nullable=False,
                                   server_default="disabled"))
        batch.add_column(sa.Column("fishial_species", sa.String(256)))
        batch.add_column(sa.Column("fishial_species_confidence", sa.Float()))
        batch.add_column(sa.Column("fishial_frames_used", sa.Integer(), nullable=False,
                                   server_default="0"))
        batch.add_column(sa.Column("fishial_votes_json", sa.Text()))
        batch.add_column(sa.Column("fishial_completed_at", sa.DateTime(timezone=True)))
        batch.create_check_constraint("ck_live_fish_tracks_fishial_state",
            "fishial_state IN ('disabled','pending','ready','submitted','identified','review_required','error')")


def downgrade():
    with op.batch_alter_table("live_fish_tracks") as batch:
        batch.drop_constraint("ck_live_fish_tracks_fishial_state", type_="check")
        for name in reversed(TRACK_COLUMNS):
            batch.drop_column(name)
    with op.batch_alter_table("live_monitor_sessions") as batch:
        for name in reversed(SESSION_COLUMNS):
            batch.drop_column(name)
