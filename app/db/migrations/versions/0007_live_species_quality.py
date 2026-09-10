"""Candidate pooling, quality-ranked selection, early stopping and regional filtering."""
import sqlalchemy as sa
from alembic import op

revision = "0007_live_species_quality"
down_revision = "0006_live_species_id"
branch_labels = None
depends_on = None

SESSION_COLUMNS = (
    "species_id_candidate_pool_size", "species_id_calls_saved", "species_id_region",
)
TRACK_COLUMNS = ("fishial_quality_score",)

OLD_STATES = "'disabled','pending','ready','submitted','identified','review_required','error'"
NEW_STATES = ("'disabled','candidate','pending','ready','submitted',"
              "'identified','review_required','error'")
CONSTRAINT = "ck_live_fish_tracks_fishial_state"


def _swap_state_constraint(states):
    # Postgres cannot alter a check constraint in place, so drop and recreate it.
    # batch_alter_table gives SQLite the same behaviour via table rebuild.
    with op.batch_alter_table("live_fish_tracks") as batch:
        batch.drop_constraint(CONSTRAINT, type_="check")
        batch.create_check_constraint(CONSTRAINT, f"fishial_state IN ({states})")


def upgrade():
    with op.batch_alter_table("live_monitor_sessions") as batch:
        batch.add_column(sa.Column("species_id_candidate_pool_size", sa.Integer(),
                                   nullable=False, server_default="0"))
        batch.add_column(sa.Column("species_id_calls_saved", sa.Integer(),
                                   nullable=False, server_default="0"))
        batch.add_column(sa.Column("species_id_region", sa.String(64)))
    with op.batch_alter_table("live_fish_tracks") as batch:
        batch.add_column(sa.Column("fishial_quality_score", sa.Float()))
    _swap_state_constraint(NEW_STATES)


def downgrade():
    # A live database can hold rows in the state the narrower constraint forbids.
    # Retire them first: an unpaid candidate is equivalent to a disabled track.
    op.execute(sa.text("UPDATE live_fish_tracks SET fishial_state = 'disabled' "
                       "WHERE fishial_state = 'candidate'"))
    _swap_state_constraint(OLD_STATES)
    with op.batch_alter_table("live_fish_tracks") as batch:
        for name in reversed(TRACK_COLUMNS):
            batch.drop_column(name)
    with op.batch_alter_table("live_monitor_sessions") as batch:
        for name in reversed(SESSION_COLUMNS):
            batch.drop_column(name)
