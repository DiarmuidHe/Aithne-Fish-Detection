"""The name a person put on a fish, recorded apart from the two machines'.

VIAME reports a class, Fishial reports a species, and neither may overwrite the
other. An operator's own decision is a third claim with a stronger warrant than
either, and it is stored in its own columns for the same reason: a reviewer
reading a name has to be able to tell who said it. Clearing the column restores
the machine answer, because nothing was destroyed to record the human one.

The timestamp is kept because "who decided, and when" is half of what makes a
manual identification auditable; a name with no date is an anonymous assertion.
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_manual_species"
down_revision = "0009_requested_species_id"
branch_labels = None
depends_on = None

TABLES = ("fish_tracks", "live_fish_tracks")
COLUMNS = (
    ("manual_species", sa.String(256), {"nullable": True}),
    ("manual_species_at", sa.DateTime(timezone=True), {"nullable": True}),
)


def upgrade():
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            for name, kind, options in COLUMNS:
                batch.add_column(sa.Column(name, kind, **options))


def downgrade():
    for table in reversed(TABLES):
        with op.batch_alter_table(table) as batch:
            for name, _, _ in reversed(COLUMNS):
                batch.drop_column(name)
