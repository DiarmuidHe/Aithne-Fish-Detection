import uuid

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from app.config import Settings


def test_live_migration_upgrade_downgrade(tmp_path, monkeypatch):
    import app.config
    settings = Settings(_env_file=None, database_url=f"sqlite:///{(tmp_path / 'migration.db').as_posix()}")
    monkeypatch.setattr(app.config, "get_settings", lambda: settings)
    config = Config("alembic.ini")
    command.upgrade(config, "0004_track_review")
    command.upgrade(config, "0005_live_monitor")
    engine = create_engine(settings.database_url)
    before = {name: {c["name"] for c in inspect(engine).get_columns(name)}
              for name in ("live_monitor_sessions", "live_fish_tracks")}
    engine.dispose()
    command.upgrade(config, "head")
    engine = create_engine(settings.database_url)
    try:
        inspector = inspect(engine)
        assert {"live_monitor_sessions", "live_fish_tracks", "live_fish_detections"} <= set(inspector.get_table_names())
        from app.db.models import LiveFishDetection, LiveFishTrack, LiveMonitorSession
        for model in (LiveMonitorSession, LiveFishTrack, LiveFishDetection):
            actual = {c["name"]: c for c in inspector.get_columns(model.__tablename__)}
            assert set(actual) == set(model.__table__.columns.keys())
            for column in model.__table__.columns:
                assert actual[column.name]["nullable"] == column.nullable
        index = next(i for i in inspector.get_indexes("live_monitor_sessions") if i["name"] == "uq_live_monitor_open_source")
        assert index["unique"]
        session_added = {"species_id_enabled", "species_id_fish_target", "species_id_frames_per_fish",
                         "species_id_fish_enrolled", "species_id_api_calls",
                         "species_id_candidate_pool_size", "species_id_calls_saved",
                         "species_id_region", "species_id_manual_api_calls"}
        track_added = {"fishial_state", "fishial_species", "fishial_species_confidence",
                       "fishial_frames_used", "fishial_votes_json", "fishial_requested_at",
                       "fishial_completed_at", "fishial_quality_score",
                       # The operator's own answer, kept apart from the classifier's.
                       "manual_species", "manual_species_at"}
        for name, added in (("live_monitor_sessions", session_added), ("live_fish_tracks", track_added)):
            columns = {c["name"]: c for c in inspector.get_columns(name)}
            assert set(columns) - before[name] == added
            for key in added:
                if not columns[key]["nullable"]:
                    assert columns[key]["default"] is not None
        constraint = next(c for c in inspector.get_check_constraints("live_fish_tracks")
                          if c["name"] == "ck_live_fish_tracks_fishial_state")
        assert "'candidate'" in constraint["sqltext"]
        command.downgrade(config, "0006_live_species_id")
        # 0007's downgrade must survive real data: the narrower constraint it
        # recreates forbids the state 0007 introduced.
        constraint = next(c for c in inspect(engine).get_check_constraints("live_fish_tracks")
                          if c["name"] == "ck_live_fish_tracks_fishial_state")
        assert "'candidate'" not in constraint["sqltext"]
        command.upgrade(config, "head")
        session_id, track_id = uuid.uuid4(), uuid.uuid4()
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO live_monitor_sessions (id, source_url, source_key, status, "
                "stop_requested, created_at, frames_processed, dropped_segments, reconnect_count) "
                "VALUES (:id, 'https://camera.example/', 'coral-city', 'stopped', 0, :now, 0, 0, 0)"),
                {"id": str(session_id), "now": "2026-09-10 00:00:00"})
            connection.execute(text(
                "INSERT INTO live_fish_tracks (id, session_id, status, first_seen_at, "
                "last_seen_at, detection_count, max_confidence, mean_confidence, x1, y1, x2, y2, "
                "fishial_state, fishial_frames_used) VALUES (:id, :session, 'finalized', "
                ":now, :now, 3, 0.8, 0.7, 1, 1, 50, 50, 'candidate', 3)"),
                {"id": str(track_id), "session": str(session_id), "now": "2026-09-10 00:00:00"})
        command.downgrade(config, "0006_live_species_id")
        with engine.connect() as connection:
            assert connection.execute(text(
                "SELECT fishial_state FROM live_fish_tracks WHERE id = :id"),
                {"id": str(track_id)}).scalar() == "disabled"
        command.upgrade(config, "head")
        command.downgrade(config, "0005_live_monitor")
        for name, columns in before.items():
            assert {c["name"] for c in inspect(engine).get_columns(name)} == columns
        command.upgrade(config, "head")
        command.downgrade(config, "0004_track_review")
        assert "live_monitor_sessions" not in inspect(engine).get_table_names()
        assert "videos" in inspect(engine).get_table_names()
        command.upgrade(config, "head")
    finally:
        engine.dispose()


def test_recording_link_columns_match_the_video_model(tmp_path, monkeypatch):
    """0008 adds the link a live recording needs, and gives it back cleanly."""

    import app.config
    settings = Settings(_env_file=None,
                        database_url=f"sqlite:///{(tmp_path / 'recording.db').as_posix()}")
    monkeypatch.setattr(app.config, "get_settings", lambda: settings)
    config = Config("alembic.ini")
    command.upgrade(config, "0007_live_species_quality")
    engine = create_engine(settings.database_url)
    try:
        before = {c["name"] for c in inspect(engine).get_columns("videos")}
        command.upgrade(config, "head")
        from app.db.models import Video
        columns = {c["name"]: c for c in inspect(engine).get_columns("videos")}
        assert set(columns) == set(Video.__table__.columns.keys())
        assert set(columns) - before == {"is_live_recording", "source_session_id"}
        # Existing rows predate live recording and must read as ordinary uploads.
        assert columns["is_live_recording"]["nullable"] is False
        assert columns["is_live_recording"]["default"] is not None
        assert any(index["name"] == "ix_videos_source_session_id"
                   for index in inspect(engine).get_indexes("videos"))
        session_id = uuid.uuid4()
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO live_monitor_sessions (id, source_url, source_key, status, "
                "stop_requested, created_at, frames_processed, dropped_segments, reconnect_count) "
                "VALUES (:id, 'https://camera.example/', 'coral-city', 'stopped', 0, :now, 0, 0, 0)"),
                {"id": str(session_id), "now": "2026-09-10 00:00:00"})
            connection.execute(text(
                "INSERT INTO videos (id, original_filename, storage_path, created_at, "
                "processing_status, model_name, pipeline_name, confidence_threshold, "
                "is_live_recording, source_session_id) VALUES (:id, 'live.mp4', '/data/live.mp4', "
                ":now, 'completed', 'm', 'p', 0.6, 1, :session)"),
                {"id": str(uuid.uuid4()), "now": "2026-09-10 00:00:00", "session": str(session_id)})
        # A recording outlives its session: the review decisions on it are the point.
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM live_monitor_sessions WHERE id = :id"),
                               {"id": str(session_id)})
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM videos")).scalar() == 1
        command.downgrade(config, "0007_live_species_quality")
        assert {c["name"] for c in inspect(engine).get_columns("videos")} == before
        command.upgrade(config, "head")
    finally:
        engine.dispose()
