from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.db import models  # noqa: F401
from app.db.database import Base, get_db
from app.main import app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        upload_root=tmp_path / "uploads",
        job_root=tmp_path / "jobs",
        output_root=tmp_path / "outputs",
        live_scratch_root=tmp_path / "scratch",
        viame_mock=True,
        fishial_enabled=False,
        fishial_client_id=None,
        fishial_client_secret=None,
        viame_sample_csv=Path("tests/fixtures/sample_viame_output.csv"),
        min_fish_confidence=0.60,
        auto_create_tables=False,
    )


@pytest.fixture
def db_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(db_session_factory, test_settings: Settings):
    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
