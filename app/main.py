from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import analytics, batch, exports, jobs, system, tracks, videos
from app.config import get_settings
from app.db.database import init_db


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    settings.upload_root.mkdir(parents=True, exist_ok=True)
    settings.job_root.mkdir(parents=True, exist_ok=True)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    if settings.auto_create_tables:
        init_db()
    yield


APP_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=APP_ROOT / "templates")

app = FastAPI(title=get_settings().app_name, version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
app.include_router(videos.router)
app.include_router(jobs.router)
app.include_router(tracks.router)
app.include_router(system.router)
app.include_router(batch.router)
app.include_router(exports.router)
app.include_router(analytics.router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"app_name": get_settings().app_name},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
