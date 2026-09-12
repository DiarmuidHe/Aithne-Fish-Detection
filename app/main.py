from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import analytics, batch, exports, jobs, live, species, system, tracks, videos
from app.config import Settings, get_settings
from app.db.database import init_db

# Paths the API owns. The single-page app is served for everything else, so an
# unknown path under one of these is a 404 rather than a page of HTML.
API_PREFIXES = frozenset(
    {
        "videos", "jobs", "tracks", "system", "batch", "exports", "analytics", "live",
        "species",
        "health", "static", "legacy", "docs", "redoc", "openapi.json",
    }
)

# Windows resolves some of these from the registry, where .js can come back as
# text/plain. A wrong type on the bundle stops the app from starting at all.
ASSET_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


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


def _dist(settings: Settings) -> Path:
    return settings.frontend_dist.expanduser().resolve()


@app.get("/static/app/{asset_path:path}", include_in_schema=False)
def spa_asset(asset_path: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Serve one built asset. A missing asset is a 404, never the app shell."""

    root = _dist(settings)
    try:
        resolved = (root / asset_path).resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    # Asset filenames carry a content hash, so they can be cached indefinitely.
    cache = "public, max-age=31536000, immutable" if resolved.parent.name == "assets" else "no-store"
    return FileResponse(
        resolved,
        media_type=ASSET_MEDIA_TYPES.get(resolved.suffix.lower()),
        headers={"Cache-Control": cache},
    )


# Registered after the route above so /static/app never reaches this mount.
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
app.include_router(videos.router)
app.include_router(jobs.router)
app.include_router(tracks.router)
app.include_router(system.router)
app.include_router(batch.router)
app.include_router(exports.router)
app.include_router(analytics.router)
app.include_router(live.router)
app.include_router(species.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/legacy", response_class=HTMLResponse, include_in_schema=False)
def legacy_dashboard(request: Request):
    """The previous server-rendered dashboard, kept until parity is signed off."""

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"app_name": get_settings().app_name},
    )


# Registered last: every path that is not the API and not a static asset is a
# client route, and the app resolves it in the browser.
@app.get("/{client_route:path}", response_class=HTMLResponse, include_in_schema=False)
def spa(client_route: str, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    head, _, tail = client_route.partition("/")
    # `/live` is both a client route and the API's prefix. The API has no bare
    # `/live` endpoint, so the page wins there, while `/live/...`, `/static/...`
    # and every other unmatched API path stay a 404 instead of a page of HTML.
    if head in API_PREFIXES and tail:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    index = _dist(settings) / "index.html"
    if not index.is_file():
        return HTMLResponse(_unbuilt_shell(settings), status_code=status.HTTP_200_OK)
    return HTMLResponse(
        index.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


def _unbuilt_shell(settings: Settings) -> str:
    """Shown when the bundle is absent, which means the build step was skipped."""

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{settings.app_name}</title></head><body><div id=\"root\">"
        "<h1>The dashboard has not been built</h1>"
        "<p>Run <code>npm install &amp;&amp; npm run build</code> in <code>frontend/</code>, "
        "or use <code>docker compose up -d --build</code>, which runs the build for you.</p>"
        "<p>The JSON API is unaffected: see <a href=\"/docs\">/docs</a>.</p>"
        "</div></body></html>"
    )
