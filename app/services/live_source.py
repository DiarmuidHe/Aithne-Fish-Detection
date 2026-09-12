"""Resolve the configured camera page and continuously record bounded short chunks."""
from __future__ import annotations

import html
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from app.config import Settings
from app.db.models import utc_now


class LiveSourceError(RuntimeError):
    pass


def check_playlist_body(body: str, label: str) -> None:
    """Reject a playlist that advertises neither a rendition nor a segment.

    A camera can be reachable and publishing nothing; FFmpeg reports that opaquely
    several seconds later. Anything that is not recognisably a playlist is left to
    FFmpeg to judge.
    """
    if not body.lstrip().startswith("#EXTM3U"):
        return
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not any("EXT-X-STREAM-INF" in line or not line.startswith("#") for line in lines):
        raise LiveSourceError(
            f"{label} is online but not currently broadcasting; its playlist advertises no stream."
        )


def check_playlist(url: str, client, label: str) -> None:
    """Fetch a playlist to check it before handing the URL to FFmpeg.

    A transport failure here is left to the worker's reconnect path rather than
    being reported as an idle camera.
    """
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError:
        return
    check_playlist_body(response.text, label)


def resolve_stream(url: str, settings: Settings, label: str | None = None, client=None) -> str:
    """Follow video/iframe embeds, then let yt-dlp resolve expiring platform URLs.

    URLs are administrator configuration, never supplied by an API caller.
    Do not persist or log the resolved URL: it can contain signed credentials.
    """
    label = label or "The camera"
    if urlparse(url).scheme not in {"https", "http"}:
        raise LiveSourceError(f"{label} needs an HTTP(S) page or media URL in its configured source")
    if client is None:
        with httpx.Client(timeout=settings.live_read_timeout_seconds, follow_redirects=True) as client:
            return resolve_stream(url, settings, label, client)
    visited = set()
    for _ in range(5):
        if url in visited:
            break
        visited.add(url)
        if re.search(r"\.(m3u8|mpd|mp4)(?:\?|$)", url, re.I):
            if re.search(r"\.m3u8(?:\?|$)", url, re.I):
                check_playlist(url, client, label)
            return url
        host = urlparse(url).hostname or ""
        if host in {"youtube.com", "www.youtube.com", "youtu.be", "www.youtube-nocookie.com"}:
            break
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LiveSourceError(
                f"{label} page could not be fetched; check its configured source URL and network access"
            ) from exc
        url = str(response.url)
        if "mpegurl" in response.headers.get("content-type", ""):
            check_playlist_body(response.text, label)
            return url
        if response.headers.get("content-type", "").startswith("video/"):
            return url
        page = html.unescape(response.text).replace("\\/", "/")
        media = re.search(r'https?://[^\s\x22\x27<>]+\.m3u8[^\s\x22\x27<>]*', page)
        embeds = re.findall(r'<(?:iframe|video|source)\b[^>]*\bsrc=[\x22\x27]([^\x22\x27]+)', page, re.I)
        if media:
            check_playlist(media.group(0), client, label)
            return media.group(0)
        # Exclude unrelated audio embeds (the camera page also has SoundCloud).
        embeds = [value for value in embeds if "soundcloud" not in value]
        if not embeds:
            break
        url = urljoin(url, embeds[0])
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
             "--socket-timeout", str(int(settings.live_read_timeout_seconds)),
             # Live sources publish video-only renditions; audio is discarded downstream,
             # so accept them and retain detail up to the configured capture height.
             "-f", f"bv*[height<=?{settings.live_capture_height}][protocol*=m3u8]/bv*[protocol*=m3u8]/b[protocol*=m3u8]/bv*/b",
             "--get-url", url],
            capture_output=True, text=True, check=False,
            timeout=settings.live_read_timeout_seconds * 2,
        )
    except subprocess.SubprocessError as exc:
        raise LiveSourceError(f"{label} media resolution timed out; retry or configure a direct HLS URL") from exc
    urls = [line.strip() for line in result.stdout.splitlines() if line.startswith(("http://", "https://"))]
    if result.returncode or not urls:
        raise LiveSourceError(
            f"{label}: no playable stream found; the camera may be offline. Install/update yt-dlp "
            "(the live extra), or configure a direct HLS URL for this camera."
        )
    return urls[0]


@dataclass(frozen=True)
class Segment:
    path: Path
    started_at: datetime


class SegmentCapture:
    """FFmpeg ingests independently of inference; only closed segments are consumed."""
    def __init__(self, url: str, directory: Path, settings: Settings):
        self.directory, self.settings = directory, settings
        directory.mkdir(parents=True, exist_ok=True)
        self.consumed: set[str] = set()
        self.last_progress = time.monotonic()
        self.newest: str | None = None
        self.anchor: datetime | None = None
        self.dropped = 0
        try:
            self.process = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-rw_timeout", str(int(settings.live_read_timeout_seconds * 1_000_000)),
                 "-i", url, "-map", "0:v:0", "-an", "-vf",
                 f"fps={settings.live_fps},scale=w='min(iw,{settings.live_capture_width})':"
                 f"h='min(ih,{settings.live_capture_height})':"
                 "force_original_aspect_ratio=decrease:force_divisible_by=2",
                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                 "-force_key_frames", f"expr:gte(t,n_forced*{settings.live_segment_seconds})",
                 "-f", "segment", "-segment_time", str(settings.live_segment_seconds),
                 "-reset_timestamps", "1", str(directory / "%08d.mp4")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise LiveSourceError("Live monitoring requires FFmpeg on PATH") from exc

    def next_segment(self) -> Segment | None:
        files = sorted(self.directory.glob("[0-9]*.mp4"))
        if files and files[-1].name != self.newest:
            # FFmpeg opening the next file proves the camera is still delivering,
            # a segment before that file becomes consumable here.
            self.newest = files[-1].name
            self.last_progress = time.monotonic()
        # The final file is still being written, including on an unexpected exit.
        ready = [p for p in files[:-1] if p.name not in self.consumed]
        if ready:
            self.last_progress = time.monotonic()
            if self.anchor is None:
                self.anchor = utc_now() - timedelta(seconds=(int(ready[-1].stem) + 1) * self.settings.live_segment_seconds)
            while len(ready) > self.settings.live_max_pending_segments:
                old = ready.pop(0)
                old.unlink(missing_ok=True)
                self.dropped += 1
            path = ready[0]
            self.consumed.add(path.name)
            # Keep bookkeeping bounded for sessions running for months.
            self.consumed = {name for name in self.consumed if (self.directory / name).exists()}
            return Segment(path, self.anchor + timedelta(seconds=int(path.stem) * self.settings.live_segment_seconds))
        if self.process.poll() is not None:
            raise LiveSourceError("Camera media connection ended; resolving a fresh stream URL")
        # A closed segment only becomes readable once the next one opens, so a stalled
        # camera cannot be declared before two segment durations have passed.
        if (time.monotonic() - self.last_progress
                > self.settings.live_read_timeout_seconds + 2 * self.settings.live_segment_seconds):
            raise LiveSourceError("Camera stopped delivering frames; reconnecting")
        return None

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for path in self.directory.glob("*.mp4"):
            path.unlink(missing_ok=True)
