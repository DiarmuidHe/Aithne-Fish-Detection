import csv
import hashlib
import io
import json
import urllib.error

import pytest

from scripts.cloud_labelling import run_windows as rw

FIELDS = ["window_id", "round", "day", "role", "evidence", "source_mp4_url", "start_frame", "frame_count",
          "reserve_window_id", "reserve_url", "reserve_start_frame", "attribution"]


def row(n, reserve=True):
    stem = f"SmartBayCam1_2025-07-{n:02d}_1200"
    return {"window_id": f"{stem}_f0100", "round": "1", "day": f"2025-07-{n:02d}", "role": "train",
            "evidence": "strong", "source_mp4_url": f"https://example.test/{stem}.mp4", "start_frame": "100",
            "frame_count": "450", "reserve_window_id": f"{stem}r_f0200" if reserve else "",
            "reserve_url": f"https://example.test/{stem}r.mp4" if reserve else "",
            "reserve_start_frame": "200" if reserve else "", "attribution": "CC BY 4.0"}


def fake_fetch(url, path):
    path.write_bytes(url.encode())
    return {"url": url, "bytes": len(url), "sha256": hashlib.sha256(url.encode()).hexdigest(), "attempts": 1}


def fake_trim(source, target, start, count):
    target.write_bytes(f"{source.name}:{start}:{count}".encode())
    return count


def test_workers_split_the_rows_without_overlap(tmp_path):
    path = tmp_path / "to_label.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row(n) for n in range(1, 8))
    first, second = rw.rows_for(path, 0, 2), rw.rows_for(path, 1, 2)
    assert [r["day"] for r in first] == ["2025-07-01", "2025-07-03", "2025-07-05", "2025-07-07"]
    assert len(second) == 3
    assert not {r["window_id"] for r in first} & {r["window_id"] for r in second}
    assert {r["window_id"] for r in first + second} == {row(n)["window_id"] for n in range(1, 8)}


def test_download_retries_with_backoff_and_records_the_sha256(tmp_path):
    body, calls, waits = b"x" * (3 << 20), [], []

    def opener(request, timeout):
        calls.append(request.get_header("User-agent"))
        if len(calls) < 3:
            raise urllib.error.URLError("timed out")
        return io.BytesIO(body)

    got = rw.download("https://example.test/a.mp4", tmp_path / "a.mp4", opener=opener, sleep=waits.append)
    assert got | {"seconds": 0} == {"url": "https://example.test/a.mp4", "bytes": len(body), "attempts": 3,
                                    "sha256": hashlib.sha256(body).hexdigest(), "seconds": 0}
    assert waits == [5, 15] and calls == [rw.AGENT] * 3
    assert (tmp_path / "a.mp4").read_bytes() == body and not list(tmp_path.glob("*.part"))


def test_download_gives_up_after_the_last_retry(tmp_path):
    def opener(request, timeout):
        raise urllib.error.URLError("refused")

    with pytest.raises(rw.DownloadFailed):
        rw.download("https://example.test/a.mp4", tmp_path / "a.mp4", opener=opener, sleep=lambda s: None)
    assert not list(tmp_path.iterdir())


def test_the_cut_is_by_frame_number_and_refuses_a_wrong_count(tmp_path):
    command = rw.cut_command(tmp_path / "s.mp4", tmp_path / "w.mp4", 946, 450)
    assert "trim=start_frame=946:end_frame=1396,setpts=PTS-STARTPTS" in command
    assert command[command.index("-c:v") + 1] == "libx264" and command[command.index("-crf") + 1] == "14"
    assert command[command.index("-preset") + 1] == "veryfast"
    assert "-an" in command and "-nostdin" in command

    def run(cmd, **kwargs):
        open(cmd[-1], "wb").write(b"video")

    target = tmp_path / "w.mp4"
    assert rw.cut(tmp_path / "s.mp4", target, 946, 450, run=run, frames=lambda p: 450) == 450
    assert target.exists() and not list(tmp_path.glob("*.part.mp4"))
    with pytest.raises(rw.CutFailed, match="449 frames"):
        rw.cut(tmp_path / "s.mp4", tmp_path / "x.mp4", 946, 450, run=run, frames=lambda p: 449)
    assert not (tmp_path / "x.mp4").exists() and not list(tmp_path.glob("*.part.mp4"))


def test_a_short_clip_is_replaced_by_its_reserve(tmp_path):
    r = row(3)
    frames = lambda path: 400 if path.name == "SmartBayCam1_2025-07-03_1200.mp4" else 1680  # noqa: E731
    record = rw.prepare(r, tmp_path, fetch=fake_fetch, trim=fake_trim, frames=frames)
    assert record["reserve_used"] and record["labelled_as"] == r["reserve_window_id"]
    assert record["start_frame"] == 200 and record["frame_count"] == 450
    assert "400 frames" in record["failures"][0]["error"]
    assert record["source"]["sha256"] == hashlib.sha256(r["reserve_url"].encode()).hexdigest()
    assert not (tmp_path / "SmartBayCam1_2025-07-03_1200r.mp4").exists()  # the cut's source is deleted
    assert (tmp_path / "SmartBayCam1_2025-07-03_1200.mp4").exists()  # the failed clip's is kept
    assert json.loads((tmp_path / f"{r['reserve_window_id']}.json").read_text()) == record

    def no_fetch(url, path):
        raise AssertionError("an intact cut is reused, not downloaded again")

    assert rw.prepare(r, tmp_path, fetch=no_fetch, trim=fake_trim, frames=frames) == record
    with pytest.raises(rw.WindowFailed):
        rw.prepare(row(4, reserve=False), tmp_path, fetch=fake_fetch, trim=fake_trim, frames=lambda p: 10)


def test_run_labels_in_order_skips_finished_windows_and_stops_after_repeated_failures(tmp_path):
    rows = [row(n) for n in range(1, 8)]
    stages, prepared = [], []

    def prepare(r, windows):
        prepared.append(r["window_id"])
        if r is rows[1]:
            raise rw.WindowFailed("no clip")
        return {"labelled_as": r["window_id"], "reserve_used": False}

    def stage(name, video, root, logs):
        stages.append((video.stem, name))
        if video.stem in (rows[3]["window_id"], rows[4]["window_id"], rows[5]["window_id"]) and name == "sam3":
            raise rw.StageFailed("CUDA error")
        return 1.0

    events = tmp_path / "run_windows" / "worker-0.jsonl"
    summary = rw.run(rows, tmp_path, events, prepare_window=prepare, stage=stage,
                     done=lambda root, window_id: window_id == rows[0]["reserve_window_id"], free_gb=lambda p: 50)
    assert summary == {"already_labelled": 1, "failed": 4, "labelled": 1, "reserve_used": 0}
    assert rows[6]["window_id"] not in prepared  # stopped after three failures in a row
    assert [s for w, s in stages if w == rows[2]["window_id"]] == ["sam3", "static", "labels"]
    assert all(s != "render" for _, s in stages)
    kinds = [json.loads(line)["event"] for line in events.read_text().splitlines()]
    assert kinds == ["start", "failed", "labelled", "failed", "failed", "failed", "stop", "end"]


def test_run_stops_when_the_disk_runs_low(tmp_path):
    summary = rw.run([row(1)], tmp_path, tmp_path / "e.jsonl", prepare_window=None, stage=None,
                     done=lambda root, window_id: False, free_gb=lambda p: 2)
    assert summary == {}
    assert "GB free" in (tmp_path / "e.jsonl").read_text()


def test_labelled_reads_the_round_journal(tmp_path):
    assert not rw.labelled(tmp_path, "w")
    (tmp_path / "journal.json").write_text(json.dumps({"entries": {
        "a/labels/fish": {"status": "complete"}, "b/labels/fish": {"status": "failed"},
        "c/sam3/fish": {"status": "complete"}}}))
    assert rw.labelled(tmp_path, "a") and not rw.labelled(tmp_path, "b") and not rw.labelled(tmp_path, "c")
