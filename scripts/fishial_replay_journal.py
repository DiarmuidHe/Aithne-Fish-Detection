"""Durable replay reservations, including retries and interrupted runs."""
import json
import sqlite3

from app.services.fishial import FishialError


class ReplayJournal:
    def __init__(self, path, specification):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK (id=1), spec TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, result TEXT);
            CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, key TEXT, retry INTEGER);
        """)
        spec = json.dumps(specification, sort_keys=True)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (spec,))
        if self.db.execute("SELECT spec FROM config WHERE id=1").fetchone()[0] != spec:
            self.close()
            raise SystemExit("Replay inputs/settings/budget changed; refusing to reuse this experiment")
        self.ceiling = specification["max_calls"]
        self.max_retries = specification["max_retries"]

    @property
    def spent(self):
        return self.db.execute("SELECT count(*) FROM attempts").fetchone()[0]

    def claim(self, key):
        # Claim before any client activity. A missing result after a crash is an
        # uncertain request and must never be replayed under a fresh reservation.
        with self.db:
            result = self.db.execute("INSERT OR IGNORE INTO results(key) VALUES (?)", (key,))
        return result.rowcount == 1

    def reserve(self, key, retry):
        try:
            self.db.execute("BEGIN IMMEDIATE")
            if self.spent >= self.ceiling:
                raise FishialError("budget exhausted")
            retries = self.db.execute("SELECT count(*) FROM attempts WHERE retry=1").fetchone()[0]
            if retry and retries >= self.max_retries:
                raise FishialError("retry limit reached")
            self.db.execute("INSERT INTO attempts(key, retry) VALUES (?, ?)", (key, int(retry)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, key, result):
        with self.db:
            self.db.execute("UPDATE results SET result=? WHERE key=?", (json.dumps(result), key))

    def results(self):
        return {key: json.loads(value) if value else {"reason": "interrupted; not replayed"}
                for key, value in self.db.execute("SELECT key, result FROM results")}

    def close(self):
        self.db.close()
