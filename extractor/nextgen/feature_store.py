"""Transactional feature blobs; one database with bounded WAL sidecars."""
from contextlib import closing
from pathlib import Path
import sqlite3


class FeatureStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, value BLOB NOT NULL)")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def get_many(self, keys):
        result = {}
        with closing(self._connect()) as db:
            for offset in range(0, len(keys), 500):
                batch = keys[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                result.update(db.execute(f"SELECT key,value FROM features WHERE key IN ({placeholders})", batch))
        return result

    def put_many(self, values):
        if values:
            with closing(self._connect()) as db, db:
                db.executemany("INSERT OR REPLACE INTO features(key,value) VALUES (?,?)", values)
