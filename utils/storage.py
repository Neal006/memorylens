"""
SQLite-backed persistent storage for benchmark runs.

Replaces flat JSON/CSV file I/O with a queryable relational store.
Python stdlib only — zero additional dependencies.

Schema
------
runs:      run_id TEXT PK, timestamp TEXT, config_json TEXT
results:   id INTEGER PK AUTOINCREMENT, run_id TEXT FK, backend TEXT,
           turn INTEGER, metric TEXT, value REAL
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_logs")


class Storage:
    """SQLite-backed storage for MemoryLens benchmark runs."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            os.makedirs(_LOG_DIR, exist_ok=True)
            db_path = os.path.join(_LOG_DIR, "memorylens.db")
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── Connection management ──────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._ensure_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                config_json TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id   TEXT NOT NULL REFERENCES runs(run_id),
                backend  TEXT NOT NULL,
                turn     INTEGER NOT NULL,
                metric   TEXT NOT NULL,
                value    REAL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id)
        """)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────

    def save_run(self, run_id: str, config: Dict[str, Any], display_data: Dict[str, Any]) -> None:
        """Insert a benchmark run into the database."""
        # Capture per-backend metadata (provider, decay) into config so
        # get_run() can faithfully reconstruct the original dict.
        config = dict(config)
        backend_meta = {}
        for k in display_data:
            if k in ("checkpoints", "has_llm_eval"):
                continue
            d = display_data[k]
            meta = {}
            if "provider" in d:
                meta["provider"] = d["provider"]
            if "decay" in d:
                meta["decay"] = d["decay"]
            if meta:
                backend_meta[k] = meta
        if backend_meta:
            config["_backend_meta"] = backend_meta

        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, timestamp, config_json) VALUES (?, ?, ?)",
                (run_id, _now_iso(), json.dumps(config)),
            )

            # Delete stale results before inserting fresh ones to prevent
            # duplicate rows on repeated run_id (INSERT OR REPLACE on runs
            # does not cascade to results).
            self.conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))

            # Unfold display_data into metric rows
            checkpoints: List[int] = display_data.get("checkpoints", [])
            backends = [k for k in display_data if k != "checkpoints" and k != "has_llm_eval"]

            rows: List[tuple] = []
            metric_keys = ["recall", "precision", "drift", "noise", "tokens",
                           "cascade_eff", "llm_recall", "llm_drift"]
            for backend in backends:
                backend_data = display_data[backend]
                for i, cp in enumerate(checkpoints):
                    for metric in metric_keys:
                        values = backend_data.get(metric, [])
                        if i < len(values):
                            v = values[i]
                            if v is not None:
                                rows.append((run_id, backend, cp, metric, float(v)))

            self.conn.executemany(
                "INSERT INTO results (run_id, backend, turn, metric, value) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    # ── Read ───────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return the full display_data dict for a run, or None if not found.

        Notes
        -----
        Metric arrays may contain ``None`` for checkpoints where the metric
        was not collected (e.g. ``llm_recall`` when LLM eval was off).
        Callers should handle ``None`` before arithmetic.
        """
        cur = self.conn.execute(
            "SELECT config_json FROM runs WHERE run_id = ?", (run_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None

        config = json.loads(row["config_json"])

        cur = self.conn.execute(
            "SELECT backend, turn, metric, value FROM results WHERE run_id = ? ORDER BY turn",
            (run_id,)
        )
        rows = cur.fetchall()

        # Group by backend, then by turn
        backends: Dict[str, Dict[int, Dict[str, float]]] = {}
        checkpoints: set = set()
        for r in rows:
            backend, turn, metric, value = r["backend"], r["turn"], r["metric"], r["value"]
            checkpoints.add(turn)
            backends.setdefault(backend, {}).setdefault(turn, {})[metric] = value

        cps = sorted(checkpoints)
        display: Dict[str, Any] = {"checkpoints": cps, "has_llm_eval": False}

        metric_keys = ["recall", "precision", "drift", "noise", "tokens",
                       "cascade_eff", "llm_recall", "llm_drift"]

        backend_meta = config.get("_backend_meta", {})

        for backend, turn_map in backends.items():
            display[backend] = {}
            for metric in metric_keys:
                display[backend][metric] = [
                    turn_map[t].get(metric) if metric in turn_map.get(t, {}) else None
                    for t in cps
                ]
            if any(
                v is not None
                for m in ["llm_recall", "llm_drift"]
                for v in display[backend].get(m, [])
            ):
                display["has_llm_eval"] = True

            # Restore scalar metadata fields (provider, decay)
            meta = backend_meta.get(backend, {})
            if "provider" in meta:
                display[backend]["provider"] = meta["provider"]
            if "decay" in meta:
                display[backend]["decay"] = meta["decay"]

        return display

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return run metadata, newest first."""
        cur = self.conn.execute(
            "SELECT run_id, timestamp, config_json FROM runs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        runs = []
        for row in cur:
            runs.append({
                "run_id": row["run_id"],
                "timestamp": row["timestamp"],
                "config": json.loads(row["config_json"]),
            })
        return runs

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Dict[str, Any]:
        """Return recall arrays for two runs, keyed by backend and run_id.

        ``None`` values (missing checkpoints) are filtered out so
        downstream arithmetic / plotting consumers receive clean arrays.
        """
        def _extract_recall(rid: str) -> Dict[str, List[float]]:
            data = self.get_run(rid)
            if data is None:
                return {}
            result: Dict[str, List[float]] = {}
            for k, v in data.items():
                if k in ("checkpoints", "has_llm_eval"):
                    continue
                result[k] = [x for x in v["recall"] if x is not None]
            return result

        return {
            "run_a": {"run_id": run_id_a, "backends": _extract_recall(run_id_a)},
            "run_b": {"run_id": run_id_b, "backends": _extract_recall(run_id_b)},
        }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
