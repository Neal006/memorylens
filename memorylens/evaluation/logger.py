"""
Experiment logger — persists benchmark runs as JSON + CSV + SQLite.

JSON files remain the interchange format; SQLite (memorylens.db) is the
queryable store used by list_runs()/get_run_results().
"""

import csv
import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

from memorylens.utils.storage import Storage, LOG_DIR


def _ensure_dir() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def log_run(display_data: Dict, config: Dict[str, Any], run_id: Optional[str] = None) -> str:
    """
    Persist a benchmark run to disk (JSON + CSV + SQLite).
    Returns the path to the saved JSON file.
    """
    _ensure_dir()

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"run_id": run_id, "config": config, "results": display_data}

    json_path = os.path.join(LOG_DIR, f"{run_id}.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    _append_csv_summary(display_data, config, run_id)

    # ── SQLite persistence (non-blocking on failure) ────────────────────────
    store: Optional[Storage] = None
    try:
        store = Storage()
        store.save_run(run_id, config, display_data)
    except Exception as exc:
        warnings.warn(f"SQLite write failed for run {run_id}: {exc}")
    finally:
        if store is not None:
            store.close()

    return json_path


def _append_csv_summary(display_data: Dict, config: Dict, run_id: str) -> None:
    csv_path = os.path.join(LOG_DIR, "runs_summary.csv")
    file_exists = os.path.exists(csv_path)

    checkpoints = display_data.get("checkpoints", [])
    backends = [k for k, v in display_data.items() if isinstance(v, dict)]

    def _at(d: Dict, key: str, i: int):
        series = d.get(key, [])
        return series[i] if i < len(series) else ""

    rows = []
    for backend in backends:
        d = display_data[backend]
        for i, cp in enumerate(checkpoints):
            rows.append({
                "run_id":        run_id,
                "backend":       backend,
                "turn":          cp,
                "recall":        _at(d, "recall", i),
                "precision":     _at(d, "precision", i),
                "drift":         _at(d, "drift", i),
                "noise":         _at(d, "noise", i),
                "contradiction": _at(d, "contradiction", i),
                "tokens":        _at(d, "tokens", i),
                "total_turns":   config.get("total_turns", ""),
            })

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    if file_exists:
        with open(csv_path, newline="") as fh:
            header = fh.readline().strip().split(",")
        if header != fieldnames:
            # Schema changed (new metric column) — rotate the old file instead
            # of appending misaligned rows.
            os.replace(csv_path, csv_path.replace(".csv", "_legacy.csv"))
            file_exists = False

    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def list_runs() -> list:
    """Return metadata for all logged runs, newest first.

    Queries SQLite first; falls back to a filesystem scan when the database
    doesn't exist yet. Each entry has ``run_id``, ``timestamp``, and
    ``config`` keys (unified schema for both storage backends).
    """
    store: Optional[Storage] = None
    try:
        store = Storage()
        runs = store.list_runs(limit=50)
        # Once any SQLite run exists, filesystem logs are bypassed.
        # Run python -m memorylens.utils.migrate_legacy_logs to import them.
        if runs:
            return runs
    except Exception:
        pass
    finally:
        if store is not None:
            store.close()

    # ── Fallback: scan filesystem (legacy) ─────────────────────────────────
    log_dir = _ensure_dir()
    runs = []
    for fname in sorted(os.listdir(log_dir), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(log_dir, fname)
            with open(fpath) as fh:
                data = json.load(fh)
            runs.append({
                "run_id": data.get("run_id"),
                "timestamp": datetime.fromtimestamp(os.path.getmtime(fpath))
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "config": data.get("config", {}),
            })
    return runs


def get_run_results(run_id: str) -> Optional[Dict]:
    """Return the display_data dict for a run — SQLite first, JSON fallback."""
    store: Optional[Storage] = None
    try:
        store = Storage()
        results = store.get_run(run_id)
        if results is not None:
            return results
    except Exception:
        pass
    finally:
        if store is not None:
            store.close()

    json_path = os.path.join(LOG_DIR, f"{run_id}.json")
    if os.path.exists(json_path):
        with open(json_path) as fh:
            return json.load(fh).get("results")
    return None
