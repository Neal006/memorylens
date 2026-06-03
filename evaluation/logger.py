"""
Experiment logger — writes benchmark results to CSV and JSON for
reproducible research and arXiv submission.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from utils.storage import Storage


LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_logs")


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
        import warnings
        warnings.warn(f"SQLite write failed for run {run_id}: {exc}")
    finally:
        if store is not None:
            store.close()

    return json_path


def _append_csv_summary(display_data: Dict, config: Dict, run_id: str) -> None:
    csv_path = os.path.join(LOG_DIR, "runs_summary.csv")
    file_exists = os.path.exists(csv_path)

    checkpoints = display_data.get("checkpoints", [])
    backends = [k for k in display_data if k not in ("checkpoints", "has_llm_eval")]

    rows = []
    for backend in backends:
        d = display_data[backend]
        for i, cp in enumerate(checkpoints):
            rows.append({
                "run_id":    run_id,
                "backend":   backend,
                "turn":      cp,
                "recall":    d["recall"][i] if i < len(d["recall"]) else "",
                "precision": d["precision"][i] if i < len(d["precision"]) else "",
                "drift":     d["drift"][i] if i < len(d["drift"]) else "",
                "noise":     d["noise"][i] if i < len(d["noise"]) else "",
                "tokens":    d["tokens"][i] if i < len(d["tokens"]) else "",
                "total_turns": config.get("total_turns", ""),
            })

    if not rows:
        return

    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def list_runs() -> list:
    """Return metadata for all logged runs, newest first.

    Queries SQLite database first; falls back to filesystem scan
    when the database doesn't exist yet.

    Returns a list of dicts, each with ``run_id``, ``timestamp``,
    and ``config`` keys (unified schema for both storage backends).
    """
    # ── Try SQLite first ────────────────────────────────────────────────────
    store: Optional[Storage] = None
    try:
        store = Storage()
        runs = store.list_runs(limit=50)
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
        if fname.endswith(".json") and fname != "runs_summary.csv":
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

