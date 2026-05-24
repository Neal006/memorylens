#!/usr/bin/env python3
"""
One-shot migration: import existing JSON log files into SQLite.

Usage:
    python utils/migrate_legacy_logs.py

Scans experiment_logs/*.json, parses each file, and inserts
into the SQLite database at experiment_logs/memorylens.db.

Safe to run multiple times — skips already-migrated run_ids.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.storage import Storage


LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_logs")


def migrate() -> int:
    """Import legacy JSON logs into SQLite. Returns count of migrated runs."""
    store = Storage()
    count = 0

    if not os.path.isdir(LOG_DIR):
        print(f"Log directory not found: {LOG_DIR}")
        return 0

    existing_ids = {
        row["run_id"]
        for row in store.conn.execute("SELECT run_id FROM runs").fetchall()
    }

    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".json") or fname == "runs_summary.csv":
            continue

        fpath = os.path.join(LOG_DIR, fname)
        with open(fpath) as fh:
            try:
                data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  SKIP {fname}: {exc}")
                continue

        run_id = data.get("run_id")
        if not run_id:
            print(f"  SKIP {fname}: no run_id")
            continue

        if run_id in existing_ids:
            print(f"  SKIP {fname}: already migrated")
            continue

        config = data.get("config", {})
        results = data.get("results", data.get("display_data", {}))

        store.save_run(run_id, config, results)
        print(f"  OK   {fname} -> run_id={run_id}")
        count += 1

    return count


if __name__ == "__main__":
    print("Migrating legacy JSON logs to SQLite...")
    total = migrate()
    print(f"Done. {total} run(s) migrated.")
    sys.exit(0 if total >= 0 else 1)
