"""
REST API exposing the MemoryLens benchmark pipeline.

Run:
    pip install "memorylens-bench[server]"
    uvicorn memorylens.api:app

Benchmark runs are CPU-bound (embedding model), so POST /v1/benchmarks returns
202 with a job id immediately; poll GET /v1/benchmarks/{job_id} for the result.
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memorylens import __version__, VALID_BACKENDS
from memorylens.simulator.scenarios import SCENARIOS, get_scenario
from memorylens.memory.decay import _REGISTRY as DECAY_REGISTRY

app = FastAPI(
    title="MemoryLens API",
    version=__version__,
    description="Run LLM memory-decay benchmarks over HTTP.",
)

# ponytail: in-memory job store + one thread per job; move to a real queue and
# persistent store when concurrent multi-user runs are actually needed.
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


class BenchmarkRequest(BaseModel):
    turns: int = Field(100, ge=10, le=1000)
    checkpoints: List[int] = Field(default=[10, 25, 50, 75, 100])
    backends: List[str] = Field(default=["naive", "rag", "cascading"])
    scenario: str = "default"
    decay: str = "ebbinghaus"
    seeds: int = Field(1, ge=1, le=5)


def _validate(req: BenchmarkRequest) -> None:
    unknown = [b for b in req.backends if b not in VALID_BACKENDS]
    if unknown:
        raise HTTPException(422, f"Unknown backends {unknown}. Valid: {VALID_BACKENDS}")
    if req.scenario not in SCENARIOS:
        raise HTTPException(422, f"Unknown scenario '{req.scenario}'. Valid: {list(SCENARIOS)}")
    if req.decay not in DECAY_REGISTRY:
        raise HTTPException(422, f"Unknown decay '{req.decay}'. Valid: {list(DECAY_REGISTRY)}")
    if not req.checkpoints:
        raise HTTPException(422, "checkpoints must not be empty")
    out_of_range = [c for c in req.checkpoints if c < 1 or c > req.turns]
    if out_of_range:
        raise HTTPException(
            422, f"checkpoints {out_of_range} outside run horizon 1..{req.turns}"
        )


def _run_job(job_id: str, req: BenchmarkRequest) -> None:
    from memorylens.evaluation.benchmark import (
        run_benchmark, run_benchmark_multi_seed, results_to_display_dict,
    )

    scenario = get_scenario(req.scenario)
    try:
        if req.seeds > 1:
            results = run_benchmark_multi_seed(
                n_seeds=req.seeds,
                total_turns=req.turns,
                eval_checkpoints=sorted(req.checkpoints),
                backends=req.backends,
                decay=req.decay,
                persona_pool=scenario.persona_pool,
                filler_turns=scenario.filler_turns,
            )
        else:
            raw = run_benchmark(
                total_turns=req.turns,
                eval_checkpoints=sorted(req.checkpoints),
                facts=scenario.facts,
                backends=req.backends,
                decay=req.decay,
                filler_turns=scenario.filler_turns,
            )
            results = results_to_display_dict(raw)
        with _jobs_lock:
            _jobs[job_id].update(status="completed", results=results)
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=str(e))


@app.get("/health")
def health() -> Dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/backends")
def backends() -> Dict:
    return {"data": VALID_BACKENDS}


@app.get("/v1/scenarios")
def scenarios() -> Dict:
    return {
        "data": [
            {"name": s.name, "description": s.description, "personas": len(s.persona_pool)}
            for s in SCENARIOS.values()
        ]
    }


@app.post("/v1/benchmarks", status_code=202)
def create_benchmark(req: BenchmarkRequest) -> Dict:
    _validate(req)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "request": req.model_dump()}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"data": {"job_id": job_id, "status": "running"}}


@app.get("/v1/benchmarks/{job_id}")
def get_benchmark(job_id: str) -> Dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No job with id '{job_id}'")
        return {"data": dict(job)}
