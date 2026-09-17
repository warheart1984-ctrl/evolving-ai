# Evolving AI — Governed, Evolving AI Runtime (v0 Prototype)

A governed, evolving AI runtime where **nothing evolves in place**. Every change is a
versioned amendment that must be evaluated and approved before promotion.

## Architecture

Four distinct roles:

| Role      | Responsibility                                            | Power          |
|-----------|-----------------------------------------------------------|----------------|
| Operator  | Executes user tasks with approved tools/memory            | Productive     |
| Steward   | Analyzes failures/telemetry, generates amendment proposals | Creative       |
| Evaluator | Replays candidate runtimes, detects regressions           | Adversarial    |
| Governor  | Enforces constitution, gates promotions, rollback          | Authoritative  |

The central constraint: **the Steward can propose changes, but cannot unilaterally enact them.**

Amendment lifecycle:

```
PROPOSED → SANDBOX → EVALUATED → REVIEW → APPROVED/REJECTED → PROMOTED
```

## Layout

```
app/
├── runtime/        # (reserved) manifest/executor/versioning
├── operator/       # executes tasks, cannot self-modify
├── steward/        # failure analysis → amendment proposals
├── evaluation/     # replay suites, metrics, regression detection
├── governance/     # models, gates, Governor, runtime registry
├── memory/         # episodic/semantic stores, governed lessons
├── tools/          # (reserved) tool implementations
└── api/            # FastAPI app + dashboard
constitution/       # constitution.yaml + model
evaluations/        # suites + datasets
prompts/            # versioned prompt artifacts
tests/              # unit / integration / governance / replay
```

## Quickstart

```bash
pip install "fastapi>=0.100" "pydantic>=2.5" uvicorn pytest
uvicorn app.api.main:app --reload
```

Dashboard: http://localhost:8000/dashboard

Or with Docker:

```bash
docker compose up --build
```

## Endpoints

- `POST /execute` — execute a task with the current runtime
- `GET /runtime/current` — inspect the current immutable runtime
- `GET /runtime/list` — list all runtime versions
- `POST /runtime/rollback/{version}` — rollback
- `POST /amendment/propose` — propose an amendment (prompt/memory only in v0)
- `GET /amendments` / `GET /amendment/{id}` — list/inspect amendments
- `POST /evaluation/run` — run a replay suite against a runtime
- `POST /governance/evaluate/{amendment_id}` — evaluate → attach evidence → REVIEW
- `POST /governance/approve/{amendment_id}` — approve + promote (human approval)
- `POST /governance/reject/{amendment_id}` — reject
- `GET /governance/audit` — full audit trail
- `POST /steward/analyze` — run the steward loop (never promotes)
- `GET /telemetry/failures` / `GET /telemetry/runs` — inspect telemetry
- `POST /memory/lesson` + lifecycle endpoints — governed lessons

## Demo flow (definition of done)

```bash
# 1. Operator repeatedly fails a task class (seed failures)
curl -X POST "localhost:8000/steward/analyze"

# 2. Steward proposes amendments; evaluate one
curl -X POST "localhost:8000/governance/evaluate/<amendment_id>?suite_id=core"

# 3. Human approves → new immutable runtime created
curl -X POST "localhost:8000/governance/approve/<amendment_id>?reviewer=me"

# 4. Old runtime remains for rollback
curl -X POST "localhost:8000/runtime/rollback/v0"

# 5. Audit trail
curl "localhost:8000/governance/audit"
```

## Tests

```bash
python -m pytest tests/ -v
```

Covers: runtime immutability, amendment lifecycle, governance gates,
approval/rejection, rollback, evaluation reproducibility, memory lifecycle,
audit logging, unauthorized self-modification attempts, and the full
end-to-end evolution pipeline.
