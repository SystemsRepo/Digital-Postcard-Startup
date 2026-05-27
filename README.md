# Digital Postcard — AI Content Moderation Pipeline

An end-to-end content moderation pipeline that combines Claude/OpenAI LLM evaluation with a Human-in-the-Loop (HITL) review gate, orchestrated as a stateful LangGraph `StateGraph`. Every decision is checkpointed to PostgreSQL so no state is lost between restarts or horizontal scaling events.

---

## Architecture

```
Content Submission  (POST /api/v1/postcards/evaluate)
        │
        ▼
[Deterministic Validation]
  Pydantic length & regex guards
  (no LLM tokens burned on junk input)
        │
        ▼
[LangGraph StateGraph]
        │
   ┌────┴─────────────────────────────┐
   │                                  │
[ai_evaluate]                   [human_review]
   │  LLM assesses content            │  Streamlit dashboard
   │  quality & policy                │  human can approve /
   │  (OpenAI structured output)      │  reject / override
   └──────────────┬───────────────────┘
                  │
           [checkpoint]
                  │  AsyncPostgresSaver writes every
                  │  state transition to Postgres
                  │  (full audit trail, time-travel debug)
                  ▼
     [APPROVED / REJECTED / NEEDS_REVIEW]
                  │
                  ▼
        [Automated Actions]
          Slack alert (NEEDS_REVIEW / severe violations)
          Email to user (REJECTED)
          DB persistence (all outcomes)
```

---

## Why This Pattern Matters for Health Systems

The HITL pattern — AI recommends, human decides, system records — is the right governance model for any high-stakes domain. In clinical AI, an LLM can surface danger signs and suggest a triage level; a trained supervisor reviews that recommendation before the CHW acts on it. The PostgreSQL checkpoint trail becomes a clinical audit log. This project's architecture maps directly onto that model; see `src/health_extension/chw_triage.py` for a concrete CHW triage adaptation.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn (async) |
| Orchestration | LangGraph `StateGraph` |
| LLM | OpenAI `gpt-4o` via `langchain_openai` |
| Structured output | Pydantic v2 (`with_structured_output`) |
| Persistence / checkpointing | PostgreSQL + `AsyncPostgresSaver` (`langgraph-checkpoint-postgres`) |
| Human review UI | Streamlit |
| Rate limiting | `slowapi` |
| Auth | `python-jose` / `passlib` / `bcrypt` |
| Infra | Docker Compose (API + Postgres on port 5435) |

---

## Local Setup

**Prerequisites:** Python 3.11+, Docker, Docker Compose, an OpenAI API key.

```bash
# 1. Install Python dependencies
make install

# 2. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and review DATABASE_SYNC_URL

# 3. Start Postgres + API
make up

# 4. Launch the HITL Streamlit dashboard (separate terminal)
make run-hitl
```

The API is available at `http://localhost:8000`. The Streamlit dashboard runs at `http://localhost:8501`.

---

## Sending a Request

```bash
curl -X POST "http://localhost:8000/api/v1/postcards/evaluate" \
     -H "Content-Type: application/json" \
     -H "X-Agentic-API-Key: agentic-demo-key-123" \
     -d '{
       "id": "pc-001",
       "user_id": "u-123",
       "text_content": "Wishing you a wonderful birthday from all of us!"
     }'
```

---

## Key Patterns

### Graceful LLM Fallback

When the LLM is unavailable (no API key, rate limit, network error), the pipeline never returns HTTP 500. Instead it returns `NEEDS_REVIEW` and routes the submission to the human dashboard:

```python
# src/agent/llm_step.py — finalize_evaluation_node
except Exception as e:
    logger.error(f"Structured Parsing failed: {e}. Fallback to NEEDS_REVIEW.")
    fallback = PostcardEvaluation(
        status=QAStatus.NEEDS_REVIEW,
        reasoning="Automated moderation failed due to technical parsing error. Handing over to Human.",
        suggested_corrections=None
    )
    return {"evaluation": fallback}
```

### Pydantic Schema Enforcement

The LLM's output is constrained to a typed `PostcardEvaluation` model. This prevents prompt injection: the agent cannot instruct the system to "do something else" because the only output surface is a boolean enum (`APPROVED | REJECTED | NEEDS_REVIEW`).

### PostgreSQL Checkpointing

Every node transition in the LangGraph `StateGraph` is written to Postgres via `AsyncPostgresSaver`. If the API process dies mid-evaluation, the next invocation with the same `thread_id` resumes from the last checkpoint. This also provides full "time-travel" debugging — you can inspect exactly which node produced which state.

### Human Supremacy

A human decision written to the `human_reviews` table overrides any prior AI classification. The Streamlit UI provides the final authority; the AI is an advisor, not a decision-maker.

---

## Health Extension

`src/health_extension/chw_triage.py` shows how the same pipeline adapts to community health worker (CHW) clinical triage for iCCM (Integrated Community Case Management) in sub-Saharan Africa.

The mapping is direct:

| Digital Postcard | CHW Triage |
|---|---|
| `PostcardSubmission` | `CHWPatientReport` (FHIR Encounter + Observations) |
| `QAStatus` enum | `TriageDecision` enum (REFER_EMERGENCY / REFER_ROUTINE / TREAT_IN_PLACE / ESCALATE) |
| Content moderation prompt | iCCM clinical decision support prompt |
| Streamlit human review | Supervisor review gate before CHW acts |
| Slack / email dispatch | SMS to CHW + DHIS2 log + facility alert |
| `human_reviews` table | Clinical audit log (FHIR ClinicalImpression) |

Pydantic validators in `CHWPatientReport` enforce plausible clinical ranges (MUAC 60–250 mm, temperature 30–43°C) as a data quality layer before any LLM token is spent. The `CHWTriageOutput` model includes `dq_flags` so MERL teams can track measurement outliers independently of the triage decision.

---

## Running Tests

```bash
# Integration test suite
python3 tests/verify_full_functionality.py

# LLM-as-a-Judge evaluation
PYTHONPATH=. python3 tests/eval_suite.py
```

---

## Production Scaling

| Concern | Approach |
|---|---|
| Compute | ECS Fargate (stateless API containers, scale horizontally) |
| Database | RDS PostgreSQL (Multi-AZ for HA) |
| Checkpoint isolation | LangGraph PostgreSQL lease-based locking — one worker per `thread_id`, no race conditions |
| Cost control | `max_retries=3` on LLM nodes; LangSmith token tracking per request |
| Observability | Loguru structured logs + LangSmith traces; node-level latency visible in trace timeline |

The `AsyncPostgresSaver` with connection pooling (`psycopg_pool`) is the key scaling enabler: because every in-flight reasoning state is in Postgres rather than process memory, you can run N identical API replicas behind a load balancer and any replica can pick up any in-progress evaluation on restart.

---

## Repository Structure

```
src/
  agent/          LangGraph state machine, resilient LLM nodes, operational tools
  api/            FastAPI routes, auth, rate limiting
  engine/         WorkflowRunner core and pipeline step composition
  health_extension/  CHW triage adaptation (HITL pattern → clinical decision support)
  models/         Pydantic schemas (PostcardSubmission, PostcardEvaluation, etc.)
  utils/          Logger, reliability helpers
  hitl_app.py     Streamlit human review dashboard
  main.py         FastAPI app entrypoint
infra/            Docker Compose, Postgres init SQL, environment templates
scripts/          Demo seeding utilities
tests/            Integration + eval suite
```
