# Supervising AI Governance Agent 🛡️

A production-grade AI oversight system that intercepts, validates, and audits every automated pricing decision before it goes live — preventing algorithmic pricing spirals, anomalies, and race conditions.

---

## Architecture

```
Pricing Bot  →  FastAPI  →  Layer 1 (Redis)  →  LangGraph State Machine
                                                        │
                    ┌───────────────────────────────────┤
                    │        4 Nodes                    │
                    │  check_static_rules               │
                    │  sentiment_circuit_breaker  ←  Mock News Events
                    │  evaluate_context_llm       ←  Gemini 2.0 Flash Lite / Groq Llama3-8B
                    │  route_decision                   │
                    └───────────────────────────────────┘
                                    │
                        ┌───────────┴────────────┐
                        │                        │
                   PostgreSQL              Streamlit Dashboard
                 (audit ledger)           (HITL review queue)
```

## Request Lifecycle

| Step | Component | Action |
|---|---|---|
| 1 | Pricing Bot | POSTs `{product_id, current_price, proposed_price, reasoning}` |
| 2 | FastAPI | Persists `pending` row, invokes LangGraph |
| 3 | Layer 1 (Redis) | Checks static rules in O(1) — auto-rejects violations |
| 4 | Sentiment Breaker | Checks mock news feed — locks essential goods during crises |
| 5 | LLM Node | Gemini/Groq evaluates contextual risk, returns confidence score |
| 6 | Router | confidence ≥ 0.85 → Approved · < 0.60 → HITL queue |
| 7 | Dashboard | Human reviews flagged items, runs blast-radius simulation |
| 8 | Ledger | Final decision immutably logged to PostgreSQL |

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker Desktop (for Postgres + Redis)
- A free API key from [Google AI Studio](https://aistudio.google.com/apikey) **or** [Groq](https://console.groq.com/keys)

### 2. Set up environment

```bash
# Clone / open the project directory
cd "d:\Projects\Athena AI Agent"

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# → Open .env and fill GEMINI_API_KEY (or GROQ_API_KEY + LLM_PROVIDER=groq)
```

### 3. Start infrastructure

```bash
docker compose up -d
# Postgres 16 on :5432, Redis 7 on :6379
```

### 4. Start the API server

```bash
uvicorn main:app --reload
# Swagger UI: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### 5. Start the rogue pricing bot

```bash
# In a new terminal
python bot/pricing_bot.py --interval 2
```

### 6. Open the dashboard

```bash
# In a new terminal
streamlit run dashboard/app.py
# Dashboard: http://localhost:8501
```

---

## LLM Configuration

| Provider | Model | Latency | Free Tier |
|---|---|---|---|
| `gemini` (default) | `gemini-2.0-flash-lite` | ~1s | ✅ Generous |
| `groq` | `llama3-8b-8192` | ~0.5s | ✅ Free tier |

Switch in `.env`: set `LLM_PROVIDER=groq` and `GROQ_API_KEY=...`

---

## Governance Rules (configurable via `.env`)

| Rule | Default | Effect |
|---|---|---|
| `MAX_DISCOUNT_PCT` | 20% | Auto-reject if discount > 20% |
| `MAX_PRICE_INCREASE_PCT` | 50% | Auto-reject if increase > 50% |
| `MIN_ABSOLUTE_PRICE` | $0.50 | Auto-reject if proposed price < $0.50 |
| `AI_AUTO_APPROVE_THRESHOLD` | 0.85 | Auto-approve if LLM confidence ≥ 85% |
| `AI_HIGH_RISK_THRESHOLD` | 0.60 | Escalate to HITL if confidence < 60% |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/governance/submit` | POST | Bot submits pricing update |
| `/api/v1/governance/logs` | GET | Paginated audit log (filterable) |
| `/api/v1/governance/logs/{id}` | GET | Single log entry |
| `/api/v1/hitl/queue` | GET | Items awaiting human review |
| `/api/v1/hitl/review/{id}` | POST | Human submits approve/reject |
| `/api/v1/hitl/stats` | GET | Dashboard KPI stats |
| `/health` | GET | Liveness probe |

Full interactive docs at **http://localhost:8000/docs**

---

## Project Structure

```
├── main.py                    # FastAPI app entry point
├── config.py                  # Pydantic Settings
├── requirements.txt
├── .env.example
├── docker-compose.yml          # Postgres 16 + Redis 7
├── models/
│   └── schemas.py             # Pydantic models + LangGraph TypedDict
├── db/
│   ├── schema.sql             # Immutable audit ledger DDL
│   └── database.py            # Async SQLAlchemy engine
├── api/
│   └── routes.py              # FastAPI routers
├── services/
│   ├── redis_rules.py         # Layer 1 static rule evaluator
│   ├── llm_factory.py         # Gemini / Groq model factory
│   ├── news_sentinel.py       # Mock sentiment circuit breaker
│   └── governance_graph.py    # LangGraph state machine (4 nodes)
├── bot/
│   └── pricing_bot.py         # Async rogue pricing bot
└── dashboard/
    └── app.py                 # Streamlit HITL dashboard
```
