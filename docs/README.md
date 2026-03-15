# Automaton QTS — Documentation Index

This directory contains the technical documentation for the Automaton QTS quantitative trading system. Each document below covers a distinct aspect of the system design, operation, and configuration.

---

## Documents

### [architecture.md](architecture.md)

**Full system architecture, data flow diagrams, and database schema.**

Covers the end-to-end system design across seven phases: data ingestion, signal computation, alpha fusion, sentiment processing, execution, oversight, and monitoring. Includes:

- ASCII system block diagram showing all layers (Data, Signal, Strategy, Execution, Oversight, Monitoring, Analytics, Database)
- Per-phase data flow descriptions with hot/warm/cold path latency targets
- Component interaction diagram showing how CLI, settings, execution engine, and persistence layer wire together
- Full database schema for `trade_records`, `signal_snapshots`, and `proposals` tables

---

### [signal_spec.md](signal_spec.md)

**Every signal: formula, output range, staleness policy, and normalisation details.**

Canonical reference for all signals consumed by the alpha fusion layer:

- **RSI(14)** — formula, `[0, 100]` range, normalisation to `[-1, 1]`
- **MACD(12, 26, 9)** — histogram computation, `tanh` normalisation
- **Bollinger Bands(20, 2σ)** — BB Position formula, `[0, 1]` range
- **ATR(14)** — true range formula, use as HMM input feature
- **Momentum(5)** — rate-of-change formula, `tanh` normalisation
- **Vol Regime (HMM)** — 2-state Hidden Markov Model, `HIGH`/`LOW` states, 60-bar staleness
- **Sentiment Score** — multi-source fusion formula (FinBERT × 0.4 + VADER × 0.3 + GDELT × 0.3)
- **Combined Alpha** — full fusion formula with regime scalar and entry/exit threshold semantics

---

### [risk_controls.md](risk_controls.md)

**Immutable hard risk limits, rationale for each, and the procedure for changing them.**

Documents all five limits enforced by `RiskManager`:

- `max_daily_drawdown_pct` (2%) — circuit breaker trigger and automatic cooldown mechanics
- `max_position_size_pct` (5%) — single-position concentration cap
- `max_open_positions` (5) — simultaneous position cap
- `circuit_breaker_cooldown_seconds` (3600) — post-trip lockout behaviour
- `sentiment_signal_max_scalar` (2.0) — sentiment influence cap

Also covers: who can change limits, the `--no-verify` commit procedure, audit trail requirements, and emergency halt/resume commands.

---

### [llm_prompt_templates.md](llm_prompt_templates.md)

**All LLM prompt templates, expected JSON output schemas, and version history.**

Defines the two prompts used by the LLM oversight layer:

1. **Daily Debrief Prompt** — used by `DebriefEngine` to analyse session performance and generate parameter change proposals. Includes full system prompt, user prompt template with all input fields, expected output schema, and constraints (LLM cannot propose risk limit changes).

2. **Regime Classifier Prompt** — used by `RegimeClassifierLLM` to classify the current geopolitical/market regime from news headlines and GDELT events. Includes regime value table with position scalar ranges (`geopolitical_risk_off` through `geopolitical_risk_on`).

Includes a version history table tracking all template changes.

---

### [incident_runbook.md](incident_runbook.md)

**Six incident response procedures covering detection, containment, investigation, resolution, and post-incident review.**

| Incident | Summary |
|---|---|
| **1. Data Feed Goes Stale** | Detect via `HealthChecker`, investigation commands, Binance restart procedure |
| **2. Circuit Breaker Triggered** | Automatic halt mechanics, trade log review, cooldown management |
| **3. LLM API Unavailable** | Cold-path isolation — trading continues; API key rotation and backfill |
| **4. Unexpected Large Loss** | Immediate manual halt, position review, root cause classification |
| **5. Database Connection Failure** | In-memory buffer fallback, flush procedure, data loss prevention |
| **6. Redis / Celery Worker Failure** | Cold-path isolation — hot trading unaffected; worker restart steps |

---

## Quick Navigation

| I want to... | Go to |
|---|---|
| Understand the overall system design | [architecture.md](architecture.md) |
| Check a signal's formula or output range | [signal_spec.md](signal_spec.md) |
| Change a risk limit (or understand why it's hard) | [risk_controls.md](risk_controls.md) |
| Inspect or modify an LLM prompt | [llm_prompt_templates.md](llm_prompt_templates.md) |
| Respond to a live system incident | [incident_runbook.md](incident_runbook.md) |

---

See the project [README](../README.md) for installation, configuration, and usage instructions.
