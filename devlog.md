# Devlog

Append-style log for safe session-to-session resumption. **Most recent entry on top.** Do not rewrite history — add new dated sections above older ones.

---

## 2026-05-31 — Crypto contagion live-paper-trading gate landed; pre-reorg snapshot

**State at checkpoint:** on `main` at commit `b4ffda2`. Project folder reorganisation imminent (this devlog exists so the next session can pick up cleanly after the re-org).

### What shipped this session

End-to-end live-paper-trading planning, plus **Plan 1 (the gate)** implemented and merged to `main`.

- **Spec:** `docs/specs/2026-05-26-crypto-contagion-live-paper-trading-v0.md`
- **Plan 1:** `docs/superpowers/plans/2026-05-26-crypto-contagion-broadened-backtest-gate.md`
- **Decision log:** `.grill/live-paper-trading-crypto-contagion.md`
- **New code on main:**
  - `src/qts/propagation/crypto/detect.py` — pluggable `ShockDetector` Protocol + `IdiosyncraticDropDetector` (idiosyncratic BTC-adjusted trailing-drop trigger).
  - `src/qts/propagation/crypto/broadened.py` — `events_per_year` (fixes the hardcoded-12 Sharpe misscale) + `broadened_verdict`.
  - `scripts/run_contagion_broadened_backtest.py` — the gate runner (detect → existing dataset/fit/Null-A/Null-B/costed backtest at 72h → GO/NO-GO).
  - `scripts/build_live_universe.py` + union builders in `crypto/{universe.py, structural_links.py}` → generated `config/universe/crypto_contagion_live.yaml` (33 tokens) + `config/links/crypto_structural_live.yaml` (31 links).
- **Tests:** 9 unit tests across `tests/unit/test_crypto_{live_union, shock_detector, broadened}.py`, all green via `.venv/bin/python -m pytest tests/unit/test_crypto_*.py -v`.
- **Branch** `feat/crypto-contagion-broadened-gate` merged into `main` (fast-forward, no remote push) and deleted.

### Next action (was blocked on env at session end)

Run the gate to get the GO/NO-GO verdict:

```bash
python scripts/run_contagion_broadened_backtest.py
```

Needs the local LLM at `:8080` + Binance price data. Verdict:

- **GO** → write Plan 2 (model freeze + multi-instrument NautilusTrader live-paper build).
- **NO-GO** → stop; rethink the trigger.

### Pre-reorg WIP snapshot (stashed)

The working tree had earlier-session WIP at session start. Per the standing "no commit without ask" rule it was not committed; to survive the folder re-org it has been stashed (`git stash push -u`). Recover with:

```bash
git stash list   # find the "pre-reorg WIP snapshot 2026-05-31" entry
git stash pop    # restore it after the re-org
```

Files in the stash, for paper trail:

- **Modified:** `scripts/run_path_a_v2.py`; `src/qts/propagation/equity/{__init__.py, comention.py, dataset.py, economic_links.py, gate.py, universe.py}`; `tests/unit/test_equity_{comention.py, economic_links.py, gate.py, universe.py}`.
- **Untracked:** `src/qts/data/market/binance_archive_adapter.py` + `binance_futures_archive_adapter.py` (and their tests); `config/events/contagion_{curve, ftx, run1, terra, usdc}.yaml`; `config/links/crypto_structural_{curve, ftx, run1, terra, usdc}.yaml`; `config/universe/crypto_contagion_{curve, ftx, run1, terra, usdc}.yaml`; `config/universe/path_a_v2_{large, smallcap}.yaml`; `docs/research/2026-05-{24-path-a-v2-real-data-study, 25-crypto-contagion-feasibility}.md`; `docs/specs/2026-05-24-crypto-contagion-propagation-v0.md`; `docs/superpowers/plans/2026-05-24-crypto-contagion-phase-{1-link-graph, 2-dataset, 3-gate-backtest}.md`.

### Resuming after the re-org

1. From the new project root: `git stash list` then `git stash pop` to restore the WIP above.
2. Sanity-check the gate suite: `.venv/bin/python -m pytest tests/unit/test_crypto_live_union.py tests/unit/test_crypto_shock_detector.py tests/unit/test_crypto_broadened.py -v` (expect **9 passed**).
3. Run the gate (above) once your LLM + Binance data are up.
4. Report the verdict; on GO I'll write Plan 2.

Auto-memory (loaded each session) lives at `~/.claude/projects/-Users-brndy-747-Projects-automaton-qts/memory/` — `MEMORY.md` index + `project_crypto_contagion.md` carry the latest state (description frontmatter + LATEST section both updated 2026-05-28).

### Key Plan-2 gotchas (carried forward so they aren't rediscovered)

- The existing `contagion_backtest` is a **return-proxy** (`-reaction - cost`, `gate.py:179`), NOT a position sim → Nautilus paper P&L won't match it 1:1 (higher fidelity, by design).
- MN hedge is **implicit** in the backtest (BTC-adjust); live needs an **explicit** perp short + BTC hedge → futures venue.
- NautilusTrader live skeleton EXISTS but is BROKEN: `src/qts/nautilus/live.py:126-135` builds `data_config` / `exec_config` then never passes them to `TradingNodeConfig` — fix + add a `SandboxExecutionClient` (sim fills on live mainnet data) + load real perp instruments.
- Operator NOT persisted (in-memory `state_dict` only, `train.py:80-86`) → Plan 2 adds fit-once → freeze → load.
- `QTSStrategy` (`nautilus/actor.py`) is **single-instrument** → contagion needs a NEW multi-instrument actor.
- Locked knobs: 72h horizon (NOT the v0 CLI's 24h), paper venue = local sandbox sim on live mainnet data.
