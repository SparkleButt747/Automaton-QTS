# LLM Prompt Templates

This document describes every prompt template used by QTS, the expected
input context, and the required output schema.  Templates are versioned;
any change to a template must increment the version and update the history
table below.

---

## 1. Daily Debrief Prompt

**Purpose:** Analyse the previous trading session's performance and propose
evidence-based adjustments to strategy parameters.

**Used by:** `qts.oversight.debrief.DebriefEngine`

### System Prompt

```
You are a quantitative trading system oversight AI.
Your role is to analyze daily trading session performance and suggest evidence-based parameter adjustments.

You MUST respond with a JSON object with the following structure:
{
  "analysis": "<string: narrative analysis of the session>",
  "proposals": [
    {
      "parameter": "<string: parameter name>",
      "current_value": <float>,
      "proposed_value": <float>,
      "reason": "<string: justification>",
      "confidence": <float between 0.0 and 1.0>
    }
  ],
  "regime_alert": "<string or null: alert message if regime change detected>"
}

Rules:
- Only propose changes to STRATEGY parameters (weights, thresholds), NEVER to risk_limits fields.
- All proposed parameter values must be within reasonable bounds.
- Confidence values must be between 0.0 and 1.0.
- The analysis field must be a non-empty string.
- The proposals list may be empty if no changes are warranted.
- Set regime_alert to null if no regime change is detected.
```

### User Prompt (template)

```
Please analyze this trading session and provide your assessment:

```json
{
  "session_date": "YYYY-MM-DD",
  "performance": {
    "total_trades": <int>,
    "win_rate": <float>,
    "total_pnl_usd": <float>,
    "sharpe_today": <float>
  },
  "failure_mode_summary": {
    "SENTIMENT_FADE": <int>,
    "NEWS_FRONT_RUN": <int>,
    "REGIME_MISMATCH": <int>,
    "EXECUTION_SLIP": <int>,
    "FALSE_ALPHA": <int>
  },
  "sentiment_analysis": {
    "hit_rate": <float>,
    "pnl_correlation": <float>
  },
  "top_loss_trades": [
    {
      "trade_id": "<uuid>",
      "symbol": "<str>",
      "pnl_usd": <float>,
      "failure_mode": "<str or null>",
      "exit_reason": "<str>",
      "sentiment_at_entry": <float>,
      "vol_regime": "<str>"
    }
  ],
  "current_params": {
    "w_rsi": <float>,
    "w_macd": <float>,
    "w_bb": <float>,
    "w_mom": <float>,
    "w_sentiment": <float>,
    "entry_threshold": <float>,
    "exit_threshold": <float>
  },
  "vol_regime": "<str>",
  "gdelt_top_events": ["<event1>", "<event2>"]
}
```
```

### Expected Output Schema

```json
{
  "analysis": "string — narrative of session performance",
  "proposals": [
    {
      "parameter": "w_sentiment",
      "current_value": 0.30,
      "proposed_value": 0.25,
      "reason": "Sentiment failed to predict direction in 60% of losing trades.",
      "confidence": 0.72
    }
  ],
  "regime_alert": "null or string describing regime change"
}
```

**Constraints:**
- `proposals[].parameter` must be one of: `w_rsi`, `w_macd`, `w_bb`, `w_mom`,
  `w_sentiment`, `entry_threshold`, `exit_threshold`, `max_hold_bars`.
- `proposals[].confidence` must be in `[0.0, 1.0]`.
- All proposals require human approval before application (see `docs/risk_controls.md`).
- **Risk limits (`max_daily_drawdown_pct`, `max_position_size_pct`, etc.) are
  NEVER proposed or changed by the LLM.**

---

## 2. Regime Classifier Prompt

**Purpose:** Classify the current geopolitical/market regime from news
headlines and GDELT events.  The regime scalar modulates position sizing.

**Used by:** `qts.oversight.regime_classifier.RegimeClassifierLLM`

### System Prompt

```
You are a geopolitical risk analyst for a quantitative trading system.
Analyze the provided news headlines and GDELT events to classify the current market regime.

You MUST respond with a JSON object with the following structure:
{
  "regime": "<one of: geopolitical_risk_off | risk_off | normal | risk_on | geopolitical_risk_on>",
  "confidence": <float between 0.0 and 1.0>,
  "recommended_position_scalar": <float between 0.0 and 1.0>,
  "reasoning": "<string: explanation of the classification>"
}

Guidelines:
- regime: Use "geopolitical_risk_off" for high geopolitical tension, "normal" for typical conditions,
  "risk_on" for high growth/optimism conditions.
- confidence: How certain you are about the regime (0=not confident, 1=very confident).
- recommended_position_scalar: 0.0 means do not trade, 1.0 means full position sizing.
  Use lower values during high uncertainty or risk-off conditions.
- reasoning: Brief explanation citing specific headlines or events.
```

### User Prompt (template)

```
Please classify the current market regime based on the following information:

## Recent News Headlines
- <headline 1>
- <headline 2>
...

## GDELT Geopolitical Events
- <event 1>
- <event 2>
...
```

### Expected Output Schema

```json
{
  "regime": "geopolitical_risk_off",
  "confidence": 0.85,
  "recommended_position_scalar": 0.40,
  "reasoning": "Multiple headlines indicate escalating trade tensions. GDELT shows elevated conflict intensity in key regions."
}
```

**Regime values:**

| Regime                    | Position Scalar Range | Description                       |
|---------------------------|-----------------------|-----------------------------------|
| `geopolitical_risk_off`   | 0.0 – 0.3             | High geopolitical tension         |
| `risk_off`                | 0.3 – 0.5             | Broad market risk aversion        |
| `normal`                  | 0.5 – 1.0             | Typical market conditions         |
| `risk_on`                 | 0.8 – 1.0             | High growth / optimism            |
| `geopolitical_risk_on`    | 0.9 – 1.0             | Strong geopolitical tailwinds     |

---

## Version History

| Version | Date       | Author  | Change Description                              |
|---------|------------|---------|-------------------------------------------------|
| 1.0     | 2024-01-15 | QTS     | Initial debrief and regime classifier prompts   |
| 1.1     | 2024-02-01 | QTS     | Added `gdelt_top_events` to debrief user prompt |
| 1.2     | 2024-03-10 | QTS     | Strengthened rule: LLM cannot propose risk limits |
