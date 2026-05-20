# Signal Specification

This document defines every signal used by QTS, including its formula,
output range, and staleness policy.  Signal staleness determines whether a
snapshot may be used after a given number of bars without refreshing.

---

## Technical Signals

### RSI(14) — Relative Strength Index

**Formula:**

```
RS       = EMA(gains, 14) / EMA(losses, 14)
RSI(14)  = 100 - (100 / (1 + RS))
```

Where `gains` and `losses` are the positive and negative components of the
1-bar price change respectively.

**Range:** `[0, 100]`

- Values above 70 indicate overbought conditions → bearish pressure.
- Values below 30 indicate oversold conditions → bullish pressure.
- The normalised value fed to alpha: `(RSI - 50) / 50` → `[-1, 1]`.

**Staleness:** 1 bar.  Must be recomputed on every new bar.

**Implementation:** `qts.signals.indicators.compute_rsi` (window=14)

---

### MACD(12, 26, 9) — Moving Average Convergence Divergence

**Formula:**

```
MACD Line    = EMA(close, 12) − EMA(close, 26)
Signal Line  = EMA(MACD Line, 9)
Histogram    = MACD Line − Signal Line
```

**Range:** Unbounded; depends on price scale.

- Positive histogram → bullish momentum.
- Negative histogram → bearish momentum.
- Normalisation for alpha: `tanh(histogram)` → `(-1, 1)`.

**Staleness:** 1 bar.

**Implementation:** `qts.signals.indicators.compute_macd` (fast=12, slow=26, signal=9)

---

### Bollinger Bands(20, 2σ) — BB Position

**Formula:**

```
Middle Band  = SMA(close, 20)
Upper Band   = Middle + 2 × StdDev(close, 20)
Lower Band   = Middle − 2 × StdDev(close, 20)

BB Position  = (close − Lower) / (Upper − Lower)
```

**Range:** `[0, 1]` (values outside this range indicate a breakout).

- Value near 0 → price at or below lower band (potential oversold).
- Value near 1 → price at or above upper band (potential overbought).
- Normalisation for alpha: `2 × BB_Position − 1` → `[-1, 1]`.

**Staleness:** 1 bar.

**Implementation:** `qts.signals.indicators.compute_bollinger_bands`,
`qts.signals.indicators.compute_bb_position`

---

### ATR(14) — Average True Range

**Formula:**

```
True Range   = max(high − low, |high − prev_close|, |low − prev_close|)
ATR(14)      = EMA(True Range, 14)
```

**Range:** `[0, ∞)`.  Expressed in price units.

ATR is used internally by the HMM `RegimeDetector` as the input feature
for volatility regime classification.  It does **not** directly contribute
to the combined alpha score.

**Staleness:** 1 bar.

**Implementation:** `qts.signals.indicators.compute_atr` (window=14)

---

### Momentum(5) — Rate of Change

**Formula:**

```
Momentum(5)  = (close[t] / close[t-5]) − 1
```

**Range:** `(-1, ∞)`.  Typical values are in `(-0.1, 0.1)` for hourly bars.

- Normalisation for alpha: `tanh(momentum × 10)` → `(-1, 1)`.

**Staleness:** 1 bar.

**Implementation:** `qts.signals.indicators.compute_momentum` (window=5)

---

### BB Position

See Bollinger Bands(20, 2σ) above.

**Range:** `[0, 1]`.

---

## Volatility Regime Signal

### Vol Level — HMM Regime Detector

**Method:** 2-state Hidden Markov Model (HMM) trained on ATR history.

**States:** `{HIGH, LOW, TRANSITIONING}`

- `HIGH` — elevated volatility; regime scalar = `1.0`.
- `LOW`  — calm market; regime scalar = `0.5` (dampens signal strength).

**Range:** Discrete categorical `{HIGH, LOW, TRANSITIONING}`.

**Confidence:** Posterior probability of the assigned state from HMM
`predict_proba`.  Stored in `SignalSnapshot.vol_level_confidence`.

**Staleness:** 60 bars (HMM state is relatively stable; re-classification
happens once per hour under 1 h bars).

**Implementation:** `qts.signals.regime.RegimeDetector`

---

## Sentiment Signal

### Sentiment Score — Multi-Source Fusion

**Formula:**

```
sentiment_score = (
    news_weight        × news_score        +
    social_weight      × social_score      +
    geopolitical_weight × geopolitical_score
)
```

Default weights (from `config/params.json`):
- `news_weight = 0.40`
- `social_weight = 0.30`
- `geopolitical_weight = 0.30`

Weights must sum to 1.0 (validated by `SentimentFusionWeights`).

**Component scores:**

| Source          | Model                   | Range     |
|-----------------|-------------------------|-----------|
| News            | FinBERT (positive − negative) | `[-1, 1]` |
| Social (Reddit) | VADER compound score    | `[-1, 1]` |
| Geopolitical    | GDELT TONE normalised   | `[-1, 1]` |

**Range:** `[-1, 1]`.

**Staleness:**
- News: staleness equals the news polling interval (e.g., 5 min).
- Social: staleness equals the Reddit polling interval (e.g., 15 min).
- Geopolitical: staleness equals the GDELT update interval (≈ 15 min).

If a source is unavailable its weight is redistributed proportionally to
the available sources.

**Implementation:** `qts.nlp.fusion.SentimentFusion`

---

## Combined Alpha Signal

### Combined Alpha

**Formula:**

```
rsi_norm      = (RSI − 50) / 50
macd_norm     = tanh(histogram)
bb_norm       = 2 × BB_Position − 1
mom_norm      = tanh(momentum × 10)
sentiment_val = clip(sentiment_score, −1, 1)

raw_alpha = (
    w_rsi       × rsi_norm      +
    w_macd      × macd_norm     +
    w_bb        × bb_norm       +
    w_mom       × mom_norm      +
    w_sentiment × sentiment_val
)

regime_scalar = 1.0 if VolLevel.HIGH else 0.5
combined_alpha = clip(raw_alpha × regime_scalar, −1, 1)
```

**Range:** `[-1, 1]`.

- Positive alpha → consider long entry.
- Negative alpha → consider short/exit.
- Values above `entry_threshold` (default 0.25) trigger a long entry.
- Values below `exit_threshold` (default -0.10) trigger an exit.

**Staleness:** 1 bar (because all component signals are 1-bar stale).

**Implementation:** `qts.signals.alpha.combined_alpha`
