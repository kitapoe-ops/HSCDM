# HSCDM — Hormuz Strait Crisis Detection Model

> Z-Score volatility anomaly detection for maritime + NLP composite index
> 海峽危機偵測評分系統 | v1.0.0

---

## Overview

HSCDM (Hormuz Strait Crisis Detection Model) is a quantitative crisis detection system designed to score the operational and sentiment risk level of the Strait of Hormuz — the world's most critical oil chokepoint. It transforms binary "open/closed" questions into a continuous **0–100 HCS (Hormuz Crisis Score)** by measuring volatility regime changes rather than absolute states.

```
HCS = Base(Z) + Adjustment(I) + Momentum(ΔI)
Z   = max(Z_level, Z_vol)  — level anomaly vs volatility surge
State: GREEN → YELLOW → ORANGE → RED
```

---

## Architecture

```
Layer 1 — Feature Engineering
  M(t) = α₁(1-TP_norm) + α₂(1-SP_norm) + α₃·DEV + α₄·ANC_norm   (maritime)
  S(t) = β₁·DG_norm - β₂·CL_norm + β₃·(1-SR)                    (sentiment)

Layer 2 — Composite Index
  I(t) = wₘ·M(t) + wₛ·S(t)   ∈ [0, 1]

Layer 3 — Volatility Anomaly
  Z(t) = (σ₄ₕ(t) - μ₃₀d) / σ₃₀d   (4h rolling std vs 30d baseline)

Layer 4 — State Machine
  GREEN  → Z < 1.0    🟢 Normal operations
  YELLOW → 1.0 ≤ Z < 2.0   🟡 Elevated attention
  ORANGE → 2.0 ≤ Z < 3.0   🟠 High alert [OVERRIDE]
  RED    → Z ≥ 3.0          🔴 TRIAGE MODE [OVERRIDE]

Layer 5 — HCS Score
  Base      = sigmoid(Z; 0.8, 2.0) × 50
  Adj       = I_norm × 30 × (1 + 0.5·min(Z,5))
  Momentum  = ΔI_norm × 20
  HCS       = clip(Base + Adj + Momentum, 0, 100)
```

---

## Quick Start

```bash
# Install
pip install numpy pandas requests feedparser

# Demo (45-day backtest)
python -m mtc.run demo

# Live scoring (requires API keys)
python -m mtc.run live --count 1

# Single snapshot
python -m mtc.run snapshot --snapshot '{
  "throughput": 5, "avg_speed": 2.0,
  "deviation_index": 0.7, "anchor_count": 40,
  "danger_score": 8.0, "calming_score": 0.1,
  "source_reliability": 0.4
}'
```

---

## Installation

```bash
git clone https://github.com/kitapoe-ops/hscdm.git
cd hscdm
pip install -e .

# Optional dev dependencies
pip install -e ".[dev]"
```

---

## Environment Variables

| Variable | Description | Free Tier |
|----------|-------------|-----------|
| `MARINETRAFFIC_API_KEY` | AIS vessel tracking | 500 req/day |
| `NEWS_API_KEY` | News aggregation | 100 req/day |

Without keys, the system runs in **mock mode** using synthetic data.

---

## Python API

```python
from mtc import HSCDM, HSCDMConfig, MTCScorer
from mtc.scanbot_integration import HSCDMScanbotBridge

# Initialize
scorer = MTCScorer()
scorer.calibrate(history_df)

# Score
result = scorer.score_now()
print(f"HCS: {result['HCS']}  State: {result['State']}")
print(result['action'])

# Scanbot bridge
bridge = HSCDMScanbotBridge()
bridge.calibrate(history_df)
rules = bridge.score_and_bridge(snapshot)
print(f"MR: {rules.mean_reversion}  OVX: {rules.ovx_exposure_pct}%")
```

---

## Performance — 45-Day Backtest

| Phase | Description | HCS Mean | GREEN | RED |
|-------|-------------|----------|-------|-----|
| P1 | Normal (Day 1–30) | 26.2 | 84% | 1% |
| P2 | Escalation (Day 31–38) | 80.0 | 12% | 70% |
| P3 | Crisis (Day 39–45) | 99.0 | 1% | 97% |

---

## Scanbot 3.4 Strategy Integration

| State | Mean Reversion | OVX Exposure | HSI Hedge |
|-------|---------------|-------------|-----------|
| 🟢 GREEN | Full | 0% | 0% |
| 🟡 YELLOW | Reduced 50% | 5% | 10% |
| 🟠 ORANGE [OVERRIDE] | Disabled | 15% | 30% |
| 🔴 RED [OVERRIDE] | Disabled | 25% | 50% |

---

## Project Structure

```
mtc/
├── model.py                  # HSCDM core (5 layers)
├── config.py                 # HSCDMConfig / MTCConfig
├── nlp.py                   # NewsSentimentAnalyzer
├── api_clients.py           # MarineTrafficClient / NewsFeedAggregator
├── scoring.py               # MTCScorer
├── scanbot_integration.py    # Scanbot bridge + strategy rules
├── data_generator.py         # 3-phase synthetic data
├── run.py                   # CLI entry
├── demo.py                  # Full demo
├── setup.py
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## License

MIT — see [LICENSE](LICENSE)
