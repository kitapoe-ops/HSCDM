# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-18

### Added
- **HSCDM v1.0** — 5-layer Hormuz Strait Crisis Detection Model
  - Layer 1: Maritime (M) + NLP Sentiment (S) feature engineering
  - Layer 2: Composite Index I(t) = 0.55·M + 0.45·S
  - Layer 3: Volatility Anomaly Z(t) = (σ₄ₕ - μ₃₀d) / σ₃₀d
  - Layer 4: State Machine — GREEN / YELLOW / ORANGE / RED
  - Layer 5: HCS Score — Base + Adjustment + Momentum
- `HSCDMScanbotBridge` — Scanbot 3.4 strategy integration
- `MTCScorer` — Real-time scoring engine with API polling
- `NewsSentimentAnalyzer` — Keyword-driven NLP for Hormuz news
- `MarineTrafficClient` — AIS data client (mock + real API)
- 45-day synthetic data generator (P1/P2/P3 phases)
- CLI entry point: `python -m mtc.run`
- `scanbot_integration.py` — Strategy rules for all 4 states

### Architecture
```
mtc/
├── model.py              # Core HSCDM (5 layers)
├── config.py              # HSCDMConfig + MTCConfig
├── nlp.py                # NewsSentimentAnalyzer
├── api_clients.py        # MarineTrafficClient + NewsFeedAggregator
├── scoring.py            # MTCScorer real-time engine
├── scanbot_integration.py # Scanbot 3.4 bridge
├── data_generator.py     # 3-phase synthetic data
├── run.py                # CLI entry
├── demo.py               # Full demo
├── setup.py              # pip installable
├── README.md
├── CHANGELOG.md
└── LICENSE               # MIT
```

### Performance (45-day backtest)
| Phase | HCS Mean | GREEN | YELLOW | ORANGE | RED |
|-------|----------|-------|--------|--------|-----|
| P1 Normal | 26.2 | 84% | 13% | 2% | 1% |
| P2 Escalation | 80.0 | 12% | 7% | 11% | 70% |
| P3 Crisis | 99.0 | 1% | 0% | 1% | 97% |

### Known Limitations
- MarineTraffic / NewsAPI keys required for real deployment
- Z-Score for single-point snapshots uses P1 level distribution (not rolling window)
- Dashboard visualization not yet implemented
