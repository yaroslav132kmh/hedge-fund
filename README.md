# CryptoHedge — Multi-agent System for Hedging Crypto Currency Risk

The system makes hedging decisions for the currency risk driven by
**cryptocurrency volatility**, using the **Heston model** and eleven autonomous
agents. The project is modular, reproducible, configurable and fully bundled with
data.

> Base scenario: capital **$10,000,000**; risk appetite and fees are set in the
> configuration; analysis horizon — **3 months**; forecast horizon — **1 day**.

- Architecture, diagrams and the agent interaction protocol: **[ARCHITECTURE.en.md](ARCHITECTURE.en.md)**.
- Single autonomous solution: the **[`solution.ipynb`](solution.ipynb)** notebook.
- 🇷🇺 **Русская версия этого документа: [README.md](README.md)** · **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 1. Capabilities (mapping to the requirements)

| Requirement | Where implemented |
|---|---|
| 1. Capital, risk, fees | `config/system.yaml` → `cryptohedge/core/config.py` |
| 2. Spot for 100 cryptos over 3 months, online source | `services/providers/{binance,bundled,synthetic}.py`, `data_acquisition` agent |
| 3. Put/call options, 30-day expiry | same providers (BTC option chain), `data_acquisition` |
| 4. Daily volatility, vol-of-vol, confidence interval, BTC hedge size | `services/volatility.py`, `market_analysis` agent |
| 5. Correlations and instrument selection | `services/correlation.py`, `market_analysis` agent |
| 5.1. Pearson, Spearman, Kendall, DCC-GARCH, cointegration (Johansen/Engle-Granger) | `services/correlation.py` |
| 5.2. Ranking by 5 criteria | `correlation.rank_instruments` |
| 6. Greeks Δ, ν | `services/greeks.py`, `greeks_calculation` agent |
| 6.1. Γ, Θ, ρ, charm (+ vanna, volga) | `services/greeks.py` |
| 6.2. Aggregated greeks (instrument / portfolio) | `domain/greeks.py`, `greeks_calculation` |
| 7. Heston hedge: spot, PnL, fees, Δ/ν hedge, positions, premium, portfolio Δ/ν | `services/hedging_engine.py`, `hedging_decision` agent |
| 7.1. Calibration via MLE / IV surface | `services/calibration.py`, `services/heston_pricing.py` |
| 7.2. Parameter-stability control | `calibration.parameter_stability` |
| 7.3. Black-Scholes and SABR benchmarks | `calibration.{black_scholes_benchmark,sabr_calibrate}` |
| 8.1. ROI, Sharpe, Sortino, Calmar, MDD, Profit Factor, Win Rate, VaR, CVaR, ES, Beta, Alpha, IR | `services/metrics.py` |
| 9.1. MV, Risk Parity, Min Variance, Max Diversification, CVaR | `services/optimization.py` |
| 9.2. Transaction costs and rebalancing frequency | `optimization` + `portfolio_optimization` agent |
| 9.3. Diversified and profitable portfolio (confirmation) | `services/portfolio_backtest.py`, `portfolio_optimization` agent |
| 9.4. Diversification metrics (DR, effective N, HHI) + rebalanced backtest | `portfolio_backtest.{diversification_report,backtest_rebalanced}` |
| 10. Adaptive stop-losses (volatility, ATR, VaR, Heston) | `services/stops.py`, `risk_management` agent |
| 10.2. Dynamic trailing stop | `stops.TrailingStop` |
| 11. Self-testing, data drift, Confidence Score | `services/drift.py`, `self_diagnostic` agent |
| 12. Backtest + time-series calibration (MLE) | `backtesting` agent, `services/walkforward.py` |
| 12.1. Walk-forward without look-ahead | `walkforward.walk_forward_splits` (purge/embargo) |
| 12.2. Survivorship / selection / transaction-cost bias | `backtesting._bias_controls` |
| 12.3. Stress testing | `backtesting._stress_test` |
| 13. Dashboard (price+PnL, greeks, portfolio constituents, rebalancing, diversification, heatmap, stress, metrics, costs) | `dashboard` agent (Plotly HTML, RU+EN) |
| 13.1. Fully Russian and fully English dashboard versions | `services/i18n.py`, `dashboard` agent (`dashboard_ru.html`, `dashboard_en.html`) |
| 14. Natural-language explainability (RU + EN) | `explainability` agent (`explanation.md`, `explanation.en.md`) |
| 15. Multi-agent structure (11 agents) | `cryptohedge/agents/*` |
| Logging, checkpoints, seed, SOLID, tests, lock file | `core/*`, `tests/*`, `uv.lock` |

---

## 2. Project structure

```
hedge fund/
├── config/                 # all system parameters (YAML), no hard-coding
│   ├── system.yaml         # seed, capital, risk, fees, horizons, paths, runtime
│   ├── data.yaml           # data provider and market parameters
│   ├── market.yaml         # market analysis, Heston, greeks, hedging
│   ├── portfolio.yaml      # optimization, risk, backtest, stress scenarios
│   └── diagnostics.yaml    # self-diagnostics, explainability, dashboard, logs
├── cryptohedge/            # the multi-agent system (Clean Architecture)
│   ├── core/               # config, seeding, logging, message/bus, context, agent, orchestrator, checkpoint
│   ├── domain/             # pure entities (market, greeks, portfolio, decisions)
│   ├── services/           # computational use-cases + data providers + i18n + portfolio backtest
│   ├── agents/             # 11 independent agents with a unified interface
│   └── cli.py              # run the pipeline from CLI / notebook
├── pyquant/                # numerical engine (numba/NumPy): Black-Scholes, Heston, vol surface
├── data/raw/               # reproducible bundled dataset
├── scripts/generate_data.py
├── tests/                  # unit + integration tests
├── artifacts/              # run outputs (logs, checkpoints, calibration, dashboards)
├── solution.ipynb          # SINGLE autonomous reproducible solution
├── pyproject.toml          # dependencies (uv/pip), metadata
├── uv.lock                 # lock file (pinned dependencies)
├── requirements.txt        # dependency export for pip
├── README.md / README.en.md
└── ARCHITECTURE.md / ARCHITECTURE.en.md   # diagrams (Mermaid), roles, routing
```

---

## 3. Installation

Requires **Python 3.10 or 3.11** (numba 0.57 constrains NumPy ≤ 1.24, so versions
are pinned).

### Option A — `uv` (recommended)

```bash
pip install uv            # if uv is not installed yet
uv sync                   # creates an environment strictly from uv.lock
```

### Option B — `pip`

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Option C — `pip` from `pyproject.toml`

```bash
pip install -e ".[dev,notebook]"
```

---

## 4. Running

### 4.1. Single solution (notebook)

```bash
jupyter notebook solution.ipynb     # or Jupyter Lab / VS Code
```

The notebook is self-contained: it sets the seed, loads the configuration, runs
all 11 agents through the orchestrator, and shows calibration, greeks, PnL,
metrics, stress tests, the portfolio, the explanation and the interactive dashboard.

### 4.2. Command line

```bash
# (optional) generate/refresh the dataset
python scripts/generate_data.py

# full pipeline
python -m cryptohedge.cli --config config

# useful flags
python -m cryptohedge.cli --provider synthetic   # switch the data source
python -m cryptohedge.cli --reset                # ignore checkpoints
python -m cryptohedge.cli --no-fail-fast         # do not stop on a stage error
```

### 4.3. Tests

```bash
pytest -q                       # all tests
pytest tests/test_core.py -q    # framework (config, seed, checkpoint, bus)
pytest tests/test_quant_services.py -q   # numerical services
pytest tests/test_integration.py -q      # 11-agent interaction + reproducibility
```

---

## 5. Data sources

The provider is selected in `config/data.yaml` (`data.provider`):

- **`bundled`** (default) — reads the reproducible dataset from `data/raw/`; if the
  files are missing it deterministically regenerates them from the `seed`. Fully
  offline.
- **`synthetic`** — generates data on the fly (Heston simulation of BTC spot, a
  correlated universe of 100 assets, a consistent option chain).
- **`binance`** — pulls **live** spot quotes from the public Binance REST API (no
  registration/keys); the option chain is built Heston-consistently around the live
  spot. With no network it automatically falls back to synthetic, so the run stays
  reproducible.

> The "online quotes" requirement is satisfied by the `binance` provider; the
> "all data bundled and reproducible" requirement is satisfied by `bundled`.
> Switching is one line in the config.

---

## 6. Interpreting the results

After a run, artifacts appear in `artifacts/`:

| Path | Content |
|---|---|
| `artifacts/results/dashboard_ru.html` | interactive dashboard — **fully Russian** version |
| `artifacts/results/dashboard_en.html` | interactive dashboard — **fully English** version |
| `artifacts/results/dashboard.html` | default dashboard copy (RU) for backward compatibility |
| `artifacts/results/performance_metrics.json` | ROI, Sharpe, Sortino, Calmar, MDD, VaR, CVaR, Beta, Alpha, IR, … |
| `artifacts/results/hedging_history.parquet` | spot, PnL, fees, Δ/ν hedge, positions per step |
| `artifacts/results/portfolio_constituents.parquet` | portfolio holdings: instruments, weights, expected return, volatility, link to BTC |
| `artifacts/results/portfolio_equity.parquet` | portfolio value dynamics vs the equal-weight benchmark |
| `artifacts/results/portfolio_weights_path.parquet` | portfolio weight evolution under rebalancing |
| `artifacts/results/portfolio_methods.parquet` | comparison of the 5 optimization methods (return, Sharpe, diversification) |
| `artifacts/results/stress_test.parquet` | per-leg PnL decomposition and "hedged vs naked BTC" comparison |
| `artifacts/results/walkforward.parquet` | per-fold walk-forward metrics |
| `artifacts/results/explanation.md` / `explanation.en.md` | natural-language decision explanation (RU / EN) |
| `artifacts/calibration/calibr_data.parquet` | Heston parameters per slice (v0, kappa, theta, eps, rho) |
| `artifacts/calibration/heston_{stability,benchmarks}.json` | stability and comparison vs BS/SABR |
| `artifacts/logs/cryptohedge.jsonl` | structured log of actions, decisions, errors and timings |
| `artifacts/checkpoints/<run_id>/` | checkpoints for recovery |

**How to read the key quantities:**

- **Portfolio Δ → 0** means the price risk is neutralised with spot (green/red
  zones are set by `hedging.delta_green_zone/red_zone`).
- **Vega hedge** shows protection against volatility changes (market "fear").
- **Stress tests** compare the hedged portfolio against a naked BTC position under
  ±5/±10% shocks and a volatility spike; `hedge_effectiveness` ≈ 1 means the hedge
  works.
- **Confidence Score** (0..1) — an integral self-assessment of model adequacy
  (data drift + calibration/forecast degradation + hedge quality + limit compliance).
- **Portfolio: diversification + profitability.** The `portfolio_optimization`
  agent builds an investable portfolio with five methods, runs a rebalanced backtest
  for each (net of fees) and automatically selects the method that is **profitable
  and most diversified**. Diversification is confirmed by metrics: the
  **diversification ratio** (DR ≥ 1, higher is better), the **effective number of
  assets** (inverse Herfindahl index, ideally close to the number of holdings), the
  **maximum weight** and the **HHI** (concentration). On the dashboard these are the
  "Portfolio Constituents", "Portfolio Value Dynamics & Rebalancing", "Portfolio
  Weight Evolution" and "Diversification & Method Comparison" panels.
- **Bilingual dashboards.** `dashboard_ru.html` and `dashboard_en.html` are
  generated with a language switch; the set of languages is configured in
  `config/diagnostics.yaml` (`dashboard.languages`).

---

## 7. Reproducibility

- A single `seed` (`config/system.yaml`) determinises all random processes.
- Dependencies are pinned in `uv.lock` (and `requirements.txt`).
- The dataset is bundled in the repository (`data/raw/`).
- The test `tests/test_integration.py::test_pipeline_reproducible` verifies that
  two independent runs produce identical PnL and hedge size.

```bash
# two runs produce identical results under one seed
python -m cryptohedge.cli --reset
python -m cryptohedge.cli --reset
```

---

## 8. Configuration

All parameters live in `config/*.yaml` and are validated by pydantic models
(`cryptohedge/core/config.py`, `extra="forbid"` — unknown keys are rejected).
Files are merged in layers; values can be overridden via `overrides`
(in the notebook/tests) or CLI flags. The key point: **there are no hard-coded
business parameters in the source code** — capital, risk, fees, windows,
thresholds and weights are all set in YAML.
