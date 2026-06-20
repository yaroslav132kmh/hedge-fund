"""Strongly-typed, file-driven configuration.

All tunable parameters of the system live in ``config/*.yaml`` and are validated
here into immutable :class:`pydantic` models. No business parameter is hard-coded
in the source: agents read everything from :class:`SystemConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class _Frozen(BaseModel):
    """Base model: validated on assignment and forbids unknown keys."""

    model_config = {"frozen": False, "extra": "forbid", "validate_assignment": True}


class InvestmentConfig(_Frozen):
    capital_usd: float = 10_000_000.0
    risk_budget_pct: float = 0.02  # max fraction of capital allowed at risk (VaR)
    transaction_fee_pct: float = 0.0003  # spot/linear taker fee
    option_fee_pct: float = 0.0003  # per-contract fee on the underlying notional
    option_fee_cap_pct: float = 0.125  # fee capped at this fraction of option price


class HorizonsConfig(_Frozen):
    analysis_days: int = 90  # historical look-back window
    forecast_days: int = 1  # forecasting horizon
    trading_days_per_year: int = 365  # crypto trades 24/7


class PathsConfig(_Frozen):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    checkpoint_dir: str = "artifacts/checkpoints"
    calibration_dir: str = "artifacts/calibration"
    results_dir: str = "artifacts/results"
    log_dir: str = "artifacts/logs"

    def ensure(self, root: Path) -> "PathsConfig":
        for field in self.model_fields:
            (root / getattr(self, field)).mkdir(parents=True, exist_ok=True)
        return self


class DataConfig(_Frozen):
    provider: str = "bundled"  # one of: bundled | synthetic | binance
    primary_symbol: str = "BTCUSDT"
    quote_currency: str = "USDT"
    universe_size: int = 100
    symbols: List[str] = Field(default_factory=list)
    bar_interval: str = "1d"
    option_expiry_days: int = 30
    options_universe: List[str] = Field(default_factory=lambda: ["BTCUSDT"])
    n_strikes_per_expiry: int = 11
    strike_width_pct: float = 0.6  # +-60% around forward
    binance_base_url: str = "https://api.binance.com"
    request_timeout_s: float = 15.0
    cache_raw: bool = True


class CorrelationConfig(_Frozen):
    methods: List[str] = Field(
        default_factory=lambda: ["pearson", "spearman", "kendall", "dcc_garch", "cointegration"]
    )
    rolling_window: int = 30
    positive_threshold: float = 0.5
    negative_threshold: float = -0.3
    zero_band: float = 0.1
    dcc_a: float = 0.02
    dcc_b: float = 0.95
    dcc_max_iter: int = 50
    cointegration_method: str = "both"  # engle_granger | johansen | both
    cointegration_pvalue: float = 0.05
    johansen_det_order: int = 0
    johansen_k_ar_diff: int = 1


class RankingWeights(_Frozen):
    correlation: float = 0.30
    stability: float = 0.25
    liquidity: float = 0.15
    hedge_cost: float = 0.15
    risk_reduction: float = 0.15


class MarketAnalysisConfig(_Frozen):
    vol_window: int = 30
    vol_of_vol_window: int = 30
    confidence_level: float = 0.95
    regime_n_states: int = 3
    regime_window: int = 30
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    top_n_hedge_instruments: int = 10


class HestonConfig(_Frozen):
    calibration_method: str = "iv_surface"  # iv_surface | mle
    num_iter: int = 50
    tol: float = 1e-8
    initial_params: List[float] = Field(default_factory=lambda: [0.04, 2.0, 0.04, 0.5, -0.5])
    mle_n_steps: int = 90
    stability_window: int = 10
    stability_max_rel_change: float = 0.5
    benchmarks: List[str] = Field(default_factory=lambda: ["black_scholes", "sabr"])
    sabr_beta: float = 0.5
    flat_yield_fallback: float = 0.0


class GreeksConfig(_Frozen):
    engine: str = "analytical"  # analytical | mc
    spot_bump_pct: float = 0.01
    vol_bump: float = 0.01
    time_bump_days: float = 1.0
    rate_bump: float = 0.0001
    mc_n_paths: int = 50_000
    mc_max_dt: float = 0.0137  # ~5/365
    mc_min_steps: int = 40
    mc_minimum_var: float = 0.01
    greeks_to_compute: List[str] = Field(
        default_factory=lambda: ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"]
    )


class HedgingConfig(_Frozen):
    delta_eps: float = 0.0
    vega_eps: float = 0.0
    delta_green_zone: float = 0.05  # |delta|/capital fraction considered balanced
    delta_red_zone: float = 0.15
    target_delta: float = 0.0
    target_vega: float = 0.0
    hedge_instrument_strike_moneyness: float = 1.0  # ATM call by default
    liability_put_moneyness: float = 0.95  # protective put strike as a fraction of spot
    vega_call_moneyness: float = 1.0  # vega-hedge call strike as a fraction of spot
    calibration_subsample: int = 1  # calibrate every k-th slice (1 = every slice)


class OptimizationConfig(_Frozen):
    method: str = "max_diversification"  # fallback primary method
    methods: List[str] = Field(
        default_factory=lambda: [
            "mean_variance",
            "risk_parity",
            "min_variance",
            "max_diversification",
            "cvar",
        ]
    )
    rebalance_frequency_days: int = 5
    max_turnover: float = 0.5
    transaction_cost_aversion: float = 1.0
    risk_aversion: float = 5.0
    cvar_alpha: float = 0.95
    long_only: bool = True
    max_weight: float = 0.2  # cap concentration to enforce diversification
    # investable portfolio construction / selection
    portfolio_universe_size: int = 15  # number of instruments held in the portfolio
    lookback_days: int = 30  # trailing window for re-estimating weights
    min_expected_return: float = 0.0  # require a profitable portfolio when selecting
    diversification_weight: float = 0.5  # weight of diversification vs Sharpe in selection
    auto_select_method: bool = True  # pick the best method by the selection score


class StopLossConfig(_Frozen):
    enabled: bool = True
    atr_window: int = 14
    atr_multiplier: float = 3.0
    var_multiplier: float = 1.5
    trailing: bool = True
    trailing_atr_multiplier: float = 2.5
    min_stop_pct: float = 0.02
    max_stop_pct: float = 0.25
    recalibrate_window: int = 30


class RiskConfig(_Frozen):
    var_method: str = "historical"  # historical | gaussian | cornish_fisher
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    var_limit_pct: float = 0.05
    max_drawdown_limit_pct: float = 0.25
    leverage_limit: float = 3.0
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)


class BacktestConfig(_Frozen):
    mode: str = "walk_forward"
    train_window: int = 30
    test_window: int = 5
    step: int = 5
    purge: int = 0
    embargo: int = 0
    account_survivorship_bias: bool = True
    account_selection_bias: bool = True
    account_transaction_cost_bias: bool = True
    stress_scenarios: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"name": "crash_-10pct", "spot_shock": -0.10, "vol_shock": 0.50},
            {"name": "crash_-5pct", "spot_shock": -0.05, "vol_shock": 0.25},
            {"name": "rally_+5pct", "spot_shock": 0.05, "vol_shock": -0.10},
            {"name": "rally_+10pct", "spot_shock": 0.10, "vol_shock": -0.20},
            {"name": "vol_spike", "spot_shock": 0.0, "vol_shock": 1.0},
        ]
    )


class DiagnosticConfig(_Frozen):
    drift_method: str = "psi"  # psi | ks
    drift_threshold: float = 0.2
    degradation_metric: str = "rmse"
    degradation_window: int = 10
    degradation_threshold: float = 2.0  # x baseline
    confidence_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "calibration": 0.25,
            "data_drift": 0.20,
            "forecast_error": 0.20,
            "hedge_quality": 0.20,
            "risk_compliance": 0.15,
        }
    )


class ExplainabilityConfig(_Frozen):
    language: str = "ru"
    max_reasons: int = 6
    decimals: int = 4


class DashboardConfig(_Frozen):
    output_html: str = "artifacts/results/dashboard.html"
    output_dir: str = "artifacts/results"  # per-language dashboards go here
    languages: List[str] = Field(default_factory=lambda: ["ru", "en"])
    delta_green_zone: float = 0.05
    delta_red_zone: float = 0.15
    height: int = 2200
    width: int = 1300
    theme: str = "plotly_dark"


class LoggingConfig(_Frozen):
    level: str = "INFO"
    console: bool = True
    jsonl: bool = True
    file_name: str = "cryptohedge.jsonl"
    timing: bool = True


class RuntimeConfig(_Frozen):
    parallel: bool = True
    n_jobs: int = -1
    checkpointing: bool = True
    resume: bool = True


class SystemConfig(_Frozen):
    """Root configuration aggregating all sub-systems."""

    seed: int = 90909090
    run_id: str = "default"
    investment: InvestmentConfig = Field(default_factory=InvestmentConfig)
    horizons: HorizonsConfig = Field(default_factory=HorizonsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    market_analysis: MarketAnalysisConfig = Field(default_factory=MarketAnalysisConfig)
    heston: HestonConfig = Field(default_factory=HestonConfig)
    greeks: GreeksConfig = Field(default_factory=GreeksConfig)
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    diagnostic: DiagnosticConfig = Field(default_factory=DiagnosticConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_dir: str | Path = "config",
    overrides: Optional[Dict[str, Any]] = None,
) -> SystemConfig:
    """Load and validate the configuration.

    All ``*.yaml`` files in ``config_dir`` are read in sorted order and deep-merged.
    A ``main.yaml`` (if present) is always merged last so it can override modules.
    The optional ``overrides`` mapping is applied on top (useful for notebooks/tests).
    """
    config_dir = Path(config_dir)
    merged: Dict[str, Any] = {}

    if config_dir.is_dir():
        files = sorted(p for p in config_dir.glob("*.yaml") if p.name != "main.yaml")
        main = config_dir / "main.yaml"
        if main.exists():
            files.append(main)
        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Config file {path} must contain a top-level mapping")
            merged = _deep_merge(merged, data)

    if overrides:
        merged = _deep_merge(merged, overrides)

    return SystemConfig(**merged)
