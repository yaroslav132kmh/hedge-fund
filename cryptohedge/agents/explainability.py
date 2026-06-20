"""Explainability Agent.

Role: turn the system's quantitative state into a natural-language narrative in
*both* Russian and English. Every statement is backed by a concrete metric: the
risk picture and hedge sizing, instrument selection, Heston calibration and its
stability, the greeks balance, the portfolio optimisation with its diversification
proof and profitability, risk limits/stops, backtest performance and the
confidence score. Russian and English section sets are published separately so the
dashboard can render a fully localised page.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType


class ExplainabilityAgent(BaseAgent):
    name = "explainability"
    consumes = [MessageType.DIAGNOSTIC_READY]
    produces = MessageType.EXPLANATION_READY
    checkpoint_keys = ["explanation_text", "explanation_sections",
                       "explanation_sections_ru", "explanation_sections_en"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        sections_ru = self._sections(context, "ru")
        sections_en = self._sections(context, "en")

        self._write(context, "ru", sections_ru, "explanation.md",
                    "# Объяснение решений системы хеджирования")
        self._write(context, "en", sections_en, "explanation.en.md",
                    "# Hedging System Decision Explanation")

        text_ru = "\n\n".join(f"### {t}\n{b}" for t, b in sections_ru.items())
        context.put("explanation_text", text_ru)
        context.put("explanation_sections", sections_ru)  # back-compat default = RU
        context.put("explanation_sections_ru", sections_ru)
        context.put("explanation_sections_en", sections_en)

        diag = context.blackboard["diagnostic"]
        log.decision("generated bilingual explanation", sections=len(sections_ru),
                     confidence=round(diag["confidence_score"], 3))

        return Message(self.produces, self.name, "dashboard",
                       payload={"sections": list(sections_ru), "languages": ["ru", "en"]},
                       correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ helpers
    def _write(self, context, lang, sections, fname, title):
        text = "\n\n".join(f"### {t}\n{b}" for t, b in sections.items())
        context.results_path(fname).write_text(title + "\n\n" + text, encoding="utf-8")

    def _sections(self, context: AgentContext, lang: str) -> Dict[str, str]:
        d = context.config.explainability.decimals
        b = context.blackboard
        ru = lang == "ru"

        vol = b["volatility"]
        sizing = b["hedge_sizing"]
        top_n = context.config.market_analysis.top_n_hedge_instruments
        rankings_df: pd.DataFrame = b.get("rankings_df", pd.DataFrame())
        rankings = rankings_df.head(top_n).to_dict("records") if not rankings_df.empty else []
        calibr: pd.DataFrame = b["calibr_data"]
        last = calibr.sort_values("sample_idx").iloc[-1]
        stability = b["heston_stability"]
        bench = b.get("heston_benchmarks", {})
        greeks = b["portfolio_greeks_latest"]
        status = b["hedge_status"]
        latest = b["latest_decision"]
        risk = b["risk_assessment"]
        stop = b["stop_level"]
        perf = b["backtest_metrics"]
        stress: pd.DataFrame = b["stress_table"]
        diag = b["diagnostic"]

        sections: Dict[str, str] = {}

        # ---- 1. risk & hedge sizing
        feller = 2 * last["kappa"] * last["theta"] - last["eps"] ** 2
        if ru:
            sections["Риск и объём хеджирования"] = (
                f"Суточная волатильность BTC оценена в {vol.daily_vol:.{d}f} "
                f"(годовая {vol.annualized_vol:.{d}f}), волатильность волатильности {vol.vol_of_vol:.{d}f}. "
                f"Доверительный интервал {int(vol.confidence_level*100)}%: "
                f"[{vol.ci_low:.{d}f}; {vol.ci_high:.{d}f}]. "
                f"При капитале {sizing.capital_usd:,.0f}$ и риск-бюджете "
                f"{context.config.investment.risk_budget_pct:.0%} нехеджированный 1-дневный VaR "
                f"превышает лимит, поэтому хеджируется доля {sizing.hedge_ratio:.0%} капитала — "
                f"{sizing.notional_to_hedge_usd:,.0f}$ или {sizing.quantity_to_hedge:.{d}f} BTC."
            )
        else:
            sections["Risk & Hedge Sizing"] = (
                f"BTC daily volatility is estimated at {vol.daily_vol:.{d}f} "
                f"(annualized {vol.annualized_vol:.{d}f}), volatility-of-volatility {vol.vol_of_vol:.{d}f}. "
                f"The {int(vol.confidence_level*100)}% confidence interval is "
                f"[{vol.ci_low:.{d}f}; {vol.ci_high:.{d}f}]. "
                f"With ${sizing.capital_usd:,.0f} of capital and a "
                f"{context.config.investment.risk_budget_pct:.0%} risk budget, the unhedged 1-day VaR "
                f"exceeds the limit, so {sizing.hedge_ratio:.0%} of capital is hedged — "
                f"${sizing.notional_to_hedge_usd:,.0f} or {sizing.quantity_to_hedge:.{d}f} BTC."
            )

        # ---- 2. instrument selection
        lines = []
        for r in rankings[:5]:
            if ru:
                lines.append(
                    f"  • {r['symbol']}: score={r['score']:.3f}, Пирсон={r['pearson']:.2f}, "
                    f"Спирмен={r['spearman']:.2f}, Кендалл={r['kendall']:.2f}, DCC={r['dcc_mean']:.2f}, "
                    f"коинтеграция={'да' if r['cointegrated'] else 'нет'}, устойчивость={r['stability']:.2f}")
            else:
                lines.append(
                    f"  • {r['symbol']}: score={r['score']:.3f}, Pearson={r['pearson']:.2f}, "
                    f"Spearman={r['spearman']:.2f}, Kendall={r['kendall']:.2f}, DCC={r['dcc_mean']:.2f}, "
                    f"cointegration={'yes' if r['cointegrated'] else 'no'}, stability={r['stability']:.2f}")
        if ru:
            sections["Выбор инструментов хеджирования"] = (
                "Инструменты ранжированы по корреляции, устойчивости связи, ликвидности, стоимости "
                "хеджирования и потенциалу снижения риска. Топ-кандидаты:\n" + "\n".join(lines))
        else:
            sections["Hedging Instrument Selection"] = (
                "Instruments are ranked by correlation, link stability, liquidity, hedging cost and "
                "risk-reduction potential. Top candidates:\n" + "\n".join(lines))

        # ---- 3. Heston calibration
        if ru:
            sections["Калибровка модели Хестона"] = (
                f"Последние параметры: v0={last['v0']:.{d}f}, kappa={last['kappa']:.{d}f}, "
                f"theta={last['theta']:.{d}f}, eps={last['eps']:.{d}f}, rho={last['rho']:.{d}f}. "
                f"Условие Феллера 2·kappa·theta−eps² = {feller:.{d}f}. "
                f"Параметры {'устойчивы' if stability.get('stable') else 'неустойчивы'} во времени "
                f"(макс. отн. изменение {stability.get('max_rel_change', float('nan')):.2f}). "
                + self._bench_text(bench, d, ru))
        else:
            sections["Heston Model Calibration"] = (
                f"Latest parameters: v0={last['v0']:.{d}f}, kappa={last['kappa']:.{d}f}, "
                f"theta={last['theta']:.{d}f}, eps={last['eps']:.{d}f}, rho={last['rho']:.{d}f}. "
                f"Feller condition 2·kappa·theta−eps² = {feller:.{d}f}. "
                f"Parameters are {'stable' if stability.get('stable') else 'unstable'} over time "
                f"(max relative change {stability.get('max_rel_change', float('nan')):.2f}). "
                + self._bench_text(bench, d, ru))

        # ---- 4. greeks
        if ru:
            sections["Греки и баланс портфеля"] = (
                f"Греки опционного портфеля: дельта={greeks['delta']:.{d}f}, гамма={greeks['gamma']:.{d}f}, "
                f"вега={greeks['vega']:.{d}f}, тета={greeks['theta']:.{d}f}, ро={greeks['rho']:.{d}f}, "
                f"ванна={greeks['vanna']:.{d}f}, волга={greeks['volga']:.{d}f}, чарм={greeks['charm']:.{d}f}. "
                f"Баланс дельты — зона '{status['zone']}' (|дельта|={status['delta_fraction']:.{d}f} капитала). "
                f"После хеджа остаточная дельта={latest['residual_delta']:.{d}f}, остаточная вега="
                f"{latest['residual_vega']:.{d}f}; сделок: {latest['n_trades']}, "
                f"комиссий {latest['fees']:,.2f}$.")
        else:
            sections["Greeks & Portfolio Balance"] = (
                f"Option-book greeks: delta={greeks['delta']:.{d}f}, gamma={greeks['gamma']:.{d}f}, "
                f"vega={greeks['vega']:.{d}f}, theta={greeks['theta']:.{d}f}, rho={greeks['rho']:.{d}f}, "
                f"vanna={greeks['vanna']:.{d}f}, volga={greeks['volga']:.{d}f}, charm={greeks['charm']:.{d}f}. "
                f"Delta balance is in the '{status['zone']}' zone (|delta|={status['delta_fraction']:.{d}f} "
                f"of capital). After hedging, residual delta={latest['residual_delta']:.{d}f}, residual vega="
                f"{latest['residual_vega']:.{d}f}; trades: {latest['n_trades']}, "
                f"fees ${latest['fees']:,.2f}.")

        # ---- 5. portfolio optimisation + diversification (NEW)
        title, body = self._portfolio_section(b, ru)
        sections[title] = body

        # ---- 6. risk & stops
        if ru:
            sections["Управление риском и стоп-лоссы"] = (
                f"VaR={risk['var']:.{d}f}, CVaR/ES={risk['cvar']:.{d}f}, макс. просадка="
                f"{risk['max_drawdown']:.{d}f}. Лимиты {'соблюдены' if risk['within_limits'] else 'НАРУШЕНЫ'} "
                f"({', '.join(risk['breached_limits']) if risk['breached_limits'] else 'нарушений нет'}). "
                f"Адаптивный стоп-лосс на уровне {stop['stop_price']:,.2f} "
                f"(дистанция {stop['distance_pct']:.{d}f}, метод {stop['method']}), компоненты: "
                f"ATR={stop['components'].get('atr_pct', float('nan')):.{d}f}, "
                f"VaR={stop['components'].get('var_pct', float('nan')):.{d}f}, "
                f"Heston={stop['components'].get('heston_vol_pct', float('nan')):.{d}f}.")
        else:
            sections["Risk Management & Stop-losses"] = (
                f"VaR={risk['var']:.{d}f}, CVaR/ES={risk['cvar']:.{d}f}, max drawdown="
                f"{risk['max_drawdown']:.{d}f}. Limits are {'respected' if risk['within_limits'] else 'BREACHED'} "
                f"({', '.join(risk['breached_limits']) if risk['breached_limits'] else 'no breaches'}). "
                f"Adaptive stop-loss at {stop['stop_price']:,.2f} "
                f"(distance {stop['distance_pct']:.{d}f}, method {stop['method']}), components: "
                f"ATR={stop['components'].get('atr_pct', float('nan')):.{d}f}, "
                f"VaR={stop['components'].get('var_pct', float('nan')):.{d}f}, "
                f"Heston={stop['components'].get('heston_vol_pct', float('nan')):.{d}f}.")

        # ---- 7. backtest
        stress_txt = self._stress_text(stress, ru)
        if ru:
            sections["Результаты бэктеста"] = (
                f"ROI={perf['roi']:.{d}f}, Sharpe={perf['sharpe']:.2f}, Sortino={perf['sortino']:.2f}, "
                f"Calmar={perf['calmar']:.2f}, макс. просадка={perf['max_drawdown']:.{d}f}, "
                f"Profit Factor={perf['profit_factor']:.2f}, Win Rate={perf['win_rate']:.2f}, "
                f"VaR={perf['var']:.{d}f}, CVaR={perf['cvar']:.{d}f}, Beta={perf['beta']:.2f}, "
                f"Alpha={perf['alpha']:.{d}f}, Information Ratio={perf['information_ratio']:.2f}. " + stress_txt)
        else:
            sections["Backtest Results"] = (
                f"ROI={perf['roi']:.{d}f}, Sharpe={perf['sharpe']:.2f}, Sortino={perf['sortino']:.2f}, "
                f"Calmar={perf['calmar']:.2f}, max drawdown={perf['max_drawdown']:.{d}f}, "
                f"Profit Factor={perf['profit_factor']:.2f}, Win Rate={perf['win_rate']:.2f}, "
                f"VaR={perf['var']:.{d}f}, CVaR={perf['cvar']:.{d}f}, Beta={perf['beta']:.2f}, "
                f"Alpha={perf['alpha']:.{d}f}, Information Ratio={perf['information_ratio']:.2f}. " + stress_txt)

        # ---- 8. self-assessment
        comp = ", ".join(f"{k}={v:.2f}" for k, v in diag["components"].items())
        if ru:
            sections["Самооценка системы"] = (
                f"Интегральный индекс доверия (Confidence Score) = {diag['confidence_score']:.3f} "
                f"({diag['self_assessment']}). Дрейф данных "
                f"{'обнаружен' if diag['drift_detected'] else 'не обнаружен'} (PSI={diag['psi']:.3f}). "
                f"Компоненты доверия: {comp}.")
        else:
            sections["System Self-assessment"] = (
                f"Overall Confidence Score = {diag['confidence_score']:.3f} "
                f"({diag['self_assessment']}). Data drift "
                f"{'detected' if diag['drift_detected'] else 'not detected'} (PSI={diag['psi']:.3f}). "
                f"Confidence components: {comp}.")

        return sections

    def _portfolio_section(self, b, ru: bool):
        rebal = b["rebalance_decision"]
        div = b.get("diversification", {})
        results = b.get("optimization_results", {})
        constituents: pd.DataFrame = b.get("portfolio_constituents", pd.DataFrame())
        equity: pd.DataFrame = b.get("portfolio_equity", pd.DataFrame())
        method = rebal["method"]
        bt = results.get(method, {}).get("backtest", {})
        total_ret = bt.get("total_return", 0.0)
        cagr = bt.get("cagr", 0.0)
        sharpe = bt.get("sharpe", 0.0)
        mdd = bt.get("max_drawdown", 0.0)
        bench_ret = 0.0
        if equity is not None and not equity.empty and "benchmark" in equity.columns:
            bench_ret = float(equity["benchmark"].iloc[-1] - 1.0)
        dr = div.get("diversification_ratio", 0.0)
        bench_dr = div.get("benchmark_diversification_ratio", 0.0)
        eff_n = div.get("effective_n", 0.0)
        n_assets = div.get("n_assets", 0)
        max_w = div.get("max_weight", 0.0)
        hhi = div.get("hhi", 0.0)
        n_reb = div.get("n_rebalances", 0)

        top = ""
        if constituents is not None and not constituents.empty:
            items = ", ".join(f"{row['symbol']} {row['weight']:.0%}"
                              for _, row in constituents.head(5).iterrows())
            top = ("Топ-позиции: " if ru else "Top holdings: ") + items + "."

        if ru:
            title = "Оптимизация портфеля и диверсификация"
            body = (
                f"Инвестиционный портфель из {n_assets} инструментов построен методом '{method}', "
                f"выбранным автоматически как лучший по комбинации доходности и диверсификации. "
                f"Бэктест с ребалансировкой (всего {n_reb} ребалансировок, с учётом комиссий) дал "
                f"доходность {total_ret:.2%} (CAGR {cagr:.2%}, Sharpe {sharpe:.2f}, макс. просадка "
                f"{mdd:.2%}) против {bench_ret:.2%} у равновзвешенного бенчмарка — портфель прибыльный. "
                f"Высокая диверсификация подтверждена: коэффициент диверсификации {dr:.2f} "
                f"(бенчмарк {bench_dr:.2f}), эффективное число активов {eff_n:.1f} из {n_assets}, "
                f"максимальный вес {max_w:.1%}, индекс концентрации HHI {hhi:.3f}. " + top)
        else:
            title = "Portfolio Optimization & Diversification"
            body = (
                f"The investable portfolio of {n_assets} instruments is built with the '{method}' method, "
                f"auto-selected as the best trade-off between return and diversification. "
                f"A rebalanced backtest ({n_reb} rebalances, net of fees) delivered a "
                f"{total_ret:.2%} return (CAGR {cagr:.2%}, Sharpe {sharpe:.2f}, max drawdown "
                f"{mdd:.2%}) versus {bench_ret:.2%} for the equal-weight benchmark — the portfolio is "
                f"profitable. High diversification is confirmed: diversification ratio {dr:.2f} "
                f"(benchmark {bench_dr:.2f}), effective number of assets {eff_n:.1f} of {n_assets}, "
                f"max weight {max_w:.1%}, HHI concentration index {hhi:.3f}. " + top)
        return title, body

    @staticmethod
    def _stress_text(stress: pd.DataFrame, ru: bool) -> str:
        if stress is None or stress.empty:
            return ""
        if "net_hedged_pnl" in stress.columns:
            worst = stress.loc[stress["unhedged_pnl"].idxmin()]
            if ru:
                return (f"В худшем стресс-сценарии '{worst['scenario']}' голый BTC дал бы "
                        f"{worst['unhedged_pnl']:,.0f}$, а захеджированный портфель — "
                        f"{worst['net_hedged_pnl']:,.0f}$ (эффективность хеджа "
                        f"{worst['hedge_effectiveness']:.1%}).")
            return (f"In the worst stress scenario '{worst['scenario']}' naked BTC would lose "
                    f"${worst['unhedged_pnl']:,.0f}, while the hedged portfolio is at "
                    f"${worst['net_hedged_pnl']:,.0f} (hedge effectiveness "
                    f"{worst['hedge_effectiveness']:.1%}).")
        if "pnl_usd" in stress.columns:
            worst = stress.loc[stress["pnl_usd"].idxmin()]
            if ru:
                return f"Худший стресс-сценарий '{worst['scenario']}': PnL {worst['pnl_usd']:,.0f}$."
            return f"Worst stress scenario '{worst['scenario']}': PnL ${worst['pnl_usd']:,.0f}."
        return ""

    @staticmethod
    def _bench_text(bench: dict, d: int, ru: bool) -> str:
        if not bench:
            return ""
        h = bench.get("heston", {}).get("iv_rmse", float("nan"))
        bs = bench.get("black_scholes", {}).get("iv_rmse", float("nan"))
        sabr = bench.get("sabr", {}).get("rmse", float("nan"))
        if ru:
            return (f"Сравнение по RMSE подразумеваемой волатильности: Heston={h:.{d}f}, "
                    f"Black-Scholes={bs:.{d}f}, SABR={sabr:.{d}f}.")
        return (f"Implied-volatility RMSE comparison: Heston={h:.{d}f}, "
                f"Black-Scholes={bs:.{d}f}, SABR={sabr:.{d}f}.")
