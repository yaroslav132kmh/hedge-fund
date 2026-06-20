"""Dashboard Agent.

Role: assemble the monitoring dashboard and persist a fully Russian and a fully
English self-contained HTML file. Panels: synchronized spot + PnL + theta/vega +
hedge; greeks imbalance indicators; the *portfolio* (constituents, value dynamics
with rebalancing, weight evolution, diversification diagnostics); a greeks
heatmap; the stress-test comparison; the key-metrics block; hedge costs and
rebalancing frequency; and the instrument ranking. Section titles are rendered as
HTML headings (not Plotly titles) so labels never overlap the charts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.services.i18n import metric_label, relationship_label, t

_POS = "#66bb6a"
_NEG = "#ef5350"
_PALETTE = ["#4fc3f7", "#66bb6a", "#ffa726", "#ab47bc", "#26c6da", "#ec407a",
            "#9ccc65", "#ff7043", "#5c6bc0", "#26a69a", "#d4e157", "#8d6e63",
            "#42a5f5", "#ffca28", "#7e57c2"]


class DashboardAgent(BaseAgent):
    name = "dashboard"
    consumes = [MessageType.EXPLANATION_READY]
    produces = MessageType.DASHBOARD_READY
    checkpoint_keys = ["dashboard_path", "dashboard_paths"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        cfg = context.config.dashboard
        languages = list(cfg.languages) or ["ru"]

        paths: Dict[str, str] = {}
        rendered: Dict[str, str] = {}
        for lang in languages:
            html = self._render(context, lang, languages)
            rendered[lang] = html
            out = context.root / cfg.output_dir / f"dashboard_{lang}.html"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            paths[lang] = str(out)

        # keep a stable default path (RU first, else first language) for back-compat
        default_lang = "ru" if "ru" in rendered else languages[0]
        default_path = context.root / cfg.output_html
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text(rendered[default_lang], encoding="utf-8")

        context.put("dashboard_paths", paths)
        context.put("dashboard_path", str(default_path))
        log.decision("dashboard generated", languages=languages, **{f"path_{k}": v for k, v in paths.items()})

        return Message(self.produces, self.name, "orchestrator",
                       payload={"paths": paths}, correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ render
    def _render(self, context: AgentContext, lang: str, languages: List[str]) -> str:
        b = context.blackboard
        cfg = context.config.dashboard

        history: pd.DataFrame = b["hedge_history"]
        chain: pd.DataFrame = b.get("chain_greeks", pd.DataFrame())
        stress: pd.DataFrame = b.get("stress_table", pd.DataFrame())
        perf: dict = b.get("backtest_metrics", {})
        rankings_df: pd.DataFrame = b.get("rankings_df", pd.DataFrame())
        trailing: pd.DataFrame = b.get("trailing_stops", pd.DataFrame())
        status = b.get("hedge_status", {})
        constituents: pd.DataFrame = b.get("portfolio_constituents", pd.DataFrame())
        equity: pd.DataFrame = b.get("portfolio_equity", pd.DataFrame())
        weights_path: pd.DataFrame = b.get("portfolio_weights_path", pd.DataFrame())
        rebalances = b.get("portfolio_rebalances", [])
        diversification = b.get("diversification", {})
        method_comparison: pd.DataFrame = b.get("method_comparison", pd.DataFrame())

        # ordered (section-key, figure-or-figures)
        panels = [
            ("sec_timeseries", self._timeseries_fig(history, trailing, cfg, lang)),
            ("sec_greeks", self._greeks_panel(history, status, context, lang)),
            ("sec_portfolio_constituents", self._constituents_table(constituents, lang)),
            ("sec_portfolio_equity", self._equity_fig(equity, rebalances, cfg, lang)),
            ("sec_portfolio_weights", self._weights_fig(weights_path, cfg, lang)),
            ("sec_diversification", self._diversification_figs(diversification, method_comparison, cfg, lang)),
            ("sec_heatmap", self._heatmap(chain, cfg, lang)),
            ("sec_stress", self._stress_fig(stress, cfg, lang)),
            ("sec_metrics", self._metrics_table(perf, lang)),
            ("sec_costs", self._costs_fig(history, cfg, lang)),
            ("sec_rankings", self._rankings_table(rankings_df, lang)),
        ]

        body: List[str] = []
        first = True
        for sec_key, figs in panels:
            if figs is None:
                continue
            fig_list = figs if isinstance(figs, list) else [figs]
            fig_list = [f for f in fig_list if f is not None]
            if not fig_list:
                continue
            body.append(f"<h2>{t(lang, sec_key)}</h2>")
            for fig in fig_list:
                body.append(fig.to_html(full_html=False,
                                        include_plotlyjs=("cdn" if first else False)))
                first = False

        sections = (b.get(f"explanation_sections_{lang}")
                    or b.get("explanation_sections", {}))
        explanation_html = self._explanation_html(sections)
        other = [l for l in languages if l != lang]
        switch = ""
        if other:
            ol = other[0]
            switch = f"<a class='lang' href='dashboard_{ol}.html'>{t(lang, 'lang_switch')}</a>"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return (
            "<!DOCTYPE html><html lang='" + lang + "'><head><meta charset='utf-8'>"
            f"<title>{t(lang, 'doc_title')}</title>"
            "<style>"
            "body{background:#0f1115;color:#e6edf3;font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1320px;}"
            "h1{color:#4fc3f7;margin-bottom:2px;} "
            "h2{color:#80cbc4;border-bottom:1px solid #263238;padding-bottom:6px;margin-top:34px;} "
            "h3{color:#9ccc65;margin-bottom:4px;} p{line-height:1.55;color:#cfd8dc;} "
            ".lang{float:right;color:#0f1115;background:#4fc3f7;padding:6px 12px;border-radius:6px;"
            "text-decoration:none;font-weight:600;} .ts{color:#78909c;font-size:13px;}"
            "</style></head><body>"
            f"{switch}"
            f"<h1>{t(lang, 'main_header')}</h1>"
            f"<div class='ts'>{t(lang, 'generated')}: {stamp}</div>"
            + "".join(body)
            + f"<h2>{t(lang, 'explanation_header')}</h2>"
            + explanation_html
            + "</body></html>"
        )

    # ------------------------------------------------------------------ panels
    def _timeseries_fig(self, h: pd.DataFrame, trailing: pd.DataFrame, cfg, lang: str) -> go.Figure:
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.075,
            subplot_titles=(t(lang, "ts_sub_spot"), t(lang, "ts_sub_pnl"),
                            t(lang, "ts_sub_thetavega"), t(lang, "ts_sub_hedges")),
        )
        fig.add_trace(go.Scatter(x=h["ts"], y=h["spot"], name=t(lang, "ser_spot"),
                                 line=dict(color="#4fc3f7")), 1, 1)
        if trailing is not None and not trailing.empty:
            fig.add_trace(go.Scatter(x=trailing["ts"], y=trailing["stop_price"],
                                     name=t(lang, "ser_trailing_stop"),
                                     line=dict(color="#ef5350", dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["pnl"], name=t(lang, "ser_pnl"),
                                 line=dict(color="#66bb6a")), 2, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["fee"], name=t(lang, "ser_fees_cum"),
                                 line=dict(color="#ffa726")), 2, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["theta"], name=t(lang, "ser_theta"),
                                 line=dict(color="#ab47bc")), 3, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["vega"], name=t(lang, "ser_vega"),
                                 line=dict(color="#26c6da")), 3, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["delta_hedge"], name=t(lang, "ser_delta_hedge"),
                                 line=dict(color="#42a5f5")), 4, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["vega_hedge"], name=t(lang, "ser_vega_hedge"),
                                 line=dict(color="#ec407a")), 4, 1)
        fig.update_layout(template=cfg.theme, height=1050, margin=dict(t=50, b=30, l=60, r=140),
                          legend=dict(orientation="v", yanchor="top", y=1.0, x=1.02))
        return fig

    def _greeks_panel(self, h: pd.DataFrame, status: dict, context, lang: str) -> go.Figure:
        last = h.iloc[-1]
        green = context.config.hedging.delta_green_zone
        red = context.config.hedging.delta_red_zone
        frac = float(status.get("delta_fraction", 0.0))
        top = max(red * 2, 0.3)
        fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.09,
                            specs=[[{"type": "indicator"}] * 4])
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=frac, title={"text": t(lang, "ind_delta_ratio"), "font": {"size": 14}},
            number={"font": {"size": 26}},
            gauge={"axis": {"range": [0, top]}, "bar": {"color": "#263238"},
                   "steps": [{"range": [0, green], "color": "#2e7d32"},
                             {"range": [green, red], "color": "#f9a825"},
                             {"range": [red, top], "color": "#c62828"}]}), 1, 1)
        for i, (key, lbl) in enumerate(
                [("gamma", "ind_gamma"), ("vega", "ind_vega"), ("theta", "ind_theta")], start=2):
            fig.add_trace(go.Indicator(mode="number", value=float(last[key]),
                                       number={"font": {"size": 30}},
                                       title={"text": t(lang, lbl), "font": {"size": 14}}), 1, i)
        fig.update_layout(template=context.config.dashboard.theme, height=300,
                          margin=dict(t=60, b=30, l=30, r=30))
        return fig

    def _constituents_table(self, df: pd.DataFrame, lang: str) -> Optional[go.Figure]:
        if df is None or df.empty:
            return None
        d = df.copy()
        rel = [relationship_label(lang, str(r)) for r in d.get("relationship", ["" for _ in range(len(d))])]
        header = [t(lang, "col_symbol"), t(lang, "col_weight"), t(lang, "col_exp_return"),
                  t(lang, "col_vol"), t(lang, "col_relationship")]
        cells = [
            d["symbol"].tolist(),
            [f"{w:.1%}" for w in d["weight"]],
            [f"{r:.1%}" for r in d["exp_return_annual"]],
            [f"{v:.1%}" for v in d["vol_annual"]],
            rel,
        ]
        fig = go.Figure(go.Table(
            columnwidth=[110, 70, 130, 120, 110],
            header=dict(values=header, fill_color="#263238", font=dict(color="white", size=13), height=30),
            cells=dict(values=cells, fill_color="#1b242b", font=dict(color="#e6edf3", size=12), height=26)))
        fig.update_layout(height=min(560, 90 + 28 * len(d)), margin=dict(t=10, b=10, l=10, r=10))
        return fig

    def _equity_fig(self, equity: pd.DataFrame, rebalances, cfg, lang: str) -> Optional[go.Figure]:
        if equity is None or equity.empty:
            return None
        ts = pd.to_datetime(equity["ts"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=equity["equity"], name=t(lang, "ser_portfolio"),
                                 line=dict(color="#66bb6a", width=2)))
        if "benchmark" in equity.columns:
            fig.add_trace(go.Scatter(x=ts, y=equity["benchmark"], name=t(lang, "ser_benchmark_eqw"),
                                     line=dict(color="#90a4ae", dash="dash")))
        # rebalance markers on the portfolio curve
        if rebalances:
            reb = pd.to_datetime([str(r) for r in rebalances])
            idx = pd.Index(ts)
            eq_at = []
            xs = []
            for r in reb:
                pos = idx.get_indexer([r], method="nearest")[0]
                if pos >= 0:
                    xs.append(ts.iloc[pos]); eq_at.append(equity["equity"].iloc[pos])
            fig.add_trace(go.Scatter(x=xs, y=eq_at, mode="markers", name=t(lang, "ser_rebalance"),
                                     marker=dict(color="#ffca28", size=8, symbol="diamond")))
        fig.update_layout(template=cfg.theme, height=400, margin=dict(t=30, b=40, l=60, r=30),
                          yaxis_title=t(lang, "axis_equity"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        return fig

    def _weights_fig(self, weights_path: pd.DataFrame, cfg, lang: str) -> Optional[go.Figure]:
        if weights_path is None or weights_path.empty:
            return None
        ts = pd.to_datetime(weights_path["ts"])
        asset_cols = [c for c in weights_path.columns if c != "ts"]
        # order by average weight; cap at 18 series to keep the legend readable
        means = weights_path[asset_cols].mean().sort_values(ascending=False)
        show = list(means.head(18).index)
        fig = go.Figure()
        for i, col in enumerate(show):
            fig.add_trace(go.Scatter(x=ts, y=weights_path[col], name=col, mode="lines",
                                     line=dict(width=0.5, color=_PALETTE[i % len(_PALETTE)]),
                                     stackgroup="one"))
        fig.update_layout(template=cfg.theme, height=430, margin=dict(t=30, b=40, l=60, r=30),
                          yaxis_title=t(lang, "axis_weight"), yaxis_range=[0, 1],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
        return fig

    def _diversification_figs(self, div: dict, methods: pd.DataFrame, cfg, lang: str) -> Optional[List[go.Figure]]:
        if not div and (methods is None or methods.empty):
            return None
        figs: List[go.Figure] = []
        if div:
            ind = make_subplots(rows=1, cols=4, horizontal_spacing=0.09,
                                specs=[[{"type": "indicator"}] * 4])
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("diversification_ratio", 0.0)),
                                       number={"valueformat": ".2f", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_div_ratio"), "font": {"size": 13}}), 1, 1)
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("effective_n", 0.0)),
                                       number={"valueformat": ".1f", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_eff_n"), "font": {"size": 13}}), 1, 2)
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("max_weight", 0.0)),
                                       number={"valueformat": ".1%", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_max_weight"), "font": {"size": 13}}), 1, 3)
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("hhi", 0.0)),
                                       number={"valueformat": ".3f", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_hhi"), "font": {"size": 13}}), 1, 4)
            ind.update_layout(template=cfg.theme, height=240, margin=dict(t=50, b=20, l=30, r=30))
            figs.append(ind)

        if methods is not None and not methods.empty:
            m = methods.copy()
            bar = make_subplots(specs=[[{"secondary_y": True}]])
            colors = ["#66bb6a" if c else "#455a64" for c in m.get("chosen", [False] * len(m))]
            bar.add_trace(go.Bar(x=m["method"], y=m["total_return"], name=t(lang, "div_bar_return"),
                                 marker_color=colors), secondary_y=False)
            bar.add_trace(go.Scatter(x=m["method"], y=m["diversification_ratio"], mode="markers+lines",
                                     name=t(lang, "div_bar_dr"), marker=dict(color="#ffca28", size=10)),
                          secondary_y=True)
            bar.update_yaxes(title_text=t(lang, "div_ret_axis"), tickformat=".0%", secondary_y=False)
            bar.update_yaxes(title_text=t(lang, "div_dr_axis"), secondary_y=True)
            bar.update_layout(template=cfg.theme, height=380, margin=dict(t=70, b=40, l=60, r=60),
                              title=dict(text=t(lang, "div_methods_title"), font=dict(size=14),
                                         x=0.0, xanchor="left", y=0.99, yanchor="top"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.32))
            figs.append(bar)
        return figs

    def _heatmap(self, chain: pd.DataFrame, cfg, lang: str) -> Optional[go.Figure]:
        if chain is None or chain.empty:
            return None
        cols = [c for c in ["delta", "gamma", "vega", "theta", "vanna", "volga", "charm"] if c in chain]
        z = chain[cols].to_numpy().T
        fig = go.Figure(go.Heatmap(z=z, x=[f"{m:.2f}" for m in chain["moneyness"]], y=cols,
                                   colorscale="RdBu", zmid=0, colorbar=dict(thickness=14)))
        fig.update_layout(template=cfg.theme, height=360, margin=dict(t=20, b=50, l=70, r=30),
                          xaxis_title=t(lang, "heatmap_x"))
        return fig

    def _stress_fig(self, stress: pd.DataFrame, cfg, lang: str) -> Optional[go.Figure]:
        if stress is None or stress.empty:
            return None
        if "net_hedged_pnl" in stress.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=stress["scenario"], y=stress["unhedged_pnl"],
                                 name=t(lang, "ser_unhedged"), marker_color=_NEG))
            fig.add_trace(go.Bar(x=stress["scenario"], y=stress["net_hedged_pnl"],
                                 name=t(lang, "ser_hedged"), marker_color=_POS))
            fig.update_layout(template=cfg.theme, height=380, barmode="group",
                              margin=dict(t=30, b=40, l=70, r=30), yaxis_title=t(lang, "stress_yaxis"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            return fig
        col = "pnl_usd" if "pnl_usd" in stress.columns else stress.columns[-1]
        colors = [_POS if v >= 0 else _NEG for v in stress[col]]
        fig = go.Figure(go.Bar(x=stress["scenario"], y=stress[col], marker_color=colors))
        fig.update_layout(template=cfg.theme, height=350, margin=dict(t=30, b=40, l=70, r=30),
                          yaxis_title=t(lang, "stress_yaxis"))
        return fig

    def _metrics_table(self, perf: dict, lang: str) -> Optional[go.Figure]:
        if not perf:
            return None
        keys = [k for k in ["roi", "cagr", "sharpe", "sortino", "calmar", "max_drawdown",
                            "profit_factor", "win_rate", "var", "cvar", "expected_shortfall",
                            "beta", "alpha", "information_ratio", "volatility"] if k in perf]
        fig = go.Figure(go.Table(
            columnwidth=[200, 120],
            header=dict(values=[t(lang, "col_metric"), t(lang, "col_value")],
                        fill_color="#263238", font=dict(color="white", size=13), height=30),
            cells=dict(values=[[metric_label(lang, k) for k in keys],
                               [f"{perf[k]:.4f}" for k in keys]],
                       fill_color="#1b242b", font=dict(color="#e6edf3", size=12), height=26)))
        fig.update_layout(height=90 + 28 * len(keys), margin=dict(t=10, b=10, l=10, r=10))
        return fig

    def _costs_fig(self, h: pd.DataFrame, cfg, lang: str) -> go.Figure:
        trades = h["pos_spot"].diff().abs().fillna(0) + h["pos_vega_option"].diff().abs().fillna(0)
        rebs = (trades > 1e-9).astype(int)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=h["ts"], y=h["fee"], name=t(lang, "ser_fees_cum"),
                                 line=dict(color="#ffa726")), secondary_y=False)
        fig.add_trace(go.Bar(x=h["ts"], y=rebs, name=t(lang, "ser_rebalances"),
                             marker_color="#5c6bc0", opacity=0.5), secondary_y=True)
        fig.update_layout(template=cfg.theme, height=340, margin=dict(t=30, b=40, l=60, r=60),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig.update_yaxes(title_text=t(lang, "axis_usd"), secondary_y=False)
        return fig

    def _rankings_table(self, rankings_df: pd.DataFrame, lang: str) -> Optional[go.Figure]:
        if rankings_df is None or rankings_df.empty:
            return None
        df = rankings_df.head(10).copy()
        cols = [c for c in ["symbol", "score", "pearson", "spearman", "kendall", "dcc_mean",
                            "cointegrated", "stability", "relationship"] if c in df.columns]
        values = []
        for c in cols:
            if c == "relationship":
                values.append([relationship_label(lang, str(v)) for v in df[c]])
            elif df[c].dtype.kind in "fc":
                values.append(df[c].round(3).tolist())
            else:
                values.append(df[c].tolist())
        fig = go.Figure(go.Table(
            header=dict(values=cols, fill_color="#263238", font=dict(color="white", size=12), height=30),
            cells=dict(values=values, fill_color="#1b242b", font=dict(color="#e6edf3", size=11), height=24)))
        fig.update_layout(height=90 + 26 * len(df), margin=dict(t=10, b=10, l=10, r=10))
        return fig

    @staticmethod
    def _explanation_html(sections: dict) -> str:
        return "".join(f"<h3>{title}</h3><p>{str(body).replace(chr(10), '<br>')}</p>"
                       for title, body in sections.items())
