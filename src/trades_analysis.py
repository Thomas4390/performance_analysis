"""
Module for trade-level analytics and visualizations.

Provides summary statistics (win rate, P&L distribution, streaks, etc.)
and Plotly charts for exploring trade performance across strategies.

Usage:
    analyzer = TradesAnalyzer({"strategy_a": df_a, "strategy_b": df_b})
    stats = analyzer.calculate_summary_stats()
    fig = analyzer.plot_cumulative_pnl()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOT_TEMPLATE, PLOT_COLORS


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeSummaryStats:
    """Summary statistics for a set of trades."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    avg_pnl: float = 0.0
    median_pnl: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_hours: float = 0.0
    median_holding_hours: float = 0.0
    max_holding_hours: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown: float = 0.0
    expectancy: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# =============================================================================
# TRADES ANALYZER
# =============================================================================

class TradesAnalyzer:
    """
    Analyze trade-level data across one or more strategies.

    Args:
        trades: Dict mapping strategy name to its trades DataFrame
                (output of TradesLoader.load).
    """

    def __init__(self, trades: dict[str, pd.DataFrame]):
        self._trades = trades
        # Build combined DataFrame with Strategy column
        parts = []
        for name, df in trades.items():
            chunk = df.copy()
            chunk["Strategy"] = name
            parts.append(chunk)
        self._combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    @property
    def combined(self) -> pd.DataFrame:
        return self._combined

    # -----------------------------------------------------------------
    # Compute methods
    # -----------------------------------------------------------------

    def calculate_summary_stats(self, strategy: Optional[str] = None) -> TradeSummaryStats:
        """Calculate summary statistics for all trades or a single strategy."""
        if strategy is not None:
            df = self._trades.get(strategy)
            if df is None or df.empty:
                return TradeSummaryStats()
        else:
            df = self._combined

        if df.empty:
            return TradeSummaryStats()

        wins = df[df["IsWin"]]
        losses = df[~df["IsWin"]]
        total_wins = wins["P&L"].sum() if not wins.empty else 0.0
        total_losses = abs(losses["P&L"].sum()) if not losses.empty else 0.0
        total_fees = df["Fees"].sum() if "Fees" in df.columns else 0.0

        # Streaks
        longest_win, longest_loss = self._compute_streaks(df["IsWin"])

        return TradeSummaryStats(
            total_trades=len(df),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(df) if len(df) > 0 else 0.0,
            total_pnl=df["P&L"].sum(),
            total_fees=total_fees,
            net_pnl=df["P&L"].sum() - total_fees,
            avg_pnl=df["P&L"].mean(),
            median_pnl=df["P&L"].median(),
            max_win=wins["P&L"].max() if not wins.empty else 0.0,
            max_loss=losses["P&L"].min() if not losses.empty else 0.0,
            avg_win=wins["P&L"].mean() if not wins.empty else 0.0,
            avg_loss=losses["P&L"].mean() if not losses.empty else 0.0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float("inf"),
            avg_holding_hours=df["Holding Hours"].mean() if "Holding Hours" in df.columns else 0.0,
            median_holding_hours=df["Holding Hours"].median() if "Holding Hours" in df.columns else 0.0,
            max_holding_hours=df["Holding Hours"].max() if "Holding Hours" in df.columns else 0.0,
            avg_drawdown=df["Drawdown"].mean() if "Drawdown" in df.columns else 0.0,
            max_drawdown=df["Drawdown"].max() if "Drawdown" in df.columns else 0.0,
            expectancy=(len(wins) / len(df) * (wins["P&L"].mean() if not wins.empty else 0) +
                        len(losses) / len(df) * (losses["P&L"].mean() if not losses.empty else 0))
            if len(df) > 0 else 0.0,
            longest_win_streak=longest_win,
            longest_loss_streak=longest_loss,
        )

    def calculate_per_strategy_stats(self) -> pd.DataFrame:
        """Return a DataFrame with one row per strategy."""
        rows = []
        for name in self._trades:
            stats = self.calculate_summary_stats(strategy=name)
            d = stats.to_dict()
            d["Strategy"] = name
            rows.append(d)
        return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()

    def calculate_pnl_by_day_of_week(self) -> pd.DataFrame:
        """Aggregate P&L by day of week (Entry Time)."""
        if self._combined.empty:
            return pd.DataFrame()
        df = self._combined.copy()
        df["DayOfWeek"] = df["Entry Time"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        agg = df.groupby("DayOfWeek")["P&L"].agg(["sum", "mean", "count"]).reindex(day_order).dropna()
        agg.columns = ["Total P&L", "Avg P&L", "Count"]
        return agg

    def calculate_pnl_by_month(self) -> pd.DataFrame:
        """Aggregate P&L by calendar month."""
        if self._combined.empty:
            return pd.DataFrame()
        df = self._combined.copy()
        df["Month"] = df["Entry Time"].dt.tz_localize(None).dt.to_period("M")
        agg = df.groupby("Month")["P&L"].agg(["sum", "mean", "count"])
        agg.columns = ["Total P&L", "Avg P&L", "Count"]
        agg.index = agg.index.astype(str)
        return agg

    # -----------------------------------------------------------------
    # Plot methods
    # -----------------------------------------------------------------

    def plot_pnl_distribution(self) -> go.Figure:
        """Histogram of trade P&L, coloured by win/loss."""
        df = self._combined
        wins = df[df["IsWin"]]["P&L"]
        losses = df[~df["IsWin"]]["P&L"]

        fig = go.Figure()
        if not wins.empty:
            fig.add_trace(go.Histogram(x=wins, name="Wins", marker_color="#2ca02c", opacity=0.75))
        if not losses.empty:
            fig.add_trace(go.Histogram(x=losses, name="Losses", marker_color="#d62728", opacity=0.75))
        fig.update_layout(
            title="P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            barmode="overlay",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_cumulative_pnl(self) -> go.Figure:
        """Cumulative P&L line chart per strategy + combined."""
        fig = go.Figure()
        for i, (name, df) in enumerate(self._trades.items()):
            sorted_df = df.sort_values("Entry Time")
            cum_pnl = sorted_df["P&L"].cumsum()
            fig.add_trace(go.Scatter(
                x=sorted_df["Entry Time"],
                y=cum_pnl,
                mode="lines",
                name=name,
                line=dict(color=PLOT_COLORS[i % len(PLOT_COLORS)]),
            ))
        if len(self._trades) > 1:
            combined_sorted = self._combined.sort_values("Entry Time")
            cum_pnl = combined_sorted["P&L"].cumsum()
            fig.add_trace(go.Scatter(
                x=combined_sorted["Entry Time"],
                y=cum_pnl,
                mode="lines",
                name="Combined",
                line=dict(color="white", width=2, dash="dash"),
            ))
        fig.update_layout(
            title="Cumulative P&L",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_win_loss_streaks(self) -> go.Figure:
        """Bar chart of win/loss streak lengths with date labels."""
        df = self._combined.sort_values("Entry Time").reset_index(drop=True)
        if df.empty:
            return go.Figure()

        streaks, streak_dates = self._get_streak_series_with_dates(
            df["IsWin"], df["Entry Time"]
        )
        labels = [
            f"{d[0].strftime('%Y-%m-%d')} → {d[1].strftime('%Y-%m-%d')}"
            for d in streak_dates
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(streaks))),
            y=[s if s > 0 else 0 for s in streaks],
            name="Win Streak",
            marker_color="#2ca02c",
            text=[l if s > 0 else "" for s, l in zip(streaks, labels)],
            hovertemplate="Streak: %{y}<br>%{text}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=list(range(len(streaks))),
            y=[s if s < 0 else 0 for s in streaks],
            name="Loss Streak",
            marker_color="#d62728",
            text=[l if s < 0 else "" for s, l in zip(streaks, labels)],
            hovertemplate="Streak: %{y}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            title="Win / Loss Streaks",
            xaxis_title="Streak #",
            yaxis_title="Streak Length",
            barmode="relative",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_pnl_by_day_of_week(self) -> go.Figure:
        """Bar chart of total P&L by day of week."""
        agg = self.calculate_pnl_by_day_of_week()
        if agg.empty:
            return go.Figure()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in agg["Total P&L"]]
        fig = go.Figure(go.Bar(
            x=agg.index,
            y=agg["Total P&L"],
            marker_color=colors,
            text=[f"${v:,.0f}" for v in agg["Total P&L"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="P&L by Day of Week",
            yaxis_title="Total P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_pnl_by_month(self) -> go.Figure:
        """Bar chart of total P&L by month."""
        agg = self.calculate_pnl_by_month()
        if agg.empty:
            return go.Figure()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in agg["Total P&L"]]
        fig = go.Figure(go.Bar(
            x=agg.index,
            y=agg["Total P&L"],
            marker_color=colors,
        ))
        fig.update_layout(
            title="P&L by Month",
            xaxis_title="Month",
            yaxis_title="Total P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_holding_period_distribution(self) -> go.Figure:
        """Histogram of holding periods in hours."""
        if "Holding Hours" not in self._combined.columns:
            return go.Figure()
        fig = go.Figure(go.Histogram(
            x=self._combined["Holding Hours"],
            marker_color=PLOT_COLORS[0],
            opacity=0.75,
        ))
        fig.update_layout(
            title="Holding Period Distribution",
            xaxis_title="Hours",
            yaxis_title="Count",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_drawdown_distribution(self) -> go.Figure:
        """Histogram of per-trade drawdowns."""
        if "Drawdown" not in self._combined.columns:
            return go.Figure()
        fig = go.Figure(go.Histogram(
            x=self._combined["Drawdown"],
            marker_color="#d62728",
            opacity=0.75,
        ))
        fig.update_layout(
            title="Per-Trade Drawdown Distribution",
            xaxis_title="Drawdown ($)",
            yaxis_title="Count",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_pnl_heatmap(self) -> go.Figure:
        """Year x Month heatmap of total P&L (RdYlGn colourscale)."""
        df = self._combined.copy()
        if df.empty:
            return go.Figure()
        df["Year"] = df["Entry Time"].dt.year
        df["Month"] = df["Entry Time"].dt.month
        pivot = df.pivot_table(values="P&L", index="Year", columns="Month", aggfunc="sum")
        pivot = pivot.reindex(columns=range(1, 13))

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"${v:,.0f}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="Year: %{y}<br>Month: %{x}<br>P&L: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title="Monthly P&L Heatmap",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_trade_scatter(self) -> go.Figure:
        """Scatter: Holding Period vs P&L, sized by quantity."""
        df = self._combined
        if df.empty or "Holding Hours" not in df.columns:
            return go.Figure()

        qty = df["Quantity"].fillna(1) if "Quantity" in df.columns else pd.Series(1, index=df.index)
        colors = ["#2ca02c" if w else "#d62728" for w in df["IsWin"]]

        fig = go.Figure(go.Scatter(
            x=df["Holding Hours"],
            y=df["P&L"],
            mode="markers",
            marker=dict(
                size=np.clip(qty / qty.max() * 20, 4, 30) if qty.max() > 0 else 6,
                color=colors,
                opacity=0.6,
            ),
            text=df.get("Symbols", ""),
            hovertemplate="Holding: %{x:.1f}h<br>P&L: $%{y:,.0f}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            title="Trade Scatter (Holding Period vs P&L)",
            xaxis_title="Holding Period (hours)",
            yaxis_title="P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_equity_curve(self) -> go.Figure:
        """Reconstructed equity curve from cumulative P&L with drawdown overlay."""
        df = self._combined.sort_values("Exit Time").reset_index(drop=True)
        if df.empty:
            return go.Figure()

        cum_pnl = df["P&L"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(
            x=df["Exit Time"], y=cum_pnl,
            mode="lines", name="Equity",
            line=dict(color=PLOT_COLORS[0]),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Exit Time"], y=running_max,
            mode="lines", name="High Water Mark",
            line=dict(color="grey", dash="dot", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Exit Time"], y=drawdown,
            mode="lines", name="Drawdown",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
            fillcolor="rgba(214,39,40,0.25)",
        ), row=2, col=1)
        fig.update_layout(
            title="Reconstructed Equity Curve",
            template=PLOT_TEMPLATE,
            height=650,
        )
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
        return fig

    def plot_pnl_sequential(self) -> go.Figure:
        """P&L vs trade number to detect performance drift."""
        df = self._combined.sort_values("Entry Time").reset_index(drop=True)
        if df.empty:
            return go.Figure()

        trade_num = list(range(1, len(df) + 1))
        colors = ["#2ca02c" if w else "#d62728" for w in df["IsWin"]]

        # Rolling average (window = 20 trades)
        window = min(20, len(df))
        rolling_avg = df["P&L"].rolling(window, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=trade_num, y=df["P&L"].values,
            name="P&L",
            marker_color=colors,
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=trade_num, y=rolling_avg.values,
            mode="lines", name=f"Rolling Avg ({window})",
            line=dict(color="#f39c12", width=2.5),
        ))
        fig.update_layout(
            title="P&L by Trade Number (Sequential)",
            xaxis_title="Trade #",
            yaxis_title="P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_rolling_win_rate(self, window: int = 30) -> go.Figure:
        """Rolling win rate over a sliding window of N trades."""
        df = self._combined.sort_values("Entry Time").reset_index(drop=True)
        if df.empty:
            return go.Figure()

        window = min(window, len(df))
        rolling_wr = df["IsWin"].astype(float).rolling(window, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Entry Time"], y=rolling_wr,
            mode="lines", name=f"Rolling Win Rate ({window} trades)",
            line=dict(color=PLOT_COLORS[0]),
        ))
        fig.add_hline(
            y=df["IsWin"].mean(), line_dash="dash", line_color="grey",
            annotation_text=f"Overall: {df['IsWin'].mean():.1%}",
        )
        fig.update_layout(
            title=f"Rolling Win Rate ({window}-Trade Window)",
            xaxis_title="Date",
            yaxis_title="Win Rate",
            yaxis_tickformat=".0%",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_pnl_by_hour(self) -> go.Figure:
        """Bar chart of P&L by hour of entry."""
        df = self._combined.copy()
        if df.empty:
            return go.Figure()

        df["Hour"] = df["Entry Time"].dt.hour
        agg = df.groupby("Hour")["P&L"].agg(["sum", "mean", "count"])
        agg.columns = ["Total P&L", "Avg P&L", "Count"]

        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in agg["Total P&L"]]
        fig = go.Figure(go.Bar(
            x=agg.index,
            y=agg["Total P&L"],
            marker_color=colors,
            text=[f"n={int(c)}" for c in agg["Count"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="P&L by Hour of Entry",
            xaxis_title="Hour (UTC)",
            yaxis_title="Total P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_mae_mfe(self) -> go.Figure:
        """MAE (Drawdown) vs P&L scatter — evaluate exit quality."""
        df = self._combined
        if df.empty or "Drawdown" not in df.columns:
            return go.Figure()

        colors = ["#2ca02c" if w else "#d62728" for w in df["IsWin"]]

        fig = go.Figure(go.Scatter(
            x=df["Drawdown"],
            y=df["P&L"],
            mode="markers",
            marker=dict(color=colors, opacity=0.5, size=6),
            hovertemplate="Drawdown: $%{x:,.0f}<br>P&L: $%{y:,.0f}<extra></extra>",
        ))
        # Add diagonal reference line (breakeven: P&L = -Drawdown)
        max_dd = df["Drawdown"].max()
        fig.add_trace(go.Scatter(
            x=[0, max_dd], y=[0, -max_dd],
            mode="lines", name="Breakeven",
            line=dict(color="grey", dash="dash", width=1),
            showlegend=True,
        ))
        fig.update_layout(
            title="MAE / MFE — Drawdown vs P&L",
            xaxis_title="Max Adverse Excursion / Drawdown ($)",
            yaxis_title="P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_risk_reward(self) -> go.Figure:
        """Risk/Reward scatter — drawdown vs P&L sized by quantity."""
        df = self._combined
        if df.empty or "Drawdown" not in df.columns:
            return go.Figure()

        qty = df["Quantity"].fillna(1) if "Quantity" in df.columns else pd.Series(1, index=df.index)
        colors = ["#2ca02c" if w else "#d62728" for w in df["IsWin"]]

        # Risk/reward ratio per trade
        rr = df["P&L"] / df["Drawdown"].replace(0, np.nan)

        fig = go.Figure(go.Scatter(
            x=df["Drawdown"],
            y=df["P&L"],
            mode="markers",
            marker=dict(
                size=np.clip(qty / qty.max() * 20, 4, 25) if qty.max() > 0 else 6,
                color=colors,
                opacity=0.55,
            ),
            text=[f"R/R: {r:.2f}" if pd.notna(r) else "" for r in rr],
            hovertemplate="Drawdown: $%{x:,.0f}<br>P&L: $%{y:,.0f}<br>%{text}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="grey", line_dash="dash", line_width=1)
        fig.update_layout(
            title="Risk / Reward per Trade",
            xaxis_title="Drawdown ($)",
            yaxis_title="P&L ($)",
            template=PLOT_TEMPLATE,
            height=550,
        )
        return fig

    def plot_strategy_comparison(self) -> go.Figure:
        """Grouped bar chart comparing strategies on key metrics."""
        per = self.calculate_per_strategy_stats()
        if per.empty or len(per) < 2:
            return go.Figure()

        metrics = ["total_pnl", "avg_pnl", "win_rate", "profit_factor"]
        labels = ["Total P&L", "Avg P&L", "Win Rate", "Profit Factor"]

        fig = make_subplots(rows=2, cols=2, subplot_titles=labels)
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            r, c = divmod(i, 2)
            for j, strategy in enumerate(per.index):
                val = per.loc[strategy, metric]
                fig.add_trace(go.Bar(
                    x=[strategy],
                    y=[val],
                    name=strategy if i == 0 else None,
                    marker_color=PLOT_COLORS[j % len(PLOT_COLORS)],
                    showlegend=(i == 0),
                ), row=r + 1, col=c + 1)
        fig.update_layout(
            title="Strategy Comparison",
            template=PLOT_TEMPLATE,
            barmode="group",
            height=600,
        )
        return fig

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_streaks(is_win: pd.Series) -> tuple[int, int]:
        """Return (longest_win_streak, longest_loss_streak)."""
        if is_win.empty:
            return 0, 0
        max_win = max_loss = cur_win = cur_loss = 0
        for w in is_win:
            if w:
                cur_win += 1
                cur_loss = 0
            else:
                cur_loss += 1
                cur_win = 0
            max_win = max(max_win, cur_win)
            max_loss = max(max_loss, cur_loss)
        return max_win, max_loss

    @staticmethod
    def _get_streak_series(is_win: pd.Series) -> list[int]:
        """Return a list of signed streak lengths (+win, -loss)."""
        if is_win.empty:
            return []
        streaks = []
        current = 0
        prev = None
        for w in is_win:
            if w == prev:
                current += 1 if w else -1
            else:
                if prev is not None:
                    streaks.append(current)
                current = 1 if w else -1
            prev = w
        streaks.append(current)
        return streaks

    @staticmethod
    def _get_streak_series_with_dates(
        is_win: pd.Series, dates: pd.Series
    ) -> tuple[list[int], list[tuple]]:
        """Return streak lengths and (start_date, end_date) for each streak."""
        if is_win.empty:
            return [], []
        streaks = []
        streak_dates = []
        current = 0
        prev = None
        start_date = None
        last_date = None
        for w, d in zip(is_win, dates):
            if w == prev:
                current += 1 if w else -1
                last_date = d
            else:
                if prev is not None:
                    streaks.append(current)
                    streak_dates.append((start_date, last_date))
                current = 1 if w else -1
                start_date = d
                last_date = d
            prev = w
        streaks.append(current)
        streak_dates.append((start_date, last_date))
        return streaks, streak_dates
