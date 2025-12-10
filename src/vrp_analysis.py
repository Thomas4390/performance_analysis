"""
Module for Volatility Risk Premium (VRP) analysis.

The VRP measures the difference between implied volatility (VIX) and realized volatility (RV).
A positive VRP indicates that options are "expensive" relative to actual market volatility,
which historically favors options sellers.

Key metrics:
- VRP = IV (VIX) - RV (Realized Volatility)
- Positive VRP: IV > RV, tailwind for options sellers
- Negative VRP: IV < RV, tailwind for options buyers

Usage:
    analyzer = VRPAnalyzer(spy_prices, vix_data)
    vrp_df = analyzer.calculate_vrp()
    fig = analyzer.plot_vrp_timeseries()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOT_TEMPLATE, PLOT_COLORS

# Import Numba-optimized functions if available
try:
    from metrics_numba import rolling_std as numba_rolling_std
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VRPRegimeConfig:
    """Configuration for a VRP regime."""
    name: str
    lower: float
    upper: float
    color: str
    description: str


DEFAULT_VRP_REGIMES = (
    VRPRegimeConfig("Very Negative (<-5)", float("-inf"), -5, "#d62728", "IV much lower than RV - favor long volatility"),
    VRPRegimeConfig("Negative (-5 to 0)", -5, 0, "#ff7f0e", "IV slightly lower than RV - caution for vol sellers"),
    VRPRegimeConfig("Low Positive (0 to 3)", 0, 3, "#bcbd22", "Mild premium - small edge for vol sellers"),
    VRPRegimeConfig("Normal (3 to 6)", 3, 6, "#2ca02c", "Typical premium - favorable for vol sellers"),
    VRPRegimeConfig("High (6 to 10)", 6, 10, "#1f77b4", "Rich premium - strong edge for vol sellers"),
    VRPRegimeConfig("Very High (>10)", 10, float("inf"), "#9467bd", "Extreme premium - very favorable but often during stress"),
)


@dataclass
class VRPStats:
    """Statistics for VRP analysis."""
    mean_vrp: float
    median_vrp: float
    std_vrp: float
    pct_positive: float
    pct_negative: float
    current_vrp: float
    current_percentile: float
    min_vrp: float
    max_vrp: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean_vrp": self.mean_vrp,
            "median_vrp": self.median_vrp,
            "std_vrp": self.std_vrp,
            "pct_positive": self.pct_positive,
            "pct_negative": self.pct_negative,
            "current_vrp": self.current_vrp,
            "current_percentile": self.current_percentile,
            "min_vrp": self.min_vrp,
            "max_vrp": self.max_vrp,
        }


# =============================================================================
# VRP ANALYZER
# =============================================================================

class VRPAnalyzer:
    """
    Analyze Volatility Risk Premium (VRP).

    The VRP is calculated as: VRP = VIX - Realized Volatility (annualized)

    Example:
        analyzer = VRPAnalyzer(spy_returns, vix_data)
        vrp_df = analyzer.calculate_vrp()
        stats = analyzer.calculate_stats()
        fig = analyzer.plot_vrp_timeseries()
    """

    def __init__(
        self,
        spy_returns: pd.Series,
        vix: pd.Series,
        strategy_returns: Optional[pd.Series] = None,
        rv_window: int = 21,
        regimes: Optional[tuple[VRPRegimeConfig, ...]] = None,
    ):
        """
        Initialize the VRP analyzer.

        Args:
            spy_returns: Daily SPY returns (decimal format).
            vix: Daily VIX values.
            strategy_returns: Optional strategy returns for performance comparison.
            rv_window: Window for realized volatility calculation (default: 21 trading days = 1 month).
            regimes: Custom VRP regime definitions.
        """
        self.rv_window = rv_window
        self.regimes = regimes or DEFAULT_VRP_REGIMES

        # Align data
        common_idx = spy_returns.index.intersection(vix.index)
        if strategy_returns is not None:
            common_idx = common_idx.intersection(strategy_returns.index)

        self.spy_returns = spy_returns.reindex(common_idx)
        self.vix = vix.reindex(common_idx)
        self.strategy_returns = (
            strategy_returns.reindex(common_idx) if strategy_returns is not None else None
        )

        # Calculate VRP
        self._vrp_df = self._calculate_vrp()
        self._regime_series = self._classify_regimes()

    def _calculate_realized_volatility(self) -> pd.Series:
        """
        Calculate annualized realized volatility using a rolling window.

        Returns:
            Series of annualized realized volatility (in percentage points).
        """
        returns_arr = self.spy_returns.values.astype(np.float64)

        if HAS_NUMBA:
            rolling_std = numba_rolling_std(returns_arr, self.rv_window)
            rv = pd.Series(rolling_std * np.sqrt(252) * 100, index=self.spy_returns.index)
        else:
            rv = self.spy_returns.rolling(self.rv_window).std() * np.sqrt(252) * 100

        return rv

    def _calculate_vrp(self) -> pd.DataFrame:
        """
        Calculate VRP time series.

        Returns:
            DataFrame with VIX, RV, and VRP columns.
        """
        rv = self._calculate_realized_volatility()

        df = pd.DataFrame({
            "vix": self.vix,
            "realized_vol": rv,
            "vrp": self.vix - rv,
        }, index=self.spy_returns.index)

        # Add strategy returns if available
        if self.strategy_returns is not None:
            df["strategy_return"] = self.strategy_returns
            df["strategy_cumulative"] = (1 + self.strategy_returns).cumprod()

        return df.dropna()

    def _classify_regimes(self) -> pd.Series:
        """Classify VRP values into regimes."""
        vrp = self._vrp_df["vrp"]
        regime_series = pd.Series("Unknown", index=vrp.index)

        for regime in self.regimes:
            mask = (vrp >= regime.lower) & (vrp < regime.upper)
            regime_series.loc[mask] = regime.name

        return regime_series

    @property
    def vrp_data(self) -> pd.DataFrame:
        """Get the VRP DataFrame."""
        return self._vrp_df.copy()

    @property
    def regime_series(self) -> pd.Series:
        """Get the regime classification series."""
        return self._regime_series.copy()

    def calculate_stats(self) -> VRPStats:
        """
        Calculate summary statistics for VRP.

        Returns:
            VRPStats object with summary statistics.
        """
        vrp = self._vrp_df["vrp"]

        return VRPStats(
            mean_vrp=vrp.mean(),
            median_vrp=vrp.median(),
            std_vrp=vrp.std(),
            pct_positive=(vrp > 0).mean() * 100,
            pct_negative=(vrp < 0).mean() * 100,
            current_vrp=vrp.iloc[-1],
            current_percentile=(vrp < vrp.iloc[-1]).mean() * 100,
            min_vrp=vrp.min(),
            max_vrp=vrp.max(),
        )

    def calculate_regime_stats(self) -> pd.DataFrame:
        """
        Calculate performance statistics by VRP regime.

        Returns:
            DataFrame with stats per regime.
        """
        stats_list = []

        for regime in self.regimes:
            mask = self._regime_series == regime.name
            if not mask.any():
                continue

            vrp_vals = self._vrp_df.loc[mask, "vrp"]
            n_days = len(vrp_vals)

            if n_days < 5:
                continue

            stat = {
                "regime": regime.name,
                "days": n_days,
                "pct_time": n_days / len(self._vrp_df) * 100,
                "mean_vrp": vrp_vals.mean(),
                "mean_vix": self._vrp_df.loc[mask, "vix"].mean(),
                "mean_rv": self._vrp_df.loc[mask, "realized_vol"].mean(),
            }

            # Add strategy performance if available
            if self.strategy_returns is not None and "strategy_return" in self._vrp_df.columns:
                strat_returns = self._vrp_df.loc[mask, "strategy_return"]
                stat["ann_return"] = strat_returns.mean() * 252 * 100
                stat["ann_vol"] = strat_returns.std() * np.sqrt(252) * 100
                stat["sharpe"] = (strat_returns.mean() / strat_returns.std() * np.sqrt(252)) if strat_returns.std() > 0 else 0
                stat["win_rate"] = (strat_returns > 0).mean() * 100
                stat["total_return"] = ((1 + strat_returns).prod() - 1) * 100

            stats_list.append(stat)

        return pd.DataFrame(stats_list)

    def get_vrp_regime_boundaries(self) -> list[tuple]:
        """
        Get consolidated VRP regime boundaries for plotting.

        Returns:
            List of (start_date, end_date, regime_name) tuples.
        """
        regime_changes = self._regime_series != self._regime_series.shift(1)
        change_points = self._regime_series.index[regime_changes].tolist()
        if self._regime_series.index[-1] not in change_points:
            change_points.append(self._regime_series.index[-1])

        boundaries = []
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            regime = self._regime_series.loc[start]
            boundaries.append((start, end, regime))

        return boundaries

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_vrp_timeseries(self) -> go.Figure:
        """
        Plot VRP time series with VIX and Realized Volatility.

        Returns:
            Plotly figure.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("VIX vs Realized Volatility", "Volatility Risk Premium (VRP)"),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
        )

        df = self._vrp_df

        # Top panel: VIX vs RV
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["vix"],
                name="VIX (Implied Vol)",
                line=dict(color="#d62728", width=1.5),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["realized_vol"],
                name=f"Realized Vol ({self.rv_window}d)",
                line=dict(color="#1f77b4", width=1.5),
            ),
            row=1, col=1,
        )

        # Bottom panel: VRP with color based on sign
        vrp_pos = df["vrp"].where(df["vrp"] >= 0)
        vrp_neg = df["vrp"].where(df["vrp"] < 0)

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vrp_pos,
                name="VRP (Positive)",
                fill="tozeroy",
                fillcolor="rgba(44, 160, 44, 0.3)",
                line=dict(color="#2ca02c", width=1),
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vrp_neg,
                name="VRP (Negative)",
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.3)",
                line=dict(color="#d62728", width=1),
            ),
            row=2, col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Add mean VRP line
        mean_vrp = df["vrp"].mean()
        fig.add_hline(
            y=mean_vrp,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"Mean: {mean_vrp:.1f}",
            row=2, col=1,
        )

        fig.update_layout(
            height=600,
            title=dict(
                text="Volatility Risk Premium Analysis",
                y=0.97,
            ),
            template=PLOT_TEMPLATE,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )

        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="VRP (%)", row=2, col=1)

        return fig

    def plot_vrp_distribution(self) -> go.Figure:
        """
        Plot VRP distribution histogram.

        Returns:
            Plotly figure.
        """
        vrp = self._vrp_df["vrp"]

        fig = go.Figure()

        # Histogram with color split
        fig.add_trace(go.Histogram(
            x=vrp[vrp >= 0],
            name="Positive VRP",
            marker_color="rgba(44, 160, 44, 0.7)",
            nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=vrp[vrp < 0],
            name="Negative VRP",
            marker_color="rgba(214, 39, 40, 0.7)",
            nbinsx=50,
        ))

        # Add vertical lines for mean and current
        fig.add_vline(x=vrp.mean(), line_dash="dash", line_color="purple",
                      annotation_text=f"Mean: {vrp.mean():.1f}")
        fig.add_vline(x=vrp.iloc[-1], line_dash="dot", line_color="black",
                      annotation_text=f"Current: {vrp.iloc[-1]:.1f}")
        fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=2)

        fig.update_layout(
            title="VRP Distribution",
            xaxis_title="VRP (%)",
            yaxis_title="Frequency",
            barmode="overlay",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_vrp_percentile(self) -> go.Figure:
        """
        Plot current VRP percentile gauge.

        Returns:
            Plotly figure.
        """
        stats = self.calculate_stats()

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stats.current_vrp,
            delta={"reference": stats.mean_vrp, "relative": False},
            title={"text": f"Current VRP<br><span style='font-size:0.8em;color:gray'>Percentile: {stats.current_percentile:.0f}%</span>"},
            gauge={
                "axis": {"range": [stats.min_vrp - 2, stats.max_vrp + 2]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [stats.min_vrp - 2, 0], "color": "rgba(214, 39, 40, 0.3)"},
                    {"range": [0, stats.max_vrp + 2], "color": "rgba(44, 160, 44, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "purple", "width": 4},
                    "thickness": 0.75,
                    "value": stats.mean_vrp,
                },
            },
        ))

        fig.update_layout(
            height=300,
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_vrp_vs_strategy_performance(self) -> go.Figure:
        """
        Plot VRP with strategy performance overlay.

        Returns:
            Plotly figure.
        """
        if self.strategy_returns is None or "strategy_cumulative" not in self._vrp_df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Strategy returns not available", showarrow=False)
            return fig

        df = self._vrp_df

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Strategy Cumulative Returns", "Volatility Risk Premium"),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
        )

        # Top panel: Strategy performance with VRP regime backgrounds
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["strategy_cumulative"],
                name="Strategy",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1, col=1,
        )

        # Add VRP regime backgrounds
        colors = {r.name: r.color for r in self.regimes}
        boundaries = self.get_vrp_regime_boundaries()

        shapes = []
        for start, end, regime in boundaries:
            color = colors.get(regime, "#999999")
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="y domain",
                x0=str(start),
                x1=str(end),
                y0=0,
                y1=1,
                fillcolor=color,
                opacity=0.15,
                layer="below",
                line_width=0,
            ))

        # Bottom panel: VRP
        vrp_pos = df["vrp"].where(df["vrp"] >= 0)
        vrp_neg = df["vrp"].where(df["vrp"] < 0)

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vrp_pos,
                name="VRP+",
                fill="tozeroy",
                fillcolor="rgba(44, 160, 44, 0.4)",
                line=dict(color="#2ca02c", width=1),
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vrp_neg,
                name="VRP-",
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.4)",
                line=dict(color="#d62728", width=1),
                showlegend=True,
            ),
            row=2, col=1,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            shapes=shapes,
            height=700,
            title=dict(
                text="Strategy Performance vs VRP Regimes",
                y=0.97,
            ),
            template=PLOT_TEMPLATE,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )

        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="VRP (%)", row=2, col=1)

        return fig

    def plot_regime_performance_bars(self) -> go.Figure:
        """
        Plot strategy performance by VRP regime as bar chart.

        Returns:
            Plotly figure.
        """
        stats = self.calculate_regime_stats()

        if stats.empty or "ann_return" not in stats.columns:
            fig = go.Figure()
            fig.add_annotation(text="Strategy returns not available", showarrow=False)
            return fig

        colors = {r.name: r.color for r in self.regimes}

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=stats["regime"],
            y=stats["ann_return"],
            marker_color=[colors.get(r, "#999999") for r in stats["regime"]],
            text=[f"{v:.1f}%" for v in stats["ann_return"]],
            textposition="outside",
            name="Ann. Return",
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="Strategy Annualized Return by VRP Regime",
            xaxis_title="VRP Regime",
            yaxis_title="Annualized Return (%)",
            template=PLOT_TEMPLATE,
            showlegend=False,
        )

        return fig

    def plot_regime_distribution_pie(self) -> go.Figure:
        """
        Plot time distribution across VRP regimes.

        Returns:
            Plotly figure.
        """
        counts = self._regime_series.value_counts()
        regime_order = [r.name for r in self.regimes]
        counts = counts.reindex([r for r in regime_order if r in counts.index])

        colors = {r.name: r.color for r in self.regimes}

        fig = go.Figure(data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                marker=dict(colors=[colors.get(r, "#999999") for r in counts.index]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Days: %{value}<br>%{percent}<extra></extra>",
            )
        ])

        fig.update_layout(
            title="Time Distribution Across VRP Regimes",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_rolling_vrp(self, window: int = 60) -> go.Figure:
        """
        Plot rolling VRP statistics.

        Args:
            window: Rolling window size.

        Returns:
            Plotly figure.
        """
        df = self._vrp_df

        rolling_mean = df["vrp"].rolling(window).mean()
        rolling_std = df["vrp"].rolling(window).std()

        fig = go.Figure()

        # VRP line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["vrp"],
            name="VRP",
            line=dict(color="#1f77b4", width=1),
            opacity=0.5,
        ))

        # Rolling mean
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean,
            name=f"{window}d Mean",
            line=dict(color="#d62728", width=2),
        ))

        # Rolling bands
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean + 2 * rolling_std,
            name="+2 Std",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean - 2 * rolling_std,
            name="-2 Std",
            line=dict(color="gray", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.2)",
            showlegend=False,
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=f"Rolling VRP ({window}-Day Window)",
            xaxis_title="Date",
            yaxis_title="VRP (%)",
            template=PLOT_TEMPLATE,
            hovermode="x unified",
        )

        return fig

    def plot_scatter_vrp_vs_returns(self) -> go.Figure:
        """
        Plot scatter of VRP vs next-day strategy returns.

        Returns:
            Plotly figure.
        """
        if self.strategy_returns is None:
            fig = go.Figure()
            fig.add_annotation(text="Strategy returns not available", showarrow=False)
            return fig

        df = self._vrp_df.copy()
        df["next_return"] = df["strategy_return"].shift(-1) * 100  # Convert to percentage
        df = df.dropna()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["vrp"],
            y=df["next_return"],
            mode="markers",
            marker=dict(
                size=6,
                color=df["next_return"],
                colorscale="RdYlGn",
                cmin=-3,
                cmax=3,
                showscale=True,
                colorbar=dict(title="Return (%)"),
            ),
            text=df.index.strftime("%Y-%m-%d"),
            hovertemplate="Date: %{text}<br>VRP: %{x:.1f}%<br>Next Return: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ))

        # Add trend line
        z = np.polyfit(df["vrp"], df["next_return"], 1)
        p = np.poly1d(z)
        vrp_range = np.linspace(df["vrp"].min(), df["vrp"].max(), 100)

        fig.add_trace(go.Scatter(
            x=vrp_range,
            y=p(vrp_range),
            mode="lines",
            name="Trend",
            line=dict(color="red", dash="dash"),
        ))

        # Add zero lines
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        fig.update_layout(
            title="VRP vs Next-Day Strategy Return",
            xaxis_title="VRP (%)",
            yaxis_title="Next-Day Return (%)",
            template=PLOT_TEMPLATE,
        )

        return fig

    def generate_all_plots(self) -> dict[str, go.Figure]:
        """Generate all VRP analysis plots."""
        plots = {
            "vrp_timeseries": self.plot_vrp_timeseries(),
            "vrp_distribution": self.plot_vrp_distribution(),
            "vrp_percentile": self.plot_vrp_percentile(),
            "rolling_vrp": self.plot_rolling_vrp(),
            "regime_distribution": self.plot_regime_distribution_pie(),
        }

        if self.strategy_returns is not None:
            plots["vrp_vs_performance"] = self.plot_vrp_vs_strategy_performance()
            plots["regime_performance"] = self.plot_regime_performance_bars()
            plots["scatter_vrp_returns"] = self.plot_scatter_vrp_vs_returns()

        return plots


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_vrp_series(
    spy_returns: pd.Series,
    vix: pd.Series,
    rv_window: int = 21,
) -> pd.DataFrame:
    """
    Calculate VRP time series without creating full analyzer.

    Args:
        spy_returns: Daily SPY returns (decimal).
        vix: Daily VIX values.
        rv_window: Window for realized volatility calculation.

    Returns:
        DataFrame with vix, realized_vol, and vrp columns.
    """
    analyzer = VRPAnalyzer(spy_returns, vix, rv_window=rv_window)
    return analyzer.vrp_data
