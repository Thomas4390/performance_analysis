"""
Module for analyzing strategy performance across VIX regimes.

Usage:
    python vix_analysis.py [--data-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    INTERMEDIATE_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    DEFAULT_VIX_REGIMES,
    VixRegimeConfig,
    PLOT_TEMPLATE,
)

# Import Numba-optimized functions for rolling metrics
try:
    from metrics_numba import (
        rolling_mean as numba_rolling_mean,
        rolling_std as numba_rolling_std,
        find_regime_boundaries,
    )
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VixRegimeStats:
    """Statistics for a single VIX regime."""
    regime: str
    days: int
    pct_time: float
    mean_vix: float
    ann_return: float
    ann_volatility: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    total_return: float
    bench_ann_return: Optional[float] = None
    alpha: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "regime": self.regime,
            "days": self.days,
            "pct_time": self.pct_time,
            "mean_vix": self.mean_vix,
            "ann_return": self.ann_return,
            "ann_volatility": self.ann_volatility,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
        }
        if self.bench_ann_return is not None:
            d["bench_ann_return"] = self.bench_ann_return
            d["alpha"] = self.alpha
        return d


# =============================================================================
# VIX REGIME ANALYZER
# =============================================================================

class VixRegimeAnalyzer:
    """
    Analyze strategy performance across different VIX regimes.

    Example:
        analyzer = VixRegimeAnalyzer(strategy_returns, vix, benchmark_returns)
        stats = analyzer.calculate_regime_stats()
        plots = analyzer.generate_plots()
    """

    def __init__(
        self,
        strategy_returns: pd.Series,
        vix: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        regimes: Optional[tuple[VixRegimeConfig, ...]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            strategy_returns: Daily strategy returns (decimal).
            vix: Daily VIX values.
            benchmark_returns: Optional benchmark returns for comparison.
            regimes: Custom VIX regime definitions.
        """
        self.regimes = regimes or DEFAULT_VIX_REGIMES

        # Align data
        common_idx = strategy_returns.index.intersection(vix.index)
        if benchmark_returns is not None:
            common_idx = common_idx.intersection(benchmark_returns.index)

        self.strategy_returns = strategy_returns.reindex(common_idx)
        self.vix = vix.reindex(common_idx)
        self.benchmark_returns = (
            benchmark_returns.reindex(common_idx) if benchmark_returns is not None else None
        )

        # Precompute regime classification using vectorized operations
        self._regime_series = self._classify_regimes_vectorized()

    def _classify_regimes_vectorized(self) -> pd.Series:
        """Classify VIX values into regimes using vectorized operations."""
        regime_series = pd.Series("Unknown", index=self.vix.index)
        for regime in self.regimes:
            mask = (self.vix >= regime.lower) & (self.vix < regime.upper)
            regime_series.loc[mask] = regime.name
        return regime_series

    def _classify_regime(self, vix_value: float) -> str:
        """Classify a single VIX value into a regime (for compatibility)."""
        for regime in self.regimes:
            if regime.lower <= vix_value < regime.upper:
                return regime.name
        return "Unknown"

    @property
    def regime_series(self) -> pd.Series:
        """Get regime classification for each date."""
        return self._regime_series

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series using numpy for speed."""
        if len(returns) == 0:
            return 0.0
        returns_arr = returns.values
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def calculate_regime_stats(self) -> pd.DataFrame:
        """
        Calculate performance statistics for each VIX regime.

        Returns:
            DataFrame with stats per regime.
        """
        stats_list = []

        for regime in self.regimes:
            mask = self._regime_series == regime.name
            if not mask.any():
                continue

            returns = self.strategy_returns[mask]
            n_days = len(returns)

            if n_days < 5:
                continue

            # Calculate statistics
            mean_return = returns.mean() * 252 * 100
            std_return = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            win_rate = (returns > 0).mean() * 100
            max_dd = self._calculate_max_drawdown(returns)
            total_return = (1 + returns).prod() - 1

            stat = VixRegimeStats(
                regime=regime.name,
                days=n_days,
                pct_time=n_days / len(self.strategy_returns) * 100,
                mean_vix=self.vix[mask].mean(),
                ann_return=mean_return,
                ann_volatility=std_return,
                sharpe=sharpe,
                win_rate=win_rate,
                max_drawdown=max_dd * 100,
                total_return=total_return * 100,
            )

            # Add benchmark comparison
            if self.benchmark_returns is not None:
                bench_returns = self.benchmark_returns[mask]
                bench_mean = bench_returns.mean() * 252 * 100
                stat.bench_ann_return = bench_mean
                stat.alpha = mean_return - bench_mean

            stats_list.append(stat.to_dict())

        return pd.DataFrame(stats_list)

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_regime_distribution(self) -> go.Figure:
        """Create pie chart of time spent in each regime."""
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
            title="Time Distribution Across VIX Regimes",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_returns_by_regime(self) -> go.Figure:
        """Create box plot of returns by VIX regime."""
        fig = go.Figure()

        regime_order = [r.name for r in self.regimes]
        colors = {r.name: r.color for r in self.regimes}

        for regime_name in regime_order:
            mask = self._regime_series == regime_name
            if not mask.any():
                continue

            returns = self.strategy_returns[mask] * 100

            fig.add_trace(go.Box(
                y=returns,
                name=regime_name,
                marker_color=colors.get(regime_name, "#999999"),
                boxmean=True,
            ))

        fig.update_layout(
            title="Daily Returns Distribution by VIX Regime",
            yaxis_title="Daily Return (%)",
            xaxis_title="VIX Regime",
            showlegend=False,
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_cumulative_with_regimes(self) -> go.Figure:
        """Plot cumulative returns with VIX regime overlay."""
        cumulative = (1 + self.strategy_returns).cumprod()

        fig = go.Figure()

        # Add cumulative return line
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="Strategy",
            line=dict(color="black", width=2),
        ))

        # Add regime backgrounds using optimized boundary detection
        colors = {r.name: r.color for r in self.regimes}
        boundaries = self._get_regime_boundaries()

        for start, end, regime in boundaries:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=colors.get(regime, "#999999"),
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        # Add legend for regimes
        for regime in self.regimes:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=regime.color),
                name=regime.name,
                showlegend=True,
            ))

        fig.update_layout(
            title="Cumulative Returns with VIX Regime Overlay",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_performance_heatmap(self) -> go.Figure:
        """Create heatmap of key metrics by VIX regime."""
        stats = self.calculate_regime_stats()

        if stats.empty:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data", showarrow=False)
            return fig

        metrics = ["ann_return", "ann_volatility", "sharpe", "win_rate", "max_drawdown"]
        metric_names = ["Ann. Return (%)", "Ann. Vol (%)", "Sharpe", "Win Rate (%)", "Max DD (%)"]

        heatmap_data = stats[metrics].values.T

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=stats["regime"].tolist(),
            y=metric_names,
            colorscale="RdYlGn",
            text=[[f"{v:.1f}" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 11},
        ))

        fig.update_layout(
            title="Performance Metrics by VIX Regime",
            xaxis_title="VIX Regime",
            yaxis_title="Metric",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_vix_vs_returns(self) -> go.Figure:
        """Create scatter plot of VIX vs daily returns."""
        fig = go.Figure()

        returns_pct = self.strategy_returns * 100

        fig.add_trace(go.Scatter(
            x=self.vix,
            y=returns_pct,
            mode="markers",
            name="Daily Returns",
            showlegend=False,  # Use colorbar instead of legend for scatter
            marker=dict(
                size=6,
                color=returns_pct,
                colorscale="RdYlGn",
                cmin=-3,
                cmax=3,
                showscale=True,
                colorbar=dict(title="Daily Return (%)"),
            ),
            text=self.vix.index.strftime("%Y-%m-%d"),
            hovertemplate="Date: %{text}<br>VIX: %{x:.1f}<br>Return: %{y:.2f}%<extra></extra>",
        ))

        # Trend line
        z = np.polyfit(self.vix, self.strategy_returns * 100, 1)
        p = np.poly1d(z)
        vix_range = np.linspace(self.vix.min(), self.vix.max(), 100)

        fig.add_trace(go.Scatter(
            x=vix_range,
            y=p(vix_range),
            mode="lines",
            name="Trend",
            line=dict(color="red", dash="dash"),
        ))

        # Regime boundaries
        for regime in self.regimes[1:]:
            if regime.lower < self.vix.max():
                fig.add_vline(
                    x=regime.lower,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"VIX={regime.lower}",
                )

        fig.update_layout(
            title="VIX Level vs Daily Returns (Color = Return Magnitude)",
            xaxis_title="VIX",
            yaxis_title="Daily Return (%)",
            template=PLOT_TEMPLATE,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
        )

        return fig

    def _get_regime_boundaries(self) -> list[tuple]:
        """
        Get consolidated regime boundaries for efficient plotting.

        Returns a list of (start_date, end_date, regime_name) tuples.
        """
        if HAS_NUMBA:
            # Use Numba-optimized boundary detection
            # Create regime index mapping
            regime_names = [r.name for r in self.regimes]
            regime_to_idx = {name: i for i, name in enumerate(regime_names)}

            # Convert regime series to numeric indices
            regime_indices = self._regime_series.map(
                lambda x: regime_to_idx.get(x, -1)
            ).values.astype(np.int64)

            starts, ends, regimes = find_regime_boundaries(regime_indices)

            # Convert back to dates and regime names
            dates = self._regime_series.index
            boundaries = []
            for i in range(len(starts)):
                start_date = dates[starts[i]]
                end_date = dates[min(ends[i], len(dates) - 1)]
                regime_idx = regimes[i]
                regime_name = regime_names[regime_idx] if 0 <= regime_idx < len(regime_names) else "Unknown"
                boundaries.append((start_date, end_date, regime_name))

            return boundaries
        else:
            # Pandas fallback
            regime_changes = self._regime_series != self._regime_series.shift(1)
            change_points = self._regime_series.index[regime_changes].tolist()
            if self.vix.index[-1] not in change_points:
                change_points.append(self.vix.index[-1])

            boundaries = []
            for i in range(len(change_points) - 1):
                start = change_points[i]
                end = change_points[i + 1]
                regime = self._regime_series.loc[start]
                boundaries.append((start, end, regime))

            return boundaries

    def plot_rolling_performance(self, window: int = 20) -> go.Figure:
        """Plot rolling performance metrics alongside VIX."""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                f"{window}-Day Rolling Return",
                f"{window}-Day Rolling Volatility",
                f"{window}-Day Rolling Sharpe",
                "VIX Level",
            ),
            vertical_spacing=0.06,
        )

        # Use Numba-optimized rolling calculations if available
        returns_arr = self.strategy_returns.values.astype(np.float64)

        if HAS_NUMBA:
            rolling_mean_arr = numba_rolling_mean(returns_arr, window)
            rolling_std_arr = numba_rolling_std(returns_arr, window)
            rolling_return = pd.Series(
                rolling_mean_arr * 252 * 100,
                index=self.strategy_returns.index
            )
            rolling_vol = pd.Series(
                rolling_std_arr * np.sqrt(252) * 100,
                index=self.strategy_returns.index
            )
            # Rolling Sharpe = (rolling_mean / rolling_std) * sqrt(252)
            with np.errstate(divide='ignore', invalid='ignore'):
                rolling_sharpe_arr = np.where(
                    rolling_std_arr > 0,
                    (rolling_mean_arr / rolling_std_arr) * np.sqrt(252),
                    0.0
                )
            rolling_sharpe = pd.Series(rolling_sharpe_arr, index=self.strategy_returns.index)
        else:
            # Pandas fallback
            rolling_mean = self.strategy_returns.rolling(window).mean()
            rolling_std = self.strategy_returns.rolling(window).std()
            rolling_return = rolling_mean * 252 * 100
            rolling_vol = rolling_std * np.sqrt(252) * 100
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
            rolling_sharpe = rolling_sharpe.fillna(0)

        # Rolling return
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return, name="Return", line=dict(color="blue", width=1.5), showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Volatility", line=dict(color="orange", width=1.5), showlegend=False),
            row=2, col=1
        )

        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Sharpe", line=dict(color="green", width=1.5), showlegend=False),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # VIX
        fig.add_trace(
            go.Scatter(x=self.vix.index, y=self.vix, name="VIX", line=dict(color="purple", width=1.5), showlegend=False),
            row=4, col=1
        )

        # Add regime backgrounds using shapes (much faster than vrects)
        colors = {r.name: r.color for r in self.regimes}
        boundaries = self._get_regime_boundaries()

        # Pre-build all shapes at once instead of calling add_vrect repeatedly
        shapes = []
        # Get y-axis references for each subplot
        yaxis_refs = ['y', 'y2', 'y3', 'y4']

        for start, end, regime in boundaries:
            color = colors.get(regime, "#999999")
            # Convert timestamps to strings for shapes
            x0_str = str(start)
            x1_str = str(end)

            for i, yref in enumerate(yaxis_refs):
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref=f"{yref} domain",
                    x0=x0_str,
                    x1=x1_str,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                ))

        # Add all shapes at once (much faster)
        fig.update_layout(shapes=shapes)

        # Add legend for VIX regimes (simple names only)
        for regime in self.regimes:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=regime.color, symbol="square"),
                name=regime.name,
                showlegend=True,
            ))

        fig.update_layout(
            height=900,
            title=dict(
                text=f"Rolling Performance Metrics ({window}-Day Window)",
                y=0.97,
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
            margin=dict(t=100),
            template=PLOT_TEMPLATE,
        )

        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        fig.update_yaxes(title_text="VIX", row=4, col=1)

        return fig

    def plot_monthly_heatmap(self) -> go.Figure:
        """Create monthly returns heatmap."""
        monthly_returns = self.strategy_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100

        df = pd.DataFrame({
            "return": monthly_returns,
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
        })

        pivot = df.pivot_table(values="return", index="year", columns="month")

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=month_names[:pivot.shape[1]],
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Return (%)"),
        ))

        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            yaxis=dict(autorange="reversed"),
            template=PLOT_TEMPLATE,
        )

        return fig

    def generate_all_plots(self) -> dict[str, go.Figure]:
        """Generate all VIX analysis plots."""
        return {
            "regime_distribution": self.plot_regime_distribution(),
            "returns_by_regime": self.plot_returns_by_regime(),
            "cumulative_with_regimes": self.plot_cumulative_with_regimes(),
            "performance_heatmap": self.plot_performance_heatmap(),
            "vix_vs_returns": self.plot_vix_vs_returns(),
            "rolling_performance": self.plot_rolling_performance(),
            "monthly_heatmap": self.plot_monthly_heatmap(),
        }

    def save_plots(
        self,
        output_dir: Union[str, Path] = PLOTS_DIR,
        format: str = "html",
    ) -> list[Path]:
        """Save all plots to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = self.generate_all_plots()
        saved = []

        for name, fig in plots.items():
            filepath = output_dir / f"vix_{name}.{format}"
            if format == "html":
                fig.write_html(filepath)
            else:
                fig.write_image(filepath, scale=2)
            saved.append(filepath)

        return saved

    def save_stats(
        self,
        output_dir: Union[str, Path] = INTERMEDIATE_DIR,
        filename: str = "vix_regime_stats.parquet",
    ) -> Path:
        """Save regime statistics to parquet."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = self.calculate_regime_stats()
        filepath = output_dir / filename
        stats.to_parquet(filepath, index=False)

        return filepath


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_data_for_vix_analysis(
    data_dir: Union[str, Path] = INTERMEDIATE_DIR,
) -> tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Load data needed for VIX analysis.

    Args:
        data_dir: Directory containing parquet files.

    Returns:
        Tuple of (strategy_returns, vix, benchmark_returns).
    """
    data_dir = Path(data_dir)

    # Load combined portfolio
    combined_path = data_dir / "combined_portfolio.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined portfolio not found: {combined_path}")

    combined = pd.read_parquet(combined_path)
    combined["date"] = pd.to_datetime(combined["date"])
    strategy_returns = combined.set_index("date")["daily_return_decimal"]

    # Load VIX
    vix_path = data_dir / "vix.parquet"
    if not vix_path.exists():
        raise FileNotFoundError(f"VIX data not found: {vix_path}")

    vix_df = pd.read_parquet(vix_path)
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    vix = vix_df.set_index("date")["vix"]

    # Load benchmark (optional)
    benchmark_returns = None
    benchmark_files = list(data_dir.glob("benchmark_*.parquet"))
    if benchmark_files:
        df = pd.read_parquet(benchmark_files[0])
        df["date"] = pd.to_datetime(df["date"])
        benchmark_returns = df.set_index("date")["daily_return_decimal"].dropna()

    return strategy_returns, vix, benchmark_returns


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyze strategy performance across VIX regimes."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(INTERMEDIATE_DIR),
        help="Directory containing data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  VIX REGIME ANALYSIS")
    print("=" * 60)

    try:
        strategy_returns, vix, benchmark_returns = load_data_for_vix_analysis(args.data_dir)
        print(f"\nLoaded data:")
        print(f"  Strategy returns: {len(strategy_returns)} days")
        print(f"  VIX data: {len(vix)} days")
        if benchmark_returns is not None:
            print(f"  Benchmark returns: {len(benchmark_returns)} days")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the previous pipeline steps first.")
        return

    # Create analyzer
    analyzer = VixRegimeAnalyzer(strategy_returns, vix, benchmark_returns)

    # Calculate stats
    stats = analyzer.calculate_regime_stats()
    print("\nVIX Regime Performance Summary:")
    print("-" * 80)
    print(stats.to_string(index=False))

    # Save stats
    stats_path = analyzer.save_stats(args.data_dir)
    print(f"\nSaved stats to: {stats_path}")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plots_dir = Path(args.output_dir) / "plots"
        saved = analyzer.save_plots(plots_dir)
        print(f"Saved {len(saved)} plots to: {plots_dir}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
