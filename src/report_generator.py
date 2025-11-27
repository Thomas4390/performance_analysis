"""
Module for generating comprehensive reports using quantstats and plotly.

Usage:
    python report_generator.py [--data-dir DIR] [--output-dir DIR]
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
    PLOT_TEMPLATE,
    PLOT_COLORS_NAMED,
)

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

try:
    from metrics_numba import calculate_all_metrics, warmup as numba_warmup
    HAS_NUMBA = True
    # Warmup Numba JIT on import
    numba_warmup()
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_dd_start: str
    max_dd_end: str
    win_rate: float
    profit_factor: float
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float
    best_year: float
    worst_year: float
    trading_days: int

    # Benchmark comparison (optional)
    benchmark_return: Optional[float] = None
    benchmark_cagr: Optional[float] = None
    benchmark_volatility: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation: Optional[float] = None
    information_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary with formatted values."""
        d = {
            "Total Return": f"{self.total_return:.2%}",
            "CAGR": f"{self.cagr:.2%}",
            "Volatility (Ann.)": f"{self.volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max DD Start": self.max_dd_start,
            "Max DD End": self.max_dd_end,
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Best Day": f"{self.best_day:.2%}",
            "Worst Day": f"{self.worst_day:.2%}",
            "Best Month": f"{self.best_month:.2%}",
            "Worst Month": f"{self.worst_month:.2%}",
            "Best Year": f"{self.best_year:.2%}",
            "Worst Year": f"{self.worst_year:.2%}",
            "Trading Days": self.trading_days,
        }

        if self.benchmark_return is not None:
            d.update({
                "Benchmark Return": f"{self.benchmark_return:.2%}",
                "Benchmark CAGR": f"{self.benchmark_cagr:.2%}",
                "Benchmark Volatility": f"{self.benchmark_volatility:.2%}",
                "Benchmark Sharpe": f"{self.benchmark_sharpe:.2f}",
                "Beta": f"{self.beta:.2f}",
                "Alpha (Ann.)": f"{self.alpha:.2%}",
                "Correlation": f"{self.correlation:.2f}",
                "Information Ratio": f"{self.information_ratio:.2f}",
            })

        return d

    def to_raw_dict(self) -> dict:
        """Convert to dictionary with raw values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Generate comprehensive backtest analysis reports.

    Example:
        generator = ReportGenerator(strategy_returns, benchmark_returns)
        metrics = generator.calculate_metrics()
        plots = generator.generate_plots()
    """

    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "Strategy",
        benchmark_name: str = "Benchmark",
    ):
        """
        Initialize the report generator.

        Args:
            strategy_returns: Daily returns (decimal) indexed by date.
            benchmark_returns: Optional benchmark returns.
            strategy_name: Name for the strategy in reports.
            benchmark_name: Name for the benchmark in reports.
        """
        self.strategy_name = strategy_name
        self.benchmark_name = benchmark_name

        # Align data if benchmark provided
        if benchmark_returns is not None:
            common_idx = strategy_returns.index.intersection(benchmark_returns.index)
            self.strategy_returns = strategy_returns.reindex(common_idx)
            self.benchmark_returns = benchmark_returns.reindex(common_idx)
        else:
            self.strategy_returns = strategy_returns
            self.benchmark_returns = None

    def calculate_metrics(self, use_numba: bool = True) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            use_numba: Use Numba-optimized calculations if available (default: True).

        Returns:
            PerformanceMetrics object with all calculated metrics.
        """
        returns = self.strategy_returns
        n_days = len(returns)

        # Use Numba if available and requested
        if use_numba and HAS_NUMBA:
            return self._calculate_metrics_numba()

        # Fallback to pandas/numpy implementation
        return self._calculate_metrics_pandas()

    def _calculate_metrics_numba(self) -> PerformanceMetrics:
        """Calculate metrics using Numba-optimized functions."""
        returns = self.strategy_returns
        returns_arr = returns.values.astype(np.float64)
        dates = returns.index.values

        # Prepare benchmark if available
        bench_arr = None
        if self.benchmark_returns is not None:
            bench_arr = self.benchmark_returns.values.astype(np.float64)

        # Calculate all metrics using Numba
        result = calculate_all_metrics(returns_arr, bench_arr, dates)

        # Best/worst periods (still use pandas for resampling)
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

        # Get drawdown dates
        dd_start_date = str(pd.Timestamp(result.get("dd_start_date", dates[0])).date())
        dd_end_date = str(pd.Timestamp(result.get("dd_end_date", dates[-1])).date())

        metrics = PerformanceMetrics(
            total_return=result["total_return"],
            cagr=result["cagr"],
            volatility=result["volatility"],
            sharpe_ratio=result["sharpe_ratio"],
            sortino_ratio=result["sortino_ratio"],
            calmar_ratio=result["calmar_ratio"],
            max_drawdown=result["max_drawdown"],
            max_dd_start=dd_start_date,
            max_dd_end=dd_end_date,
            win_rate=result["win_rate"],
            profit_factor=result["profit_factor"],
            best_day=result["best_day"],
            worst_day=result["worst_day"],
            best_month=float(monthly.max()),
            worst_month=float(monthly.min()),
            best_year=float(yearly.max()) if len(yearly) > 0 else 0.0,
            worst_year=float(yearly.min()) if len(yearly) > 0 else 0.0,
            trading_days=result["trading_days"],
        )

        # Add benchmark metrics if available
        if "benchmark_return" in result:
            metrics.benchmark_return = result["benchmark_return"]
            metrics.benchmark_cagr = result["benchmark_cagr"]
            metrics.benchmark_volatility = result["benchmark_volatility"]
            metrics.benchmark_sharpe = result["benchmark_sharpe"]
            metrics.beta = result["beta"]
            metrics.alpha = result["alpha"]
            metrics.correlation = result["correlation"]
            metrics.information_ratio = result["information_ratio"]

        return metrics

    def _calculate_metrics_pandas(self) -> PerformanceMetrics:
        """Calculate metrics using pandas/numpy (fallback implementation)."""
        returns = self.strategy_returns
        returns_arr = returns.values
        n_days = len(returns)

        # Basic metrics using numpy for speed
        total_return = float(np.prod(1 + returns_arr) - 1)
        cagr = (1 + total_return) ** (252 / n_days) - 1
        std_val = float(np.std(returns_arr, ddof=1))
        volatility = std_val * np.sqrt(252)
        mean_val = float(np.mean(returns_arr))
        sharpe = (mean_val / std_val * np.sqrt(252)) if std_val > 0 else 0

        # Drawdown using numpy
        cumulative_arr = np.cumprod(1 + returns_arr)
        running_max_arr = np.maximum.accumulate(cumulative_arr)
        drawdown_arr = (cumulative_arr - running_max_arr) / running_max_arr
        max_dd = float(np.min(drawdown_arr))
        dd_end_idx = int(np.argmin(drawdown_arr))
        dd_start_idx = int(np.argmax(cumulative_arr[:dd_end_idx + 1])) if dd_end_idx > 0 else 0
        dd_end = returns.index[dd_end_idx]
        dd_start = returns.index[dd_start_idx]

        # Calmar & Sortino
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252)
        sortino = (returns.mean() * 252 / downside_std) if downside_std > 0 else 0

        # Win rate & Profit factor
        winners = returns[returns > 0]
        losers = returns[returns < 0]
        non_zero = returns[returns != 0]
        win_rate = len(winners) / len(non_zero) if len(non_zero) > 0 else 0
        profit_factor = winners.sum() / abs(losers.sum()) if losers.sum() != 0 else float("inf")

        # Best/worst periods
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

        metrics = PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_dd_start=str(dd_start.date()),
            max_dd_end=str(dd_end.date()),
            win_rate=win_rate,
            profit_factor=min(profit_factor, 999.99),  # Cap for display
            best_day=returns.max(),
            worst_day=returns.min(),
            best_month=monthly.max(),
            worst_month=monthly.min(),
            best_year=yearly.max() if len(yearly) > 0 else 0,
            worst_year=yearly.min() if len(yearly) > 0 else 0,
            trading_days=n_days,
        )

        # Benchmark comparison
        if self.benchmark_returns is not None:
            bench = self.benchmark_returns
            bench_return = (1 + bench).prod() - 1
            bench_cagr = (1 + bench_return) ** (252 / len(bench)) - 1
            bench_vol = bench.std() * np.sqrt(252)
            bench_sharpe = (bench.mean() / bench.std() * np.sqrt(252)) if bench.std() > 0 else 0

            # Beta & Alpha
            cov = np.cov(returns, bench)[0, 1]
            var = bench.var()
            beta = cov / var if var > 0 else 0
            alpha = cagr - (beta * bench_cagr)

            # Correlation & Information ratio
            corr = returns.corr(bench)
            active = returns - bench
            tracking_error = active.std() * np.sqrt(252)
            info_ratio = (active.mean() * 252 / tracking_error) if tracking_error > 0 else 0

            metrics.benchmark_return = bench_return
            metrics.benchmark_cagr = bench_cagr
            metrics.benchmark_volatility = bench_vol
            metrics.benchmark_sharpe = bench_sharpe
            metrics.beta = beta
            metrics.alpha = alpha
            metrics.correlation = corr
            metrics.information_ratio = info_ratio

        return metrics

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_cumulative_returns(self) -> go.Figure:
        """Plot cumulative returns comparison."""
        fig = go.Figure()

        # Strategy
        strategy_cum = (1 + self.strategy_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=strategy_cum.index,
            y=strategy_cum.values,
            mode="lines",
            name=self.strategy_name,
            line=dict(color=PLOT_COLORS_NAMED["primary"], width=2),
        ))

        # Benchmark
        if self.benchmark_returns is not None:
            bench_cum = (1 + self.benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode="lines",
                name=self.benchmark_name,
                line=dict(color=PLOT_COLORS_NAMED["dark"], width=2, dash="dash"),
            ))

        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_drawdown(self) -> go.Figure:
        """Plot drawdown chart."""
        cumulative = (1 + self.strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=PLOT_COLORS_NAMED["danger"]),
            fillcolor="rgba(214, 39, 40, 0.3)",
        ))

        fig.update_layout(
            title="Underwater Plot (Drawdown)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_rolling_metrics(self, window: int = 252) -> go.Figure:
        """Plot rolling Sharpe ratio and volatility."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=(f"{window}-Day Rolling Sharpe", f"{window}-Day Rolling Volatility"),
            vertical_spacing=0.1,
        )

        # Rolling Sharpe
        rolling_mean = self.strategy_returns.rolling(window).mean()
        rolling_std = self.strategy_returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).dropna()

        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Sharpe", line=dict(color=PLOT_COLORS_NAMED["primary"])),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color="green", row=1, col=1)

        # Rolling Volatility
        rolling_vol = (rolling_std * np.sqrt(252) * 100).dropna()
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Volatility", line=dict(color=PLOT_COLORS_NAMED["secondary"])),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            title=f"Rolling Performance ({window}-Day Window)",
            showlegend=False,
            template=PLOT_TEMPLATE,
        )

        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        return fig

    def plot_returns_distribution(self) -> go.Figure:
        """Plot returns distribution histogram."""
        returns_pct = self.strategy_returns * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name=self.strategy_name,
            marker_color=PLOT_COLORS_NAMED["primary"],
            opacity=0.7,
        ))

        if self.benchmark_returns is not None:
            bench_pct = self.benchmark_returns * 100
            fig.add_trace(go.Histogram(
                x=bench_pct,
                nbinsx=50,
                name=self.benchmark_name,
                marker_color=PLOT_COLORS_NAMED["dark"],
                opacity=0.5,
            ))

        fig.add_vline(x=0, line_dash="dash", line_color="red")

        # Stats annotation
        mean_ret = returns_pct.mean()
        std_ret = returns_pct.std()
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Mean: {mean_ret:.3f}%<br>Std: {std_ret:.3f}%",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )

        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            barmode="overlay",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_monthly_returns_table(self) -> go.Figure:
        """Create monthly returns table."""
        monthly = self.strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

        df = pd.DataFrame({
            "return": monthly,
            "year": monthly.index.year,
            "month": monthly.index.month,
        })

        pivot = df.pivot_table(values="return", index="year", columns="month")

        # Add yearly totals
        yearly = self.strategy_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
        pivot["Year"] = yearly.values

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year"]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Year"] + month_names[:len(pivot.columns)],
                fill_color="paleturquoise",
                align="center",
            ),
            cells=dict(
                values=[pivot.index.tolist()] + [pivot[col].tolist() for col in pivot.columns],
                fill_color=[["white"] * len(pivot)] + [
                    ["lightgreen" if v > 0 else "lightsalmon" if v < 0 else "white"
                     for v in pivot[col].fillna(0)]
                    for col in pivot.columns
                ],
                format=[None] + [".1f"] * len(pivot.columns),
                align="center",
            )
        )])

        fig.update_layout(
            title="Monthly Returns (%)",
            height=max(300, 50 * len(pivot) + 100),
        )

        return fig

    def plot_yearly_comparison(self) -> go.Figure:
        """Plot yearly returns comparison."""
        yearly = self.strategy_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=yearly.index.year,
            y=yearly.values,
            name=self.strategy_name,
            marker_color=[PLOT_COLORS_NAMED["success"] if v > 0 else PLOT_COLORS_NAMED["danger"] for v in yearly.values],
        ))

        if self.benchmark_returns is not None:
            bench_yearly = self.benchmark_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
            fig.add_trace(go.Bar(
                x=bench_yearly.index.year,
                y=bench_yearly.values,
                name=self.benchmark_name,
                marker_color=PLOT_COLORS_NAMED["dark"],
                opacity=0.6,
            ))

        fig.update_layout(
            title="Yearly Returns Comparison",
            xaxis_title="Year",
            yaxis_title="Return (%)",
            barmode="group",
            template=PLOT_TEMPLATE,
        )

        return fig

    def generate_all_plots(self) -> dict[str, go.Figure]:
        """Generate all report plots."""
        return {
            "cumulative_returns": self.plot_cumulative_returns(),
            "drawdown": self.plot_drawdown(),
            "rolling_metrics": self.plot_rolling_metrics(),
            "returns_distribution": self.plot_returns_distribution(),
            "monthly_returns": self.plot_monthly_returns_table(),
            "yearly_comparison": self.plot_yearly_comparison(),
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
            filepath = output_dir / f"report_{name}.{format}"
            if format == "html":
                fig.write_html(filepath)
            else:
                fig.write_image(filepath, scale=2)
            saved.append(filepath)

        return saved

    def generate_quantstats_report(
        self,
        output_path: Union[str, Path] = REPORTS_DIR / "quantstats_report.html",
        title: str = "Backtest Analysis Report",
    ) -> Path:
        """Generate a full quantstats HTML report."""
        if not HAS_QUANTSTATS:
            raise ImportError("quantstats is required. Install with: pip install quantstats")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        returns = self.strategy_returns.copy()
        returns.index = pd.to_datetime(returns.index)

        benchmark = None
        if self.benchmark_returns is not None:
            benchmark = self.benchmark_returns.copy()
            benchmark.index = pd.to_datetime(benchmark.index)

        qs.reports.html(returns, benchmark=benchmark, output=str(output_path), title=title)

        return output_path

    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        metrics = self.calculate_metrics()

        print("\n" + "=" * 50)
        print(f"  {self.strategy_name} - Performance Summary")
        print("=" * 50)

        for key, value in metrics.to_dict().items():
            print(f"  {key:.<30} {value}")

        print("=" * 50 + "\n")

    def save_metrics(
        self,
        output_dir: Union[str, Path] = INTERMEDIATE_DIR,
        filename: str = "performance_metrics.parquet",
    ) -> Path:
        """Save metrics to parquet."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = self.calculate_metrics()
        df = pd.DataFrame([metrics.to_raw_dict()])
        filepath = output_dir / filename
        df.to_parquet(filepath, index=False)

        return filepath


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_returns_for_report(
    data_dir: Union[str, Path] = INTERMEDIATE_DIR,
) -> tuple[pd.Series, Optional[pd.Series]]:
    """
    Load strategy and benchmark returns for report generation.

    Args:
        data_dir: Directory containing parquet files.

    Returns:
        Tuple of (strategy_returns, benchmark_returns).
    """
    data_dir = Path(data_dir)

    # Load combined portfolio
    combined_path = data_dir / "combined_portfolio.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined portfolio not found: {combined_path}")

    combined = pd.read_parquet(combined_path)
    combined["date"] = pd.to_datetime(combined["date"])
    strategy_returns = combined.set_index("date")["daily_return_decimal"]

    # Load benchmark
    benchmark_returns = None
    benchmark_files = list(data_dir.glob("benchmark_*.parquet"))
    if benchmark_files:
        df = pd.read_parquet(benchmark_files[0])
        df["date"] = pd.to_datetime(df["date"])
        benchmark_returns = df.set_index("date")["daily_return_decimal"].dropna()

    return strategy_returns, benchmark_returns


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive backtest reports."
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
        "--strategy-name",
        type=str,
        default="Combined Portfolio",
        help="Strategy name for reports.",
    )
    parser.add_argument(
        "--no-quantstats",
        action="store_true",
        help="Skip quantstats HTML report.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  REPORT GENERATOR")
    print("=" * 60)

    try:
        strategy_returns, benchmark_returns = load_returns_for_report(args.data_dir)
        print(f"\nLoaded data:")
        print(f"  Strategy returns: {len(strategy_returns)} days")
        if benchmark_returns is not None:
            print(f"  Benchmark returns: {len(benchmark_returns)} days")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the previous pipeline steps first.")
        return

    # Create generator
    generator = ReportGenerator(
        strategy_returns,
        benchmark_returns,
        strategy_name=args.strategy_name,
    )

    # Print summary
    generator.print_summary()

    # Save metrics
    metrics_path = generator.save_metrics(args.data_dir)
    print(f"Saved metrics to: {metrics_path}")

    # Generate quantstats report
    if not args.no_quantstats:
        if HAS_QUANTSTATS:
            print("\nGenerating quantstats report...")
            try:
                report_path = generator.generate_quantstats_report(
                    Path(args.output_dir) / "quantstats_report.html"
                )
                print(f"Saved to: {report_path}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("\nQuantstats not installed. Skipping HTML report.")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plots_dir = Path(args.output_dir) / "plots"
        saved = generator.save_plots(plots_dir)
        print(f"Saved {len(saved)} plots to: {plots_dir}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
