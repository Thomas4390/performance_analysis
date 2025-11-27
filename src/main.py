"""
Main entry point for backtest analysis.

This script orchestrates the full analysis pipeline:
1. Load backtest data from various file formats
2. Combine backtests with specified weights
3. Download benchmark and VIX data
4. Generate quantstats report
5. Perform VIX regime analysis
6. Generate comprehensive visualizations

Usage:
    python main.py [--help]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from config import (
    RAW_DATA_DIR,
    INTERMEDIATE_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    AnalysisConfig,
    PlotFormat,
    PLOT_COLORS,
    PLOT_TEMPLATE,
)
from backtest_loader import BacktestLoader, load_backtests_from_parquet
from backtest_combiner import BacktestCombiner, CombinedPortfolio
from market_data import MarketDataDownloader
from vix_analysis import VixRegimeAnalyzer
from report_generator import ReportGenerator


# =============================================================================
# ANALYSIS RUNNER
# =============================================================================

class AnalysisRunner:
    """
    Orchestrates the complete backtest analysis pipeline.

    Example:
        config = AnalysisConfig(
            portfolio_weights={"bt1": 0.5, "bt2": 0.5},
            benchmark_ticker="SPY",
        )
        runner = AnalysisRunner(config)
        results = runner.run()
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the analysis runner.

        Args:
            config: Analysis configuration.
        """
        self.config = config
        self.results = {}

        # Initialize components
        self.loader = BacktestLoader(config.raw_data_dir, config.intermediate_dir)
        self.downloader = MarketDataDownloader(config.cache_dir)

    def _setup_directories(self) -> None:
        """Create output directories if needed."""
        Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.plots_dir).mkdir(parents=True, exist_ok=True)

    def _load_backtests(self) -> dict[str, pd.DataFrame]:
        """Load and validate backtests."""
        print("\n[1/6] Loading backtests...")

        available = self.loader.list_backtests()
        print(f"  Available backtests: {available}")

        backtests = {}
        for name in self.config.portfolio_weights.keys():
            if name not in available:
                raise ValueError(f"Backtest not found: {name}")

            df = self.loader.load(name)
            self.loader.save_parquet(df, name)
            backtests[name] = df
            print(f"  Loaded {name}: {len(df)} days")

        return backtests

    def _combine_backtests(
        self,
        backtests: dict[str, pd.DataFrame],
    ) -> CombinedPortfolio:
        """Combine backtests with weights."""
        print("\n[2/6] Combining backtests...")

        combiner = BacktestCombiner(backtests, self.config.portfolio_weights)
        print(combiner.get_weight_summary())

        portfolio = combiner.combine(self.config.initial_capital)
        combiner.save_combined(portfolio, self.config.intermediate_dir)

        print(f"  Combined portfolio: {len(portfolio.data)} days")
        print(f"  Date range: {portfolio.data['date'].min().date()} to {portfolio.data['date'].max().date()}")

        return portfolio

    def _download_market_data(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> tuple[pd.Series, pd.Series]:
        """Download benchmark and VIX data."""
        print("\n[3/6] Downloading market data...")

        # Download benchmark
        benchmark = self.downloader.download_benchmark(
            self.config.benchmark_ticker,
            start_date,
            end_date,
        )
        self.downloader.save(
            benchmark,
            self.config.intermediate_dir,
            f"benchmark_{self.config.benchmark_ticker.lower()}.parquet",
        )
        print(f"  Downloaded {self.config.benchmark_ticker}: {len(benchmark.data)} days")

        # Download VIX
        vix_data = self.downloader.download_vix(start_date, end_date)
        self.downloader.save(vix_data, self.config.intermediate_dir, "vix.parquet")
        print(f"  Downloaded VIX: {len(vix_data.data)} days")

        return benchmark.returns, vix_data.data.set_index("date")["vix"]

    def _generate_report(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> ReportGenerator:
        """Generate performance report."""
        print("\n[4/6] Generating performance report...")

        report_gen = ReportGenerator(
            strategy_returns,
            benchmark_returns,
            strategy_name=self.config.strategy_name,
            benchmark_name=self.config.benchmark_name,
        )

        # Print metrics summary
        metrics = report_gen.calculate_metrics()
        print(f"\n  Performance Summary:")
        print(f"  {'-' * 40}")
        print(f"  Total Return:  {metrics.total_return:.2%}")
        print(f"  CAGR:          {metrics.cagr:.2%}")
        print(f"  Volatility:    {metrics.volatility:.2%}")
        print(f"  Sharpe Ratio:  {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:  {metrics.max_drawdown:.2%}")

        # Generate quantstats report
        if self.config.generate_quantstats_report:
            report_path = Path(self.config.output_dir) / "quantstats_report.html"
            report_gen.generate_quantstats_report(
                output_path=report_path,
                title=self.config.report_title,
            )
            print(f"\n  Quantstats report saved to: {report_path}")

        return report_gen

    def _run_vix_analysis(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        vix: pd.Series,
    ) -> Optional[VixRegimeAnalyzer]:
        """Perform VIX regime analysis."""
        if not self.config.generate_vix_analysis:
            print("\n[5/6] Skipping VIX regime analysis...")
            return None

        print("\n[5/6] Performing VIX regime analysis...")

        vix_analyzer = VixRegimeAnalyzer(
            strategy_returns,
            vix,
            benchmark_returns,
            regimes=self.config.vix_regimes,
        )

        # Calculate and display regime stats
        regime_stats = vix_analyzer.calculate_regime_stats()

        print("\n  VIX Regime Performance Summary:")
        print("  " + "-" * 80)
        for _, row in regime_stats.iterrows():
            print(f"  {row['regime']:15} | Days: {row['days']:5.0f} | "
                  f"Return: {row['total_return']:8.2f}% | Sharpe: {row['sharpe']:6.2f} | "
                  f"Win Rate: {row['win_rate']:6.2f}%")

        return vix_analyzer

    def _analyze_individual_backtests(
        self,
        backtests: dict[str, pd.DataFrame],
        benchmark_returns: pd.Series,
    ) -> dict:
        """Analyze individual backtests."""
        if not self.config.generate_individual_analysis or len(backtests) <= 1:
            print("\n[6/6] Skipping individual backtest analysis...")
            return {}

        print("\n[6/6] Analyzing individual backtests...")

        individual_results = {}
        for name, bt_df in backtests.items():
            bt_returns = bt_df.set_index("date")["daily_return_decimal"]

            # Align with benchmark
            common_idx = bt_returns.index.intersection(benchmark_returns.index)
            bt_returns_aligned = bt_returns.reindex(common_idx)
            bench_aligned = benchmark_returns.reindex(common_idx)

            bt_report = ReportGenerator(
                bt_returns_aligned,
                bench_aligned,
                strategy_name=name,
                benchmark_name=self.config.benchmark_name,
            )

            metrics = bt_report.calculate_metrics()
            individual_results[name] = {
                "metrics": metrics,
                "returns": bt_returns_aligned,
            }

            print(f"\n  {name}:")
            print(f"    Total Return: {metrics.total_return:.2%}")
            print(f"    CAGR: {metrics.cagr:.2%}")
            print(f"    Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"    Max DD: {metrics.max_drawdown:.2%}")

        return individual_results

    def _save_plots(
        self,
        plots: dict[str, go.Figure],
        prefix: str,
    ) -> None:
        """Save plots to files."""
        if not self.config.save_plots:
            return

        plots_dir = Path(self.config.plots_dir)
        for name, fig in plots.items():
            ext = self.config.plot_format.value
            filepath = plots_dir / f"{prefix}_{name}.{ext}"

            if self.config.plot_format == PlotFormat.HTML:
                fig.write_html(filepath)
            else:
                fig.write_image(filepath, scale=2)

    def _create_comparison_plot(
        self,
        backtests: dict[str, pd.DataFrame],
        portfolio: CombinedPortfolio,
    ) -> go.Figure:
        """Create cumulative returns comparison plot."""
        fig = go.Figure()

        # Plot individual backtests
        for i, (name, df) in enumerate(backtests.items()):
            df_indexed = df.set_index("date")
            cumulative = (1 + df_indexed["daily_return_decimal"]).cumprod()

            weight = portfolio.weights.get(name, 0)
            label = f"{name} ({weight:.0%})"

            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode="lines",
                name=label,
                line=dict(color=PLOT_COLORS[i % len(PLOT_COLORS)], width=1.5),
                opacity=0.7,
            ))

        # Plot combined portfolio
        combined_cum = portfolio.equity_curve / portfolio.initial_capital

        fig.add_trace(go.Scatter(
            x=combined_cum.index,
            y=combined_cum.values,
            mode="lines",
            name=self.config.strategy_name,
            line=dict(color="black", width=3),
        ))

        fig.update_layout(
            title="Cumulative Returns: Individual Backtests vs Combined Portfolio",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            hovermode="x unified",
            template=PLOT_TEMPLATE,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig

    def run(self) -> dict:
        """
        Run the complete analysis pipeline.

        Returns:
            Dictionary containing all analysis results.
        """
        print("=" * 60)
        print("  BACKTEST ANALYSIS")
        print("=" * 60)

        self._setup_directories()

        # 1. Load backtests
        backtests = self._load_backtests()
        self.results["backtests"] = backtests

        # 2. Combine backtests
        portfolio = self._combine_backtests(backtests)
        self.results["portfolio"] = portfolio

        # 3. Download market data
        start_date = portfolio.data["date"].min()
        end_date = portfolio.data["date"].max()

        benchmark_returns, vix = self._download_market_data(start_date, end_date)
        self.results["benchmark_returns"] = benchmark_returns
        self.results["vix"] = vix

        # Align all data
        strategy_returns = portfolio.returns
        strategy_returns, benchmark_returns, vix = self.downloader.align_with_strategy(
            strategy_returns,
            self.config.benchmark_ticker,
        )
        print(f"  Aligned data: {len(strategy_returns)} common trading days")

        # 4. Generate performance report
        report_gen = self._generate_report(strategy_returns, benchmark_returns)
        self.results["metrics"] = report_gen.calculate_metrics()

        # Save report plots
        report_plots = report_gen.generate_all_plots()
        self.results["report_plots"] = report_plots
        self._save_plots(report_plots, "report")

        # 5. VIX analysis
        vix_analyzer = self._run_vix_analysis(strategy_returns, benchmark_returns, vix)
        if vix_analyzer:
            self.results["regime_stats"] = vix_analyzer.calculate_regime_stats()
            vix_plots = vix_analyzer.generate_all_plots()
            self.results["vix_plots"] = vix_plots
            self._save_plots(vix_plots, "vix")

        # 6. Individual backtest analysis
        individual_results = self._analyze_individual_backtests(backtests, benchmark_returns)
        self.results["individual_backtests"] = individual_results

        # Create and save comparison plot
        if len(backtests) > 1:
            comparison_fig = self._create_comparison_plot(backtests, portfolio)
            self.results["comparison_plot"] = comparison_fig
            self._save_plots({"comparison": comparison_fig}, "backtest")

        print("\n" + "=" * 60)
        print(f"  Analysis complete! Results saved to: {self.config.output_dir}")
        print("=" * 60)

        return self.results


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Run complete backtest analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py
    python main.py --benchmark QQQ
    python main.py --weights '{"bt1": 0.7, "bt2": 0.3}'
        """,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Directory containing raw backtest files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Benchmark ticker symbol.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help='Portfolio weights as JSON (e.g., \'{"bt1": 0.5, "bt2": 0.5}\').',
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
        help="Initial portfolio capital.",
    )
    parser.add_argument(
        "--no-quantstats",
        action="store_true",
        help="Skip quantstats report generation.",
    )
    parser.add_argument(
        "--no-vix",
        action="store_true",
        help="Skip VIX regime analysis.",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        choices=["html", "png"],
        default="html",
        help="Output format for plots.",
    )
    args = parser.parse_args()

    # Parse weights
    import json
    if args.weights:
        weights = json.loads(args.weights)
    else:
        # Default: use all available backtests with equal weights
        loader = BacktestLoader(args.raw_dir)
        available = loader.list_backtests()
        if not available:
            print(f"No backtest files found in {args.raw_dir}")
            return
        weights = {name: 1.0 / len(available) for name in available}

    # Create configuration
    config = AnalysisConfig(
        raw_data_dir=args.raw_dir,
        intermediate_dir=str(INTERMEDIATE_DIR),
        output_dir=args.output_dir,
        cache_dir=str(CACHE_DIR),
        plots_dir=str(PLOTS_DIR),
        portfolio_weights=weights,
        initial_capital=args.capital,
        benchmark_ticker=args.benchmark,
        benchmark_name=args.benchmark,
        strategy_name="Combined Portfolio",
        report_title="Backtest Analysis Report",
        plot_format=PlotFormat(args.plot_format),
        generate_quantstats_report=not args.no_quantstats,
        generate_vix_analysis=not args.no_vix,
        generate_individual_analysis=True,
        save_plots=True,
    )

    # Run analysis
    runner = AnalysisRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
