"""
Module for combining multiple backtests with weighted allocations.

Usage:
    python backtest_combiner.py [--data-dir DIR] [--weights JSON] [--initial-capital N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from config import BACKTESTS_DIR

# Import Numba-optimized functions if available
try:
    from metrics_numba import calculate_core_metrics
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trading_days: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trading_days": self.trading_days,
        }


@dataclass
class CombinedPortfolio:
    """Container for combined portfolio data."""
    data: pd.DataFrame
    weights: dict[str, float]
    initial_capital: float
    individual_returns: pd.DataFrame = field(default=None)

    @property
    def returns(self) -> pd.Series:
        """Get daily returns as a Series indexed by date."""
        return self.data.set_index("date")["daily_return_decimal"]

    @property
    def equity_curve(self) -> pd.Series:
        """Get equity curve as a Series indexed by date."""
        return self.data.set_index("date")["equity"]

    @property
    def final_equity(self) -> float:
        """Get final portfolio value."""
        return self.data["equity"].iloc[-1]

    @property
    def total_return(self) -> float:
        """Get total return as decimal."""
        return self.final_equity / self.initial_capital - 1

    def calculate_metrics(self) -> PortfolioMetrics:
        """Calculate portfolio performance metrics using Numba if available."""
        returns = self.returns
        n_days = len(returns)

        if HAS_NUMBA:
            # Use Numba-optimized calculations
            returns_arr = returns.values.astype(np.float64)
            (
                total_return,
                cagr,
                volatility,
                sharpe,
                _sortino,
                max_dd,
                _win_rate,
                _profit_factor,
                _dd_peak_idx,
                _dd_trough_idx,
            ) = calculate_core_metrics(returns_arr, 252)
        else:
            # Fallback to pandas/numpy
            total_return = (1 + returns).prod() - 1
            cagr = (1 + total_return) ** (252 / n_days) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

        return PortfolioMetrics(
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            trading_days=n_days,
        )


# =============================================================================
# PORTFOLIO COMBINER
# =============================================================================

class BacktestCombiner:
    """
    Combine multiple backtests with specified weights.

    Example:
        combiner = BacktestCombiner(backtests, {"bt1": 0.6, "bt2": 0.4})
        portfolio = combiner.combine(initial_capital=1_000_000)
    """

    def __init__(
        self,
        backtests: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ):
        """
        Initialize the combiner.

        Args:
            backtests: Dictionary mapping backtest names to DataFrames.
            weights: Dictionary mapping backtest names to weights.
        """
        self._validate_inputs(backtests, weights)
        self.backtests = backtests
        self.weights = self._normalize_weights(weights)

    @staticmethod
    def _validate_inputs(
        backtests: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> None:
        """Validate inputs."""
        missing = set(weights.keys()) - set(backtests.keys())
        if missing:
            raise ValueError(f"Backtests not found for weights: {missing}")

        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"Negative weight not allowed: {name}={weight}")

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Total weight cannot be zero")

        if not np.isclose(total, 1.0, atol=1e-6):
            return {k: v / total for k, v in weights.items()}

        return weights.copy()

    def _align_returns(self) -> pd.DataFrame:
        """Create aligned returns DataFrame for all backtests."""
        # Get all unique dates
        all_dates = set()
        for name in self.weights.keys():
            all_dates.update(self.backtests[name]["date"].tolist())

        # Create aligned DataFrame
        returns_df = pd.DataFrame(index=pd.DatetimeIndex(sorted(all_dates)))
        returns_df.index.name = "date"

        for name in self.weights.keys():
            bt = self.backtests[name].set_index("date")
            returns_df[name] = bt["daily_return_decimal"].reindex(returns_df.index).fillna(0)

        return returns_df

    def combine(self, initial_capital: float = 1_000_000) -> CombinedPortfolio:
        """
        Combine backtests using weighted returns.

        Args:
            initial_capital: Initial portfolio value.

        Returns:
            CombinedPortfolio object with combined data.
        """
        returns_df = self._align_returns()

        # Calculate weighted returns
        weighted_returns = sum(
            returns_df[name] * weight
            for name, weight in self.weights.items()
        )

        # Build equity curve
        equity = [initial_capital]
        for ret in weighted_returns.iloc[1:]:
            equity.append(equity[-1] * (1 + ret))

        # Create combined DataFrame
        combined = pd.DataFrame({
            "date": returns_df.index,
            "equity": equity,
            "daily_return": weighted_returns.values * 100,
            "daily_return_decimal": weighted_returns.values,
        }).reset_index(drop=True)

        return CombinedPortfolio(
            data=combined,
            weights=self.weights,
            initial_capital=initial_capital,
            individual_returns=returns_df,
        )

    def get_weight_summary(self) -> str:
        """Get a formatted summary of weights."""
        lines = ["Portfolio Weights:"]
        for name, weight in sorted(self.weights.items()):
            lines.append(f"  {name}: {weight:.2%}")
        return "\n".join(lines)

    def save_combined(
        self,
        portfolio: CombinedPortfolio,
        output_dir: Union[str, Path] = BACKTESTS_DIR,
        filename: str = "combined_portfolio.parquet",
    ) -> Path:
        """
        Save combined portfolio to parquet.

        Args:
            portfolio: Combined portfolio to save.
            output_dir: Output directory.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        portfolio.data.to_parquet(filepath, index=False)

        return filepath


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_and_combine(
    data_dir: Union[str, Path] = BACKTESTS_DIR,
    weights: Optional[dict[str, float]] = None,
    initial_capital: float = 1_000_000,
) -> CombinedPortfolio:
    """
    Load backtests from parquet and combine them.

    Args:
        data_dir: Directory containing parquet files.
        weights: Backtest weights (default: equal weights).
        initial_capital: Initial portfolio value.

    Returns:
        CombinedPortfolio object.
    """
    from backtest_loader import load_backtests_from_parquet

    backtests = load_backtests_from_parquet(data_dir)

    if not backtests:
        raise ValueError(f"No backtests found in {data_dir}")

    if weights is None:
        n = len(backtests)
        weights = {name: 1.0 / n for name in backtests.keys()}

    combiner = BacktestCombiner(backtests, weights)
    return combiner.combine(initial_capital)


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Combine multiple backtests with weighted allocations."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BACKTESTS_DIR),
        help="Directory containing backtest parquet files.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help='Weights as JSON, e.g., \'{"bt1": 0.5, "bt2": 0.5}\'',
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000,
        help="Initial portfolio capital.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BACKTEST COMBINER")
    print("=" * 60)

    # Parse weights
    weights = json.loads(args.weights) if args.weights else None

    try:
        portfolio = load_and_combine(
            data_dir=args.data_dir,
            weights=weights,
            initial_capital=args.initial_capital,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please run backtest_loader.py first to convert backtests to parquet.")
        return

    # Load backtests for summary
    from backtest_loader import load_backtests_from_parquet
    backtests = load_backtests_from_parquet(args.data_dir)

    # Print summary
    print(f"\nLoaded backtests: {list(portfolio.weights.keys())}")
    combiner = BacktestCombiner(backtests, portfolio.weights)
    print(f"\n{combiner.get_weight_summary()}")

    metrics = portfolio.calculate_metrics()

    print(f"\nCombined Portfolio:")
    print(f"  Trading days: {metrics.trading_days}")
    print(f"  Date range: {portfolio.data['date'].min().date()} to {portfolio.data['date'].max().date()}")
    print(f"  Initial capital: ${args.initial_capital:,.0f}")
    print(f"  Final equity: ${portfolio.final_equity:,.0f}")
    print(f"  Total return: {metrics.total_return:.2%}")
    print(f"  CAGR: {metrics.cagr:.2%}")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")

    # Save
    filepath = combiner.save_combined(portfolio, args.data_dir)
    print(f"\nSaved to: {filepath}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
