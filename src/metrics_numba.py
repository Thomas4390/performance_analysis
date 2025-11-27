"""
Optimized metrics calculation using Numba for JIT compilation.

This module provides high-performance implementations of common
financial metrics calculations using Numba's just-in-time compilation.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange
from typing import Tuple


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True)
def _cumulative_product(returns: np.ndarray) -> np.ndarray:
    """Calculate cumulative product of (1 + returns)."""
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[0] = 1.0 + returns[0]
    for i in range(1, n):
        result[i] = result[i - 1] * (1.0 + returns[i])
    return result


@jit(nopython=True, cache=True)
def _running_max(arr: np.ndarray) -> np.ndarray:
    """Calculate running maximum."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = max(result[i - 1], arr[i])
    return result


@jit(nopython=True, cache=True)
def _drawdown_series(cumulative: np.ndarray) -> np.ndarray:
    """Calculate drawdown series from cumulative returns."""
    running_max = _running_max(cumulative)
    n = len(cumulative)
    drawdown = np.empty(n, dtype=np.float64)
    for i in range(n):
        if running_max[i] != 0:
            drawdown[i] = (cumulative[i] - running_max[i]) / running_max[i]
        else:
            drawdown[i] = 0.0
    return drawdown


@jit(nopython=True, cache=True)
def _max_drawdown_with_indices(cumulative: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its start/end indices.

    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    n = len(cumulative)
    max_dd = 0.0
    peak_idx = 0
    trough_idx = 0

    running_max = cumulative[0]
    running_max_idx = 0

    for i in range(1, n):
        if cumulative[i] > running_max:
            running_max = cumulative[i]
            running_max_idx = i
        else:
            dd = (cumulative[i] - running_max) / running_max if running_max != 0 else 0.0
            if dd < max_dd:
                max_dd = dd
                peak_idx = running_max_idx
                trough_idx = i

    return max_dd, peak_idx, trough_idx


@jit(nopython=True, cache=True)
def _total_return(returns: np.ndarray) -> float:
    """Calculate total return from daily returns."""
    product = 1.0
    for r in returns:
        product *= (1.0 + r)
    return product - 1.0


@jit(nopython=True, cache=True)
def _mean(arr: np.ndarray) -> float:
    """Calculate mean of array."""
    if len(arr) == 0:
        return 0.0
    total = 0.0
    for x in arr:
        total += x
    return total / len(arr)


@jit(nopython=True, cache=True)
def _std(arr: np.ndarray) -> float:
    """Calculate standard deviation of array."""
    n = len(arr)
    if n < 2:
        return 0.0

    mean_val = _mean(arr)
    variance = 0.0
    for x in arr:
        variance += (x - mean_val) ** 2

    return np.sqrt(variance / (n - 1))  # Sample std (ddof=1)


@jit(nopython=True, cache=True)
def _downside_std(returns: np.ndarray) -> float:
    """Calculate downside standard deviation (only negative returns)."""
    # Count negative returns
    count = 0
    for r in returns:
        if r < 0:
            count += 1

    if count < 2:
        return 0.0

    # Extract negative returns
    negative = np.empty(count, dtype=np.float64)
    idx = 0
    for r in returns:
        if r < 0:
            negative[idx] = r
            idx += 1

    return _std(negative)


@jit(nopython=True, cache=True)
def _win_rate_and_profit_factor(returns: np.ndarray) -> Tuple[float, float]:
    """Calculate win rate and profit factor."""
    winners_sum = 0.0
    losers_sum = 0.0
    winners_count = 0
    losers_count = 0

    for r in returns:
        if r > 0:
            winners_sum += r
            winners_count += 1
        elif r < 0:
            losers_sum += r
            losers_count += 1

    total_trades = winners_count + losers_count
    win_rate = winners_count / total_trades if total_trades > 0 else 0.0
    profit_factor = winners_sum / abs(losers_sum) if losers_sum != 0 else 999.99

    return win_rate, min(profit_factor, 999.99)


@jit(nopython=True, cache=True)
def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate covariance between two arrays."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = _mean(x)
    mean_y = _mean(y)

    cov = 0.0
    for i in range(n):
        cov += (x[i] - mean_x) * (y[i] - mean_y)

    return cov / (n - 1)


@jit(nopython=True, cache=True)
def _variance(arr: np.ndarray) -> float:
    """Calculate variance of array."""
    std_val = _std(arr)
    return std_val ** 2


@jit(nopython=True, cache=True)
def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation between two arrays."""
    std_x = _std(x)
    std_y = _std(y)

    if std_x == 0 or std_y == 0:
        return 0.0

    cov = _covariance(x, y)
    return cov / (std_x * std_y)


# =============================================================================
# HIGH-LEVEL METRICS FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True)
def calculate_core_metrics(
    returns: np.ndarray,
    trading_days_per_year: int = 252,
) -> Tuple[float, float, float, float, float, float, float, float, int, int]:
    """
    Calculate core performance metrics using Numba.

    Args:
        returns: Array of daily returns (decimal format).
        trading_days_per_year: Trading days per year for annualization.

    Returns:
        Tuple containing:
        - total_return
        - cagr
        - volatility (annualized)
        - sharpe_ratio
        - sortino_ratio
        - max_drawdown
        - win_rate
        - profit_factor
        - dd_peak_idx
        - dd_trough_idx
    """
    n_days = len(returns)

    # Total return
    total_return = _total_return(returns)

    # CAGR
    years = n_days / trading_days_per_year
    if years > 0 and total_return > -1:
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Volatility (annualized)
    daily_std = _std(returns)
    volatility = daily_std * np.sqrt(trading_days_per_year)

    # Sharpe ratio
    daily_mean = _mean(returns)
    sharpe = (daily_mean / daily_std * np.sqrt(trading_days_per_year)) if daily_std > 0 else 0.0

    # Sortino ratio
    downside_std_val = _downside_std(returns)
    ann_downside_std = downside_std_val * np.sqrt(trading_days_per_year)
    sortino = (daily_mean * trading_days_per_year / ann_downside_std) if ann_downside_std > 0 else 0.0

    # Max drawdown with indices
    cumulative = _cumulative_product(returns)
    max_dd, dd_peak_idx, dd_trough_idx = _max_drawdown_with_indices(cumulative)

    # Win rate and profit factor
    win_rate, profit_factor = _win_rate_and_profit_factor(returns)

    return (
        total_return,
        cagr,
        volatility,
        sharpe,
        sortino,
        max_dd,
        win_rate,
        profit_factor,
        dd_peak_idx,
        dd_trough_idx,
    )


@jit(nopython=True, cache=True)
def calculate_benchmark_metrics(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    strategy_cagr: float,
    trading_days_per_year: int = 252,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate benchmark comparison metrics using Numba.

    Args:
        strategy_returns: Strategy daily returns.
        benchmark_returns: Benchmark daily returns.
        strategy_cagr: Pre-calculated strategy CAGR.
        trading_days_per_year: Trading days per year.

    Returns:
        Tuple containing:
        - benchmark_return
        - benchmark_cagr
        - benchmark_volatility
        - benchmark_sharpe
        - beta
        - alpha
        - correlation
        - information_ratio
    """
    n_days = len(benchmark_returns)

    # Benchmark total return
    bench_return = _total_return(benchmark_returns)

    # Benchmark CAGR
    years = n_days / trading_days_per_year
    if years > 0 and bench_return > -1:
        bench_cagr = (1.0 + bench_return) ** (1.0 / years) - 1.0
    else:
        bench_cagr = 0.0

    # Benchmark volatility
    bench_std = _std(benchmark_returns)
    bench_vol = bench_std * np.sqrt(trading_days_per_year)

    # Benchmark Sharpe
    bench_mean = _mean(benchmark_returns)
    bench_sharpe = (bench_mean / bench_std * np.sqrt(trading_days_per_year)) if bench_std > 0 else 0.0

    # Beta
    cov = _covariance(strategy_returns, benchmark_returns)
    var = _variance(benchmark_returns)
    beta = cov / var if var > 0 else 0.0

    # Alpha
    alpha = strategy_cagr - (beta * bench_cagr)

    # Correlation
    corr = _correlation(strategy_returns, benchmark_returns)

    # Information ratio
    n = len(strategy_returns)
    active_returns = np.empty(n, dtype=np.float64)
    for i in range(n):
        active_returns[i] = strategy_returns[i] - benchmark_returns[i]

    active_mean = _mean(active_returns)
    active_std = _std(active_returns)
    tracking_error = active_std * np.sqrt(trading_days_per_year)
    info_ratio = (active_mean * trading_days_per_year / tracking_error) if tracking_error > 0 else 0.0

    return (
        bench_return,
        bench_cagr,
        bench_vol,
        bench_sharpe,
        beta,
        alpha,
        corr,
        info_ratio,
    )


# =============================================================================
# UTILITY FUNCTIONS FOR PANDAS INTEGRATION
# =============================================================================

def calculate_all_metrics(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    dates: np.ndarray | None = None,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Calculate all metrics and return as dictionary.

    This is the main entry point for calculating metrics with Numba optimization.

    Args:
        strategy_returns: Strategy daily returns as numpy array.
        benchmark_returns: Optional benchmark returns as numpy array.
        dates: Optional array of dates for drawdown period identification.
        trading_days_per_year: Trading days per year.

    Returns:
        Dictionary with all calculated metrics.
    """
    # Ensure float64 for Numba
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64)

    # Calculate core metrics
    (
        total_return,
        cagr,
        volatility,
        sharpe,
        sortino,
        max_dd,
        win_rate,
        profit_factor,
        dd_peak_idx,
        dd_trough_idx,
    ) = calculate_core_metrics(strategy_returns, trading_days_per_year)

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Best/worst day
    best_day = float(np.max(strategy_returns))
    worst_day = float(np.min(strategy_returns))

    result = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "dd_peak_idx": dd_peak_idx,
        "dd_trough_idx": dd_trough_idx,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "best_day": best_day,
        "worst_day": worst_day,
        "trading_days": len(strategy_returns),
    }

    # Add dates if available
    if dates is not None:
        result["dd_start_date"] = dates[dd_peak_idx]
        result["dd_end_date"] = dates[dd_trough_idx]

    # Calculate benchmark metrics if provided
    if benchmark_returns is not None:
        benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

        (
            bench_return,
            bench_cagr,
            bench_vol,
            bench_sharpe,
            beta,
            alpha,
            corr,
            info_ratio,
        ) = calculate_benchmark_metrics(
            strategy_returns,
            benchmark_returns,
            cagr,
            trading_days_per_year,
        )

        result.update({
            "benchmark_return": bench_return,
            "benchmark_cagr": bench_cagr,
            "benchmark_volatility": bench_vol,
            "benchmark_sharpe": bench_sharpe,
            "beta": beta,
            "alpha": alpha,
            "correlation": corr,
            "information_ratio": info_ratio,
        })

    return result


# =============================================================================
# WARMUP FUNCTION
# =============================================================================

def warmup():
    """
    Warm up Numba JIT compilation by running on small dummy data.

    Call this at module import or application startup for better
    performance on first real calculation.
    """
    dummy_returns = np.random.randn(100).astype(np.float64) * 0.01
    dummy_benchmark = np.random.randn(100).astype(np.float64) * 0.01

    # Trigger compilation
    calculate_all_metrics(dummy_returns, dummy_benchmark)


# =============================================================================
# ROLLING METRICS FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True)
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling mean efficiently."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan

    # Initial window sum
    window_sum = 0.0
    for i in range(window):
        window_sum += arr[i]
    result[window-1] = window_sum / window

    # Slide window
    for i in range(window, n):
        window_sum = window_sum - arr[i - window] + arr[i]
        result[i] = window_sum / window

    return result


@jit(nopython=True, cache=True)
def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling standard deviation efficiently using online algorithm."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan

    # Initialize first window
    mean_val = 0.0
    m2 = 0.0
    for i in range(window):
        delta = arr[i] - mean_val
        mean_val += delta / (i + 1)
        delta2 = arr[i] - mean_val
        m2 += delta * delta2

    result[window-1] = np.sqrt(m2 / (window - 1))

    # Slide window using update formulas
    for i in range(window, n):
        old_val = arr[i - window]
        new_val = arr[i]

        # Remove old value and add new value to mean
        old_mean = mean_val
        mean_val += (new_val - old_val) / window

        # Update M2 (sum of squared deviations)
        m2 += (new_val - old_val) * (new_val - mean_val + old_val - old_mean)

        result[i] = np.sqrt(m2 / (window - 1)) if m2 > 0 else 0.0

    return result


@jit(nopython=True, cache=True, parallel=True)
def rolling_stats_parallel(arr: np.ndarray, window: int) -> tuple:
    """Calculate rolling mean and std in parallel."""
    n = len(arr)
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)

    means[:window-1] = np.nan
    stds[:window-1] = np.nan

    for i in prange(window - 1, n):
        # Calculate mean
        total = 0.0
        for j in range(i - window + 1, i + 1):
            total += arr[j]
        mean_val = total / window
        means[i] = mean_val

        # Calculate variance
        variance = 0.0
        for j in range(i - window + 1, i + 1):
            variance += (arr[j] - mean_val) ** 2
        stds[i] = np.sqrt(variance / (window - 1))

    return means, stds


@jit(nopython=True, cache=True)
def find_regime_boundaries(regime_indices: np.ndarray) -> tuple:
    """
    Find regime change boundaries efficiently.

    Returns:
        Tuple of (start_indices, end_indices, regime_at_start)
    """
    n = len(regime_indices)
    if n == 0:
        # Return empty int64 arrays for empty input
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty

    # Count changes first
    n_changes = 1
    for i in range(1, n):
        if regime_indices[i] != regime_indices[i-1]:
            n_changes += 1

    # Allocate arrays
    starts = np.empty(n_changes, dtype=np.int64)
    ends = np.empty(n_changes, dtype=np.int64)
    regimes = np.empty(n_changes, dtype=np.int64)

    # Fill arrays
    starts[0] = 0
    regimes[0] = regime_indices[0]
    change_idx = 0

    for i in range(1, n):
        if regime_indices[i] != regime_indices[i-1]:
            ends[change_idx] = i
            change_idx += 1
            starts[change_idx] = i
            regimes[change_idx] = regime_indices[i]

    ends[change_idx] = n

    return starts, ends, regimes


@jit(nopython=True, cache=True)
def calculate_regime_stats_fast(
    returns: np.ndarray,
    regime_indices: np.ndarray,
    n_regimes: int,
    trading_days: int = 252,
) -> tuple:
    """
    Calculate statistics for each regime efficiently.

    Returns:
        Tuple of arrays: (days, total_returns, ann_returns, ann_vols, sharpes, win_rates)
    """
    days = np.zeros(n_regimes, dtype=np.int64)
    total_returns = np.zeros(n_regimes, dtype=np.float64)
    ann_returns = np.zeros(n_regimes, dtype=np.float64)
    ann_vols = np.zeros(n_regimes, dtype=np.float64)
    sharpes = np.zeros(n_regimes, dtype=np.float64)
    win_rates = np.zeros(n_regimes, dtype=np.float64)

    # First pass: count days per regime
    for i in range(len(returns)):
        regime = regime_indices[i]
        if 0 <= regime < n_regimes:
            days[regime] += 1

    # Allocate temp arrays for each regime's returns
    # We need to calculate stats for each regime separately
    for regime in range(n_regimes):
        if days[regime] == 0:
            continue

        # Extract returns for this regime
        regime_returns = np.empty(days[regime], dtype=np.float64)
        idx = 0
        for i in range(len(returns)):
            if regime_indices[i] == regime:
                regime_returns[idx] = returns[i]
                idx += 1

        # Calculate stats
        n = len(regime_returns)

        # Total return
        product = 1.0
        for r in regime_returns:
            product *= (1.0 + r)
        total_returns[regime] = (product - 1.0) * 100  # In percentage

        # Mean and std
        mean_ret = 0.0
        for r in regime_returns:
            mean_ret += r
        mean_ret /= n

        variance = 0.0
        for r in regime_returns:
            variance += (r - mean_ret) ** 2
        std_ret = np.sqrt(variance / (n - 1)) if n > 1 else 0.0

        # Annualized return and volatility
        ann_returns[regime] = mean_ret * trading_days * 100
        ann_vols[regime] = std_ret * np.sqrt(trading_days) * 100

        # Sharpe
        if std_ret > 0:
            sharpes[regime] = (mean_ret / std_ret) * np.sqrt(trading_days)

        # Win rate
        wins = 0
        non_zero = 0
        for r in regime_returns:
            if r > 0:
                wins += 1
                non_zero += 1
            elif r < 0:
                non_zero += 1

        if non_zero > 0:
            win_rates[regime] = (wins / non_zero) * 100

    return days, total_returns, ann_returns, ann_vols, sharpes, win_rates


# =============================================================================
# CLI MAIN FOR TESTING
# =============================================================================

def main():
    """Test and benchmark the Numba implementations."""
    import time
    import pandas as pd

    print("=" * 60)
    print("  NUMBA METRICS BENCHMARK")
    print("=" * 60)

    # Load test data
    try:
        portfolio = pd.read_parquet("../data/intermediate/combined_portfolio.parquet")
        returns = portfolio["daily_return_decimal"].values.astype(np.float64)
        dates = portfolio["date"].values
        print(f"\nLoaded {len(returns)} returns from portfolio data")
    except FileNotFoundError:
        print("\nUsing random test data")
        np.random.seed(42)
        returns = np.random.randn(2500).astype(np.float64) * 0.01
        dates = None

    # Warmup
    print("\nWarming up Numba JIT...")
    warmup()

    # Benchmark Numba
    print("\nRunning Numba calculation...")
    n_iterations = 100

    start = time.perf_counter()
    for _ in range(n_iterations):
        numba_result = calculate_all_metrics(returns, dates=dates)
    numba_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Numba average time: {numba_time:.3f} ms")

    # Compare with pandas/numpy implementation
    print("\nRunning pandas/numpy calculation...")

    returns_series = pd.Series(returns)

    start = time.perf_counter()
    for _ in range(n_iterations):
        # Replicate original calculation
        total_return = (1 + returns_series).prod() - 1
        n_days = len(returns_series)
        cagr = (1 + total_return) ** (252 / n_days) - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)

        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
    pandas_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Pandas average time: {pandas_time:.3f} ms")
    print(f"\nSpeedup: {pandas_time / numba_time:.1f}x")

    # Validate results
    print("\n" + "-" * 40)
    print("RESULTS VALIDATION")
    print("-" * 40)

    # Pandas calculation for comparison
    total_return_pd = float((1 + returns_series).prod() - 1)
    cagr_pd = float((1 + total_return_pd) ** (252 / len(returns_series)) - 1)
    volatility_pd = float(returns_series.std() * np.sqrt(252))
    sharpe_pd = float(returns_series.mean() / returns_series.std() * np.sqrt(252))

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd_pd = float(drawdown.min())

    print(f"\n{'Metric':<20} {'Numba':>15} {'Pandas':>15} {'Match':>10}")
    print("-" * 60)

    metrics_to_check = [
        ("Total Return", numba_result["total_return"], total_return_pd),
        ("CAGR", numba_result["cagr"], cagr_pd),
        ("Volatility", numba_result["volatility"], volatility_pd),
        ("Sharpe Ratio", numba_result["sharpe_ratio"], sharpe_pd),
        ("Max Drawdown", numba_result["max_drawdown"], max_dd_pd),
    ]

    all_match = True
    for name, numba_val, pandas_val in metrics_to_check:
        match = np.isclose(numba_val, pandas_val, rtol=1e-6)
        all_match = all_match and match
        match_str = "✓" if match else "✗"
        print(f"{name:<20} {numba_val:>15.6f} {pandas_val:>15.6f} {match_str:>10}")

    print("\n" + "=" * 60)
    if all_match:
        print("  ✓ All metrics match! Numba implementation validated.")
    else:
        print("  ✗ Some metrics differ! Please investigate.")
    print("=" * 60)

    return numba_result


if __name__ == "__main__":
    main()
