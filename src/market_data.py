"""
Module for downloading market data (benchmark and VIX) from Yahoo Finance.

Usage:
    python market_data.py [--ticker TICKER] [--start DATE] [--end DATE]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yfinance as yf

from config import (
    CACHE_DIR,
    INTERMEDIATE_DIR,
    BENCHMARK_TICKERS,
    VIX_TICKER,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketData:
    """Container for market data."""
    ticker: str
    data: pd.DataFrame
    start_date: datetime
    end_date: datetime

    @property
    def returns(self) -> pd.Series:
        """Get daily returns as a Series indexed by date."""
        return self.data.set_index("date")["daily_return_decimal"].dropna()

    @property
    def prices(self) -> pd.Series:
        """Get closing prices as a Series indexed by date."""
        return self.data.set_index("date")["close"]


# =============================================================================
# MARKET DATA DOWNLOADER
# =============================================================================

class MarketDataDownloader:
    """
    Download and cache market data from Yahoo Finance.

    Example:
        downloader = MarketDataDownloader()
        spy_data = downloader.download_benchmark("SPY", "2020-01-01")
        vix_data = downloader.download_vix("2020-01-01")
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = CACHE_DIR):
        """
        Initialize the downloader.

        Args:
            cache_dir: Directory for caching data. None to disable caching.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, ticker: str, start: str, end: str) -> Optional[Path]:
        """Get cache file path for a ticker."""
        if not self.cache_dir:
            return None
        clean_ticker = ticker.replace("^", "").replace("/", "_")
        return self.cache_dir / f"{clean_ticker}_{start}_{end}.parquet"

    def _download_from_yahoo(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance."""
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data.columns = [c.lower() for c in data.columns]

        # Add adj close alias (auto_adjust makes close = adjusted close)
        if "adj close" not in data.columns and "close" in data.columns:
            data["adj close"] = data["close"]

        return data

    def download(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical data for a ticker.

        Args:
            ticker: Yahoo Finance ticker symbol.
            start_date: Start date.
            end_date: End date (default: today).
            use_cache: Whether to use cached data.

        Returns:
            DataFrame with OHLCV data.
        """
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date or datetime.now()).strftime("%Y-%m-%d")

        cache_path = self._get_cache_path(ticker, start_str, end_str)

        # Try cache first
        if use_cache and cache_path and cache_path.exists():
            return pd.read_parquet(cache_path)

        # Download from Yahoo
        data = self._download_from_yahoo(ticker, start_str, end_str)

        # Save to cache
        if cache_path:
            data.to_parquet(cache_path, index=False)

        return data

    def download_benchmark(
        self,
        ticker: str = "SPY",
        start_date: Union[str, datetime] = "2019-01-01",
        end_date: Optional[Union[str, datetime]] = None,
    ) -> MarketData:
        """
        Download benchmark data with returns.

        Args:
            ticker: Benchmark ticker (default: SPY).
            start_date: Start date.
            end_date: End date (default: today).

        Returns:
            MarketData object with benchmark data.
        """
        df = self.download(ticker, start_date, end_date)

        # Calculate returns
        price_col = "close"
        df["daily_return"] = df[price_col].pct_change() * 100
        df["daily_return_decimal"] = df[price_col].pct_change()

        return MarketData(
            ticker=ticker,
            data=df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date or datetime.now()),
        )

    def download_vix(
        self,
        start_date: Union[str, datetime] = "2019-01-01",
        end_date: Optional[Union[str, datetime]] = None,
    ) -> MarketData:
        """
        Download VIX data.

        Args:
            start_date: Start date.
            end_date: End date (default: today).

        Returns:
            MarketData object with VIX data.
        """
        df = self.download(VIX_TICKER, start_date, end_date)

        # Add VIX specific columns
        df["vix"] = df["close"]
        df["vix_close"] = df["close"]
        df["daily_return_decimal"] = df["close"].pct_change()

        return MarketData(
            ticker=VIX_TICKER,
            data=df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date or datetime.now()),
        )

    def get_benchmark_returns(
        self,
        ticker: str = "SPY",
        start_date: Union[str, datetime] = "2019-01-01",
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.Series:
        """
        Get benchmark returns as a Series indexed by date.

        Args:
            ticker: Benchmark ticker.
            start_date: Start date.
            end_date: End date.

        Returns:
            Series of daily returns (decimal).
        """
        market_data = self.download_benchmark(ticker, start_date, end_date)
        returns = market_data.returns
        returns.name = ticker
        return returns

    def get_vix_series(
        self,
        start_date: Union[str, datetime] = "2019-01-01",
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.Series:
        """
        Get VIX values as a Series indexed by date.

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            Series of VIX values.
        """
        vix_data = self.download_vix(start_date, end_date)
        vix = vix_data.data.set_index("date")["vix"]
        vix.name = "VIX"
        return vix

    def align_with_strategy(
        self,
        strategy_returns: pd.Series,
        benchmark_ticker: str = "SPY",
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Download and align benchmark/VIX data with strategy returns.

        Args:
            strategy_returns: Strategy returns indexed by date.
            benchmark_ticker: Benchmark ticker symbol.

        Returns:
            Tuple of (strategy_returns, benchmark_returns, vix) aligned to same dates.
        """
        start_date = strategy_returns.index.min()
        end_date = strategy_returns.index.max()

        benchmark_returns = self.get_benchmark_returns(benchmark_ticker, start_date, end_date)
        vix = self.get_vix_series(start_date, end_date)

        # Align all series
        common_idx = (
            strategy_returns.index
            .intersection(benchmark_returns.index)
            .intersection(vix.index)
        )

        return (
            strategy_returns.reindex(common_idx),
            benchmark_returns.reindex(common_idx),
            vix.reindex(common_idx),
        )

    def save(
        self,
        market_data: MarketData,
        output_dir: Union[str, Path] = INTERMEDIATE_DIR,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save market data to parquet.

        Args:
            market_data: MarketData object to save.
            output_dir: Output directory.
            filename: Output filename (auto-generated if None).

        Returns:
            Path to saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            clean_ticker = market_data.ticker.replace("^", "").lower()
            filename = f"{clean_ticker}.parquet"

        filepath = output_dir / filename
        market_data.data.to_parquet(filepath, index=False)

        return filepath


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_market_data(
    data_dir: Union[str, Path] = INTERMEDIATE_DIR,
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Load benchmark returns and VIX from parquet files.

    Args:
        data_dir: Directory containing parquet files.

    Returns:
        Tuple of (benchmark_returns, vix) or (None, None) if not found.
    """
    data_dir = Path(data_dir)

    # Load benchmark
    benchmark_returns = None
    benchmark_files = list(data_dir.glob("benchmark_*.parquet"))
    if not benchmark_files:
        benchmark_files = list(data_dir.glob("spy.parquet"))

    if benchmark_files:
        df = pd.read_parquet(benchmark_files[0])
        df["date"] = pd.to_datetime(df["date"])
        benchmark_returns = df.set_index("date")["daily_return_decimal"].dropna()

    # Load VIX
    vix = None
    vix_path = data_dir / "vix.parquet"
    if vix_path.exists():
        df = pd.read_parquet(vix_path)
        df["date"] = pd.to_datetime(df["date"])
        vix = df.set_index("date")["vix"]

    return benchmark_returns, vix


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Download market data from Yahoo Finance."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Benchmark ticker symbol.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(INTERMEDIATE_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-vix",
        action="store_true",
        help="Skip VIX download.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MARKET DATA DOWNLOADER")
    print("=" * 60)

    downloader = MarketDataDownloader()
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    # Download benchmark
    print(f"\nDownloading {args.ticker}...")
    print(f"  Period: {args.start} to {end_date}")

    try:
        benchmark = downloader.download_benchmark(args.ticker, args.start, end_date)
        filepath = downloader.save(
            benchmark,
            args.output_dir,
            f"benchmark_{args.ticker.lower()}.parquet"
        )

        returns = benchmark.returns
        print(f"  Downloaded: {len(benchmark.data)} rows")
        print(f"  Saved to: {filepath}")
        print(f"  Mean daily return: {returns.mean():.4%}")
        print(f"  Total return: {(1 + returns).prod() - 1:.2%}")

    except Exception as e:
        print(f"  Error: {e}")

    # Download VIX
    if not args.no_vix:
        print(f"\nDownloading VIX...")

        try:
            vix_data = downloader.download_vix(args.start, end_date)
            filepath = downloader.save(vix_data, args.output_dir, "vix.parquet")

            vix = vix_data.data["vix"]
            print(f"  Downloaded: {len(vix_data.data)} rows")
            print(f"  Saved to: {filepath}")
            print(f"  Mean VIX: {vix.mean():.2f}")
            print(f"  Current VIX: {vix.iloc[-1]:.2f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Show available benchmarks
    print(f"\nAvailable benchmark tickers:")
    for ticker, name in BENCHMARK_TICKERS.items():
        print(f"  {ticker}: {name}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
