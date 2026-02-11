"""
Module for loading and processing trade-level data from QuantConnect exports.

Each row in a QuantConnect trade log contains both entry and exit on the same line.
This module loads those files, validates columns, computes derived fields, and can
split combined rows into individual entry/exit legs.

Usage:
    loader = TradesLoader()
    df = loader.load("Measured Light Brown Rabbit_trades")
    split_df = TradesLoader.split_trades(df)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import TRADES_DIR


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

REQUIRED_TRADE_COLUMNS = [
    "Entry Time",
    "Exit Time",
    "Direction",
    "Entry Price",
    "Exit Price",
    "P&L",
    "IsWin",
]


# =============================================================================
# TRADES LOADER
# =============================================================================

class TradesLoader:
    """
    Loader for QuantConnect trade log CSV files.

    Each row contains a complete round-trip trade (entry + exit).
    The loader validates columns, parses timestamps, computes holding
    period, and can split rows into individual entry/exit legs.

    Example:
        loader = TradesLoader()
        df = loader.load("Measured Light Brown Rabbit_trades")
        legs = TradesLoader.split_trades(df)
    """

    def __init__(self, data_dir: Union[str, Path] = TRADES_DIR):
        self.data_dir = Path(data_dir)

    def list_trade_files(self) -> list[str]:
        """List all CSV trade files in the data directory (stems only)."""
        return sorted(f.stem for f in self.data_dir.glob("*.csv"))

    def find_file(self, name: str) -> Optional[Path]:
        """
        Find a trade file by name.

        Args:
            name: File name with or without .csv extension.

        Returns:
            Path to the file or None if not found.
        """
        name_path = Path(name)
        if name_path.suffix.lower() == ".csv":
            filepath = self.data_dir / name
        else:
            filepath = self.data_dir / f"{name}.csv"
        return filepath if filepath.exists() else None

    def load(self, name: str) -> pd.DataFrame:
        """
        Load a trade file by name.

        Raises:
            FileNotFoundError: If file not found.
        """
        filepath = self.find_file(name)
        if filepath is None:
            raise FileNotFoundError(f"Trade file not found: {name}")
        return self.load_file(filepath)

    def load_file(self, filepath: Path) -> pd.DataFrame:
        """Load a trade file from a specific path."""
        df = pd.read_csv(filepath)
        return self._validate_and_process(df)

    def load_from_bytes(self, data: bytes, filename: str) -> pd.DataFrame:
        """Load a trade file from bytes (for file uploads)."""
        df = pd.read_csv(io.BytesIO(data))
        return self._validate_and_process(df)

    def _validate_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and process data types."""
        missing = [c for c in REQUIRED_TRADE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required trade columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Parse timestamps as UTC then convert to US/Eastern
        df["Entry Time"] = pd.to_datetime(df["Entry Time"], utc=True).dt.tz_convert("US/Eastern")
        df["Exit Time"] = pd.to_datetime(df["Exit Time"], utc=True).dt.tz_convert("US/Eastern")

        # Numeric columns
        for col in ["Entry Price", "Exit Price", "P&L", "Quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Fees" in df.columns:
            df["Fees"] = pd.to_numeric(df["Fees"], errors="coerce").fillna(0.0)

        if "Drawdown" in df.columns:
            df["Drawdown"] = pd.to_numeric(df["Drawdown"], errors="coerce").fillna(0.0)

        # Boolean
        df["IsWin"] = df["IsWin"].astype(bool)

        # Strip tab prefix from Order Ids
        if "Order Ids" in df.columns:
            df["Order Ids"] = df["Order Ids"].astype(str).str.strip().str.lstrip("\t")

        # Compute holding period
        df["Holding Period"] = df["Exit Time"] - df["Entry Time"]
        df["Holding Hours"] = df["Holding Period"].dt.total_seconds() / 3600.0

        # Sort by entry time
        df = df.sort_values("Entry Time").reset_index(drop=True)

        return df

    @staticmethod
    def split_trades(df: pd.DataFrame) -> pd.DataFrame:
        """
        Split combined entry+exit rows into individual legs.

        Each input row produces two output rows:
        - Entry leg: uses Entry Time, Entry Price, original Direction
        - Exit leg: uses Exit Time, Exit Price, opposite Direction

        Returns:
            DataFrame with columns: Timestamp, Symbol, Leg, Direction,
            Price, Quantity, Trade_Index
        """
        rows = []
        for idx, trade in df.iterrows():
            symbol = trade.get("Symbols", "")
            quantity = trade.get("Quantity", 0)
            direction = trade["Direction"]
            exit_direction = "Buy" if direction == "Sell" else "Sell"

            rows.append({
                "Timestamp": trade["Entry Time"],
                "Symbol": symbol,
                "Leg": "Entry",
                "Direction": direction,
                "Price": trade["Entry Price"],
                "Quantity": quantity,
                "Trade_Index": idx,
            })
            rows.append({
                "Timestamp": trade["Exit Time"],
                "Symbol": symbol,
                "Leg": "Exit",
                "Direction": exit_direction,
                "Price": trade["Exit Price"],
                "Quantity": quantity,
                "Trade_Index": idx,
            })

        split_df = pd.DataFrame(rows)
        if not split_df.empty:
            split_df = split_df.sort_values("Timestamp").reset_index(drop=True)
        return split_df

    @staticmethod
    def format_for_export(df: pd.DataFrame) -> pd.DataFrame:
        """
        Format trades for export: one row per leg, simplified columns.

        Splits each round-trip trade into entry/exit rows and keeps only:
        - Date: entry or exit day (date only, no time)
        - Symbol: option root (e.g. ``SPXW`` from ``SPXW  190114P02460000``)
        - Direction: ``BUY`` or ``SELL`` (exit uses opposite direction)
        - Price: entry or exit price

        Returns:
            DataFrame with columns ``Date``, ``Symbol``, ``Direction``, ``Price``.
        """
        import re

        split = TradesLoader.split_trades(df)
        if split.empty:
            return pd.DataFrame(columns=["Date", "Symbol", "Direction", "Price"])

        # Date: day only
        split["Date"] = pd.to_datetime(split["Timestamp"]).dt.date

        # Symbol root: strip OCC-style suffix (digits + P/C + strike)
        split["Symbol"] = split["Symbol"].apply(
            lambda s: re.split(r"\s+\d", str(s).strip(), maxsplit=1)[0].strip()
        )

        # Direction: uppercase
        split["Direction"] = split["Direction"].str.upper()

        return split[["Date", "Symbol", "Direction", "Price"]].reset_index(drop=True)
