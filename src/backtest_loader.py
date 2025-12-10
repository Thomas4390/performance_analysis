"""
Module for loading backtest data from various file formats.

Supports: ODS, XLSX, XLS, CSV, Parquet

Usage:
    python backtest_loader.py [--data-dir DIR] [--output-dir DIR] [--backtest NAME]
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config import (
    BACKTESTS_DIR,
    FileFormat,
    REQUIRED_BACKTEST_COLUMNS,
    COLUMN_ALIASES,
    normalize_column_name,
)


# =============================================================================
# FILE READERS (Strategy Pattern)
# =============================================================================

class FileReader(ABC):
    """Abstract base class for file readers."""

    @abstractmethod
    def read(self, filepath: Path) -> pd.DataFrame:
        """Read a file and return a DataFrame."""
        pass

    @staticmethod
    def get_reader(format: FileFormat) -> "FileReader":
        """Factory method to get the appropriate reader."""
        readers = {
            FileFormat.ODS: OdsReader(),
            FileFormat.XLSX: ExcelReader(),
            FileFormat.XLS: ExcelReader(),
            FileFormat.CSV: CsvReader(),
            FileFormat.PARQUET: ParquetReader(),
        }
        return readers[format]


class OdsReader(FileReader):
    """Reader for ODS (OpenDocument Spreadsheet) files."""

    def read(self, filepath: Path) -> pd.DataFrame:
        return pd.read_excel(filepath, engine="odf")


class ExcelReader(FileReader):
    """Reader for Excel files (XLSX, XLS)."""

    def read(self, filepath: Path) -> pd.DataFrame:
        return pd.read_excel(filepath)


class CsvReader(FileReader):
    """Reader for CSV files."""

    def read(self, filepath: Path) -> pd.DataFrame:
        # Try different separators
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(filepath, sep=sep)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(filepath)


class ParquetReader(FileReader):
    """Reader for Parquet files."""

    def read(self, filepath: Path) -> pd.DataFrame:
        return pd.read_parquet(filepath)


# =============================================================================
# BACKTEST LOADER
# =============================================================================

class BacktestLoader:
    """
    Loader for backtest data from various file formats.

    Supports ODS, XLSX, XLS, CSV, and Parquet files.

    Example:
        loader = BacktestLoader("data/raw")
        df = loader.load("my_backtest.csv")
        loader.save_parquet(df, "my_backtest")
    """

    SUPPORTED_EXTENSIONS = {".ods", ".xlsx", ".xls", ".csv", ".parquet"}

    def __init__(
        self,
        data_dir: Union[str, Path] = BACKTESTS_DIR,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the backtest loader.

        Args:
            data_dir: Directory containing backtest files.
            output_dir: Directory for output parquet files (optional).
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir

    def list_backtests(self) -> list[str]:
        """
        List all available backtest files.

        Returns:
            List of backtest names (without extension).
        """
        backtests = []
        for ext in self.SUPPORTED_EXTENSIONS:
            backtests.extend(f.stem for f in self.data_dir.glob(f"*{ext}"))
        return sorted(set(backtests))

    def find_file(self, name: str) -> Optional[Path]:
        """
        Find a backtest file by name (tries all supported extensions).

        Args:
            name: Backtest name (with or without extension).

        Returns:
            Path to the file or None if not found.
        """
        # If name has extension, use it directly
        name_path = Path(name)
        if name_path.suffix in self.SUPPORTED_EXTENSIONS:
            filepath = self.data_dir / name
            return filepath if filepath.exists() else None

        # Try all supported extensions
        for ext in self.SUPPORTED_EXTENSIONS:
            filepath = self.data_dir / f"{name}{ext}"
            if filepath.exists():
                return filepath

        return None

    def load(self, name: str) -> pd.DataFrame:
        """
        Load a backtest from a file.

        Args:
            name: Backtest name (with or without extension).

        Returns:
            DataFrame with standardized columns.

        Raises:
            FileNotFoundError: If file not found.
            ValueError: If required columns are missing.
        """
        filepath = self.find_file(name)
        if filepath is None:
            raise FileNotFoundError(f"Backtest not found: {name}")

        return self.load_file(filepath)

    def load_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load a backtest from a specific file path.

        Args:
            filepath: Path to the file.

        Returns:
            DataFrame with standardized columns.
        """
        filepath = Path(filepath)
        format = FileFormat.from_extension(filepath)
        reader = FileReader.get_reader(format)

        df = reader.read(filepath)
        df = self._normalize_columns(df)
        df = self._validate_and_process(df)

        return df

    def load_from_bytes(
        self,
        data: bytes,
        filename: str,
    ) -> pd.DataFrame:
        """
        Load a backtest from bytes (for file uploads).

        Args:
            data: File content as bytes.
            filename: Original filename (used to determine format).

        Returns:
            DataFrame with standardized columns.
        """
        import io

        filepath = Path(filename)
        format = FileFormat.from_extension(filepath)

        if format == FileFormat.CSV:
            df = pd.read_csv(io.BytesIO(data))
        elif format == FileFormat.PARQUET:
            df = pd.read_parquet(io.BytesIO(data))
        elif format in (FileFormat.XLSX, FileFormat.XLS):
            df = pd.read_excel(io.BytesIO(data))
        elif format == FileFormat.ODS:
            df = pd.read_excel(io.BytesIO(data), engine="odf")
        else:
            raise ValueError(f"Unsupported format: {format}")

        df = self._normalize_columns(df)
        df = self._validate_and_process(df)

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        column_mapping = {}

        for col in df.columns:
            normalized = normalize_column_name(str(col))
            if normalized:
                column_mapping[col] = normalized

        if column_mapping:
            df = df.rename(columns=column_mapping)

        return df

    def _validate_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and process data."""
        # Check required columns
        missing = [col for col in REQUIRED_BACKTEST_COLUMNS if col not in df.columns]
        if missing:
            available = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing}. Available: {available}"
            )

        # Process date column
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Add decimal return if not present
        if "daily_return_decimal" not in df.columns:
            # Detect if returns are in percentage or decimal
            max_abs_return = df["daily_return"].abs().max()
            if max_abs_return > 1:
                # Likely percentage format
                df["daily_return_decimal"] = df["daily_return"] / 100.0
            else:
                # Already decimal format
                df["daily_return_decimal"] = df["daily_return"]

        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        """
        Load all available backtests.

        Returns:
            Dictionary mapping backtest names to DataFrames.
        """
        backtests = {}
        for name in self.list_backtests():
            try:
                backtests[name] = self.load(name)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        return backtests

    def save_parquet(
        self,
        df: pd.DataFrame,
        name: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save a DataFrame to parquet format.

        Args:
            df: DataFrame to save.
            name: Output filename (without extension).
            output_dir: Output directory (default: self.output_dir).

        Returns:
            Path to the saved file.
        """
        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"{name}.parquet"
        df.to_parquet(filepath, index=False)

        return filepath


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_backtests_from_parquet(
    data_dir: Union[str, Path] = BACKTESTS_DIR,
    names: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load backtests from parquet files.

    Args:
        data_dir: Directory containing parquet files.
        names: Specific backtests to load (default: all).

    Returns:
        Dictionary mapping backtest names to DataFrames.
    """
    data_dir = Path(data_dir)
    backtests = {}

    if names is None:
        parquet_files = list(data_dir.glob("*.parquet"))
        # Exclude known non-backtest files
        exclude = {"combined_portfolio", "vix", "vix_regime_stats", "performance_metrics"}
        names = [f.stem for f in parquet_files if f.stem not in exclude and not f.stem.startswith("benchmark_")]

    for name in names:
        filepath = data_dir / f"{name}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            df["date"] = pd.to_datetime(df["date"])
            backtests[name] = df

    return backtests


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Load and convert backtest files to parquet format."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BACKTESTS_DIR),
        help="Directory containing backtest files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BACKTESTS_DIR),
        help="Directory for output parquet files.",
    )
    parser.add_argument(
        "--backtest",
        type=str,
        default=None,
        help="Specific backtest to load (default: all).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BACKTEST LOADER")
    print("=" * 60)

    loader = BacktestLoader(args.data_dir, args.output_dir)

    # List available backtests
    available = loader.list_backtests()
    print(f"\nAvailable backtests in '{args.data_dir}':")
    for name in available:
        print(f"  - {name}")

    if not available:
        print("\nNo backtest files found.")
        return

    # Load backtests
    backtests_to_load = [args.backtest] if args.backtest else available

    print(f"\nLoading {len(backtests_to_load)} backtest(s)...")

    for name in backtests_to_load:
        try:
            df = loader.load(name)
            filepath = loader.save_parquet(df, name)

            print(f"\n  {name}:")
            print(f"    Rows: {len(df)}")
            print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"    Saved to: {filepath}")

        except Exception as e:
            print(f"\n  Error loading {name}: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
