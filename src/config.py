"""
Centralized configuration module for the backtest analysis project.

This module provides:
- Project paths and directory structure
- Default configuration values
- Dataclasses for type-safe configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DATA_DIR = DATA_DIR / "market"      # Market data (benchmarks, VIX)
BACKTESTS_DIR = DATA_DIR / "backtests"     # Strategy backtest files (.ods, .xlsx)
CACHE_DIR = DATA_DIR / "cache"             # Temporary cache for downloads

# Output directories
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    for directory in [MARKET_DATA_DIR, BACKTESTS_DIR, CACHE_DIR, REPORTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ENUMS
# =============================================================================

class FileFormat(str, Enum):
    """Supported file formats for backtest data."""
    ODS = "ods"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    PARQUET = "parquet"

    @classmethod
    def from_extension(cls, filepath: Path) -> "FileFormat":
        """Get format from file extension."""
        ext = filepath.suffix.lower().lstrip(".")
        try:
            return cls(ext)
        except ValueError:
            raise ValueError(f"Unsupported file format: {ext}")


class PlotFormat(str, Enum):
    """Supported output formats for plots."""
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"


# =============================================================================
# VIX REGIME CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class VixRegimeConfig:
    """Configuration for a VIX regime."""
    name: str
    lower: float
    upper: float
    color: str


DEFAULT_VIX_REGIMES = (
    VixRegimeConfig("Very Low (<12)", 0, 12, "#2ecc71"),
    VixRegimeConfig("Low (12-15)", 12, 15, "#27ae60"),
    VixRegimeConfig("Normal (15-20)", 15, 20, "#3498db"),
    VixRegimeConfig("Elevated (20-25)", 20, 25, "#f39c12"),
    VixRegimeConfig("High (25-30)", 25, 30, "#e74c3c"),
    VixRegimeConfig("Extreme (>30)", 30, float("inf"), "#8e44ad"),
)


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

BENCHMARK_TICKERS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF",
    "VTI": "Total Stock Market ETF",
}

VIX_TICKER = "^VIX"


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for backtest analysis."""

    # Data paths
    market_data_dir: Path = MARKET_DATA_DIR
    backtests_dir: Path = BACKTESTS_DIR
    cache_dir: Path = CACHE_DIR
    output_dir: Path = REPORTS_DIR
    plots_dir: Path = PLOTS_DIR

    # Portfolio configuration
    portfolio_weights: dict[str, float] = field(default_factory=dict)
    initial_capital: float = 1_000_000

    # Benchmark configuration
    benchmark_ticker: str = "SPY"
    benchmark_name: str = "S&P 500"

    # Report settings
    strategy_name: str = "Combined Portfolio"
    report_title: str = "Backtest Analysis Report"

    # Plot settings
    plot_format: PlotFormat = PlotFormat.HTML
    rolling_window: int = 20

    # VIX regimes
    vix_regimes: tuple[VixRegimeConfig, ...] = DEFAULT_VIX_REGIMES

    # Analysis flags
    generate_quantstats_report: bool = True
    generate_vix_analysis: bool = True
    generate_individual_analysis: bool = True
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self):
        """Convert string paths to Path objects if necessary."""
        if isinstance(self.market_data_dir, str):
            self.market_data_dir = Path(self.market_data_dir)
        if isinstance(self.backtests_dir, str):
            self.backtests_dir = Path(self.backtests_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.plots_dir, str):
            self.plots_dir = Path(self.plots_dir)


# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

# Required columns for backtest data
REQUIRED_BACKTEST_COLUMNS = ["date", "equity", "daily_return"]

# Standard column names (for normalization)
COLUMN_ALIASES = {
    "date": ["date", "Date", "DATE", "timestamp", "Timestamp", "time", "Time"],
    "equity": ["equity", "Equity", "EQUITY", "portfolio_value", "value", "Value"],
    "daily_return": [
        "daily_return", "Daily_Return", "return", "Return", "returns", "Returns",
        "daily_return_pct", "pct_return", "daily_pct_return"
    ],
}


def normalize_column_name(col: str) -> Optional[str]:
    """
    Normalize a column name to its standard form.

    Args:
        col: Original column name.

    Returns:
        Standard column name or None if not recognized.
    """
    for standard_name, aliases in COLUMN_ALIASES.items():
        if col in aliases:
            return standard_name
    return None


# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Named colors for specific uses
PLOT_COLORS_NAMED = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#f39c12",
    "info": "#17a2b8",
    "dark": "#343a40",
    "light": "#f8f9fa",
}

# Color palette for data series (sliceable list)
PLOT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

PLOT_TEMPLATE = "plotly_white"
