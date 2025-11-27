"""
Backtest Analysis Package

A comprehensive toolkit for analyzing QuantConnect backtests with:
- Multi-format file support (ODS, XLSX, CSV, Parquet)
- Portfolio combination with weighted allocations
- VIX regime analysis
- Quantstats reporting
- Interactive visualizations

Project Structure:
    data/
        raw/            - Original backtest files
        intermediate/   - Processed parquet files
        cache/          - Cached market data downloads
    src/
        reports/        - Generated reports
        plots/          - Generated plot files
"""

# Configuration and constants
from .config import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERMEDIATE_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    # Enums
    FileFormat,
    PlotFormat,
    # Dataclasses
    VixRegimeConfig,
    AnalysisConfig,
    # Constants
    BENCHMARK_TICKERS,
    VIX_TICKER,
    DEFAULT_VIX_REGIMES,
    REQUIRED_BACKTEST_COLUMNS,
    COLUMN_ALIASES,
    PLOT_COLORS,
    PLOT_COLORS_NAMED,
    PLOT_TEMPLATE,
    # Functions
    normalize_column_name,
)

# Backtest loading
from .backtest_loader import (
    BacktestLoader,
    FileReader,
    load_backtests_from_parquet,
)

# Portfolio combination
from .backtest_combiner import (
    BacktestCombiner,
    CombinedPortfolio,
    PortfolioMetrics,
    load_and_combine,
)

# Market data
from .market_data import (
    MarketDataDownloader,
    MarketData,
    load_market_data,
)

# VIX analysis
from .vix_analysis import (
    VixRegimeAnalyzer,
    VixRegimeStats,
    load_data_for_vix_analysis,
)

# Report generation
from .report_generator import (
    ReportGenerator,
    PerformanceMetrics,
    load_returns_for_report,
)


__version__ = "2.0.0"

__all__ = [
    # Version
    "__version__",
    # Configuration
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERMEDIATE_DIR",
    "CACHE_DIR",
    "REPORTS_DIR",
    "PLOTS_DIR",
    "FileFormat",
    "PlotFormat",
    "VixRegimeConfig",
    "AnalysisConfig",
    "BENCHMARK_TICKERS",
    "VIX_TICKER",
    "DEFAULT_VIX_REGIMES",
    "REQUIRED_BACKTEST_COLUMNS",
    "COLUMN_ALIASES",
    "PLOT_COLORS",
    "PLOT_COLORS_NAMED",
    "PLOT_TEMPLATE",
    "normalize_column_name",
    # Backtest loading
    "BacktestLoader",
    "FileReader",
    "load_backtests_from_parquet",
    # Portfolio combination
    "BacktestCombiner",
    "CombinedPortfolio",
    "PortfolioMetrics",
    "load_and_combine",
    # Market data
    "MarketDataDownloader",
    "MarketData",
    "load_market_data",
    # VIX analysis
    "VixRegimeAnalyzer",
    "VixRegimeStats",
    "load_data_for_vix_analysis",
    # Report generation
    "ReportGenerator",
    "PerformanceMetrics",
    "load_returns_for_report",
]
