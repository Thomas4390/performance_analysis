"""
Streamlit Application for Performance Analysis.

A comprehensive interactive dashboard for:
- Uploading and analyzing multiple strategies
- Combining portfolios with custom weights
- VIX regime analysis
- Performance metrics and visualizations

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    INTERMEDIATE_DIR,
    CACHE_DIR,
    BENCHMARK_TICKERS,
    PLOT_COLORS,
    PLOT_TEMPLATE,
    DEFAULT_VIX_REGIMES,
)
from backtest_loader import BacktestLoader
from backtest_combiner import BacktestCombiner, CombinedPortfolio
from market_data import MarketDataDownloader
from vix_analysis import VixRegimeAnalyzer
from report_generator import ReportGenerator

# Warmup Numba JIT compilation at module load
try:
    from metrics_numba import warmup as numba_warmup
    numba_warmup()
except ImportError:
    pass


# =============================================================================
# METRIC DEFINITIONS & TOOLTIPS
# =============================================================================

METRIC_DEFINITIONS = {
    "total_return": {
        "name": "Total Return",
        "short": "Cumulative profit/loss over the entire period.",
        "detailed": """**Total Return** measures the overall percentage gain or loss of your portfolio
        from start to end. A 50% total return means your portfolio grew by half its initial value.

        Formula: (Final Value - Initial Value) / Initial Value""",
    },
    "cagr": {
        "name": "CAGR",
        "short": "Compound Annual Growth Rate - annualized return.",
        "detailed": """**CAGR (Compound Annual Growth Rate)** represents the mean annual growth rate
        assuming profits are reinvested. It smooths out volatility to show what the investment
        would have returned if it grew at a steady rate.

        Formula: (Final Value / Initial Value)^(1/years) - 1

        Example: A 10% CAGR means your investment grows by 10% each year on average.""",
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "short": "Risk-adjusted return (return per unit of risk).",
        "detailed": """**Sharpe Ratio** measures excess return per unit of volatility. It helps
        compare strategies with different risk levels.

        Formula: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

        Interpretation:
        - **< 1.0**: Suboptimal risk-adjusted returns
        - **1.0 - 2.0**: Good risk-adjusted returns
        - **2.0 - 3.0**: Very good
        - **> 3.0**: Excellent (but verify data quality)""",
    },
    "max_drawdown": {
        "name": "Max Drawdown",
        "short": "Largest peak-to-trough decline.",
        "detailed": """**Maximum Drawdown** is the largest percentage drop from a peak to a trough
        before a new peak is achieved. It measures the worst-case scenario for an investor
        who bought at the peak.

        Example: A -30% max drawdown means the portfolio lost 30% from its highest point
        before recovering.

        Important: This metric is crucial for risk management and setting stop-losses.""",
    },
    "volatility": {
        "name": "Volatility",
        "short": "Annualized standard deviation of returns.",
        "detailed": """**Volatility** measures the dispersion of returns around their mean.
        Higher volatility indicates larger price swings and greater uncertainty.

        Formula: Standard Deviation of Daily Returns × √252

        Interpretation:
        - **< 10%**: Low volatility (bonds, stable stocks)
        - **10-20%**: Moderate (diversified equity portfolios)
        - **20-30%**: High (individual stocks, growth sectors)
        - **> 30%**: Very high (crypto, leveraged products)""",
    },
    "alpha": {
        "name": "Alpha",
        "short": "Excess return vs benchmark (skill measure).",
        "detailed": """**Alpha** measures the portfolio's excess return relative to what would be
        expected given its beta (market exposure). Positive alpha indicates the strategy
        adds value beyond market exposure.

        Formula: Portfolio Return - (Risk-Free Rate + Beta × (Market Return - Risk-Free Rate))

        Interpretation:
        - **Positive Alpha**: Strategy outperforms its expected return
        - **Zero Alpha**: Returns match expected market-adjusted return
        - **Negative Alpha**: Strategy underperforms expectations""",
    },
    "beta": {
        "name": "Beta",
        "short": "Sensitivity to market movements.",
        "detailed": """**Beta** measures how much the portfolio moves relative to the benchmark.
        It indicates systematic (market) risk exposure.

        Interpretation:
        - **β = 1.0**: Moves with the market
        - **β > 1.0**: More volatile than market (amplifies moves)
        - **β < 1.0**: Less volatile than market (dampens moves)
        - **β < 0**: Moves opposite to market (rare)

        Example: β = 1.5 means if the market rises 10%, expect ~15% rise.""",
    },
    "sortino_ratio": {
        "name": "Sortino Ratio",
        "short": "Like Sharpe but only penalizes downside volatility.",
        "detailed": """**Sortino Ratio** is similar to Sharpe Ratio but only considers downside
        volatility. It doesn't penalize upside volatility, making it a better measure
        for strategies with asymmetric returns.

        Formula: (Portfolio Return - Risk-Free Rate) / Downside Deviation

        Higher is better. A Sortino of 2.0+ indicates good risk-adjusted returns
        with limited downside.""",
    },
    "calmar_ratio": {
        "name": "Calmar Ratio",
        "short": "CAGR divided by maximum drawdown.",
        "detailed": """**Calmar Ratio** compares annualized return to maximum drawdown,
        measuring return per unit of drawdown risk.

        Formula: CAGR / |Max Drawdown|

        Interpretation:
        - **< 1.0**: Drawdown exceeds annual returns
        - **1.0 - 2.0**: Acceptable
        - **> 2.0**: Good
        - **> 3.0**: Excellent""",
    },
    "win_rate": {
        "name": "Win Rate",
        "short": "Percentage of positive return days.",
        "detailed": """**Win Rate** is the percentage of trading days with positive returns.

        Note: Win rate alone doesn't determine profitability. A 40% win rate can be
        profitable if winners are larger than losers (and vice versa).

        Typical values:
        - **Trend following**: 30-45%
        - **Mean reversion**: 50-65%
        - **Market making**: 55-70%""",
    },
}

VIX_REGIME_DEFINITIONS = {
    "very_low": {
        "name": "Very Low VIX (<12)",
        "description": "Market complacency. Extremely low fear, often seen before corrections.",
        "color": "#2ecc71",
    },
    "low": {
        "name": "Low VIX (12-15)",
        "description": "Calm markets. Low implied volatility, favorable for carry strategies.",
        "color": "#27ae60",
    },
    "normal": {
        "name": "Normal VIX (15-20)",
        "description": "Typical market conditions. Balanced risk/reward environment.",
        "color": "#3498db",
    },
    "elevated": {
        "name": "Elevated VIX (20-25)",
        "description": "Increased uncertainty. Market stress beginning, options more expensive.",
        "color": "#f39c12",
    },
    "high": {
        "name": "High VIX (25-30)",
        "description": "Significant fear. Large market moves expected, hedging costs high.",
        "color": "#e74c3c",
    },
    "extreme": {
        "name": "Extreme VIX (>30)",
        "description": "Crisis mode. Panic selling, liquidity concerns, extreme moves.",
        "color": "#8e44ad",
    },
}


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Performance Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# MODERN CSS STYLING
# =============================================================================

def inject_custom_css():
    """Inject modern CSS styling for the application."""
    st.markdown("""
    <style>
        /* ===== ROOT VARIABLES ===== */
        :root {
            --primary-color: #6366f1;
            --primary-light: #818cf8;
            --primary-dark: #4f46e5;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
        }

        /* ===== GLOBAL STYLES ===== */
        .stApp {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        }

        /* ===== HEADER STYLES ===== */
        .app-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
            padding: 2.5rem 3rem;
            border-radius: var(--radius-lg);
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
        }

        .app-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 400px;
            height: 200%;
            background: radial-gradient(ellipse, rgba(13, 148, 136, 0.3) 0%, transparent 70%);
            transform: rotate(-15deg);
        }

        .app-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        }

        .app-title {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            color: #5eead4 !important;
            margin: 0 0 0.5rem 0 !important;
            letter-spacing: -0.03em;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .app-header h1 {
            color: #5eead4 !important;
        }

        .app-subtitle {
            font-size: 1.1rem;
            color: rgba(255,255,255,0.8);
            margin: 0;
            font-weight: 400;
            letter-spacing: 0.01em;
        }

        /* ===== SIDEBAR STYLES ===== */
        section[data-testid="stSidebar"] {
            background: var(--bg-primary);
            border-right: 1px solid var(--border-color);
        }

        section[data-testid="stSidebar"] .stMarkdown h2 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--primary-color);
            margin-bottom: 1rem;
        }

        section[data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        /* ===== METRIC CARDS ===== */
        div[data-testid="stMetric"] {
            background: var(--bg-primary);
            padding: 1.25rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            box-shadow: var(--shadow-md);
            border-color: var(--primary-light);
        }

        div[data-testid="stMetric"] label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* ===== TAB STYLES ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: var(--radius-sm);
            font-weight: 500;
            font-size: 0.9rem;
            color: var(--text-secondary);
            background: transparent;
            border: none;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary);
            background: rgba(99, 102, 241, 0.1);
        }

        .stTabs [aria-selected="true"] {
            background: var(--bg-primary) !important;
            color: var(--primary-color) !important;
            box-shadow: var(--shadow-sm);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* ===== CARD COMPONENT ===== */
        .card {
            background: var(--bg-primary);
            border-radius: var(--radius-md);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .card-icon {
            width: 40px;
            height: 40px;
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .card-icon.primary { background: rgba(99, 102, 241, 0.1); }
        .card-icon.success { background: rgba(16, 185, 129, 0.1); }
        .card-icon.warning { background: rgba(245, 158, 11, 0.1); }

        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
        }

        .card-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin: 0;
        }

        /* ===== BUTTON STYLES ===== */
        .stButton > button {
            border-radius: var(--radius-sm);
            font-weight: 500;
            padding: 0.625rem 1.25rem;
            transition: all 0.2s ease;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            border: none;
            box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5);
            transform: translateY(-1px);
        }

        /* ===== FILE UPLOADER ===== */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--border-color);
            border-radius: var(--radius-md);
            padding: 1.5rem;
            background: var(--bg-secondary);
            transition: all 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary-light);
            background: rgba(99, 102, 241, 0.02);
        }

        /* ===== DATA FRAME ===== */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        /* ===== STATUS BADGES ===== */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .status-badge.success {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success-color);
        }

        .status-badge.warning {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning-color);
        }

        .status-badge.info {
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary-color);
        }

        /* ===== INFO BOX ===== */
        .info-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(99, 102, 241, 0.02) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: var(--radius-md);
            padding: 1rem 1.25rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }

        .info-box-icon {
            font-size: 1.25rem;
            line-height: 1;
        }

        .info-box-content {
            flex: 1;
        }

        .info-box-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .info-box-text {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin: 0;
        }

        /* ===== STEP INDICATOR ===== */
        .step-indicator {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 2rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }

        .step {
            flex: 1;
            text-align: center;
            padding: 1rem 1.25rem;
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: default;
        }

        .step.active {
            background: linear-gradient(135deg, rgba(13, 148, 136, 0.1) 0%, rgba(30, 58, 95, 0.1) 100%);
            border-color: #0d9488;
            box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15);
        }

        .step.completed {
            background: rgba(16, 185, 129, 0.1);
            border-color: var(--success-color);
        }

        .step-number {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--text-muted);
            color: white;
            font-size: 0.875rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.5rem;
        }

        .step.active .step-number {
            background: linear-gradient(135deg, #0d9488 0%, #1e3a5f 100%);
            box-shadow: 0 2px 8px rgba(13, 148, 136, 0.4);
        }

        .step.completed .step-number {
            background: var(--success-color);
        }

        .step-label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .step.active .step-label {
            color: #0d9488;
        }

        .step.completed .step-label {
            color: var(--success-color);
        }

        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: var(--text-primary);
        }

        /* ===== DIVIDER ===== */
        hr {
            border: none;
            height: 1px;
            background: var(--border-color);
            margin: 1.5rem 0;
        }

        /* ===== FOOTER ===== */
        .app-footer {
            text-align: center;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.8rem;
        }

        .app-footer a {
            color: var(--primary-color);
            text-decoration: none;
        }

        /* ===== QUICK STATS CARD ===== */
        .stat-card {
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 1rem;
            text-align: center;
        }

        .stat-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-value.positive { color: var(--success-color); }
        .stat-value.negative { color: var(--danger-color); }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ===== PLOTLY CONTAINER ===== */
        .stPlotlyChart {
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: var(--bg-primary);
        }

        /* ===== ALERT STYLES ===== */
        .stAlert {
            border-radius: var(--radius-md);
        }

        /* ===== CHECKBOX ===== */
        .stCheckbox label {
            font-weight: 500;
        }

        /* ===== SLIDER ===== */
        .stSlider [data-testid="stTickBar"] {
            background: var(--bg-tertiary);
        }

        /* ===== SELECT BOX ===== */
        .stSelectbox [data-baseweb="select"] {
            border-radius: var(--radius-sm);
        }

        /* ===== DOWNLOAD BUTTON ===== */
        .stDownloadButton > button {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
        }

        .stDownloadButton > button:hover {
            border-color: var(--primary-color);
            background: rgba(99, 102, 241, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "backtests": {},
        "portfolio": None,
        "strategy_returns": None,
        "strategy_returns_full": None,
        "benchmark_returns": None,
        "benchmark_ticker": "SPY",
        "vix": None,
        "analysis_complete": False,
        "quantstats_report": None,
        "use_aligned_data": True,
        # Cached analysis results to avoid recalculation
        "cached_metrics": None,
        "cached_aligned_metrics": None,
        "cached_report_gen": None,
        "cached_comparison_plot": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def series_to_cache_args(series: pd.Series) -> tuple[tuple, tuple]:
    """Convert a pandas Series to cache-friendly tuple arguments."""
    return (
        tuple(series.values.tolist()),
        tuple(series.index.strftime("%Y-%m-%d").tolist()),
    )


@st.cache_data(ttl=3600)
def download_market_data(
    benchmark_ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> tuple[pd.Series, pd.Series]:
    """Download and cache market data."""
    downloader = MarketDataDownloader(cache_dir=CACHE_DIR)
    benchmark = downloader.download_benchmark(benchmark_ticker, start_date, end_date)
    vix_data = downloader.download_vix(start_date, end_date)
    return benchmark.returns, vix_data.data.set_index("date")["vix"]


@st.cache_resource(ttl=300)
def get_vix_analyzer(
    _strategy_returns_hash: str,
    strategy_returns_values: tuple,
    strategy_returns_index: tuple,
    vix_values: tuple,
    vix_index: tuple,
    benchmark_values: tuple | None,
    benchmark_index: tuple | None,
) -> VixRegimeAnalyzer:
    """Create and cache VIX analyzer instance."""
    strategy_returns = pd.Series(list(strategy_returns_values), index=pd.to_datetime(list(strategy_returns_index)))
    vix = pd.Series(list(vix_values), index=pd.to_datetime(list(vix_index)))
    benchmark_returns = None
    if benchmark_values is not None and benchmark_index is not None:
        benchmark_returns = pd.Series(list(benchmark_values), index=pd.to_datetime(list(benchmark_index)))
    return VixRegimeAnalyzer(strategy_returns, vix, benchmark_returns)


_ROLLING_PLOT_VERSION = "v5_shapes_optimized"


def _tuples_to_series(values: tuple, index: tuple) -> pd.Series:
    """Convert cached tuples back to pandas Series efficiently."""
    return pd.Series(np.array(values, dtype=np.float64), index=pd.DatetimeIndex(index))


@st.cache_data(ttl=300)
def create_vix_rolling_plot(
    _data_hash: str,
    _version: str,
    strategy_values: tuple,
    strategy_index: tuple,
    vix_values: tuple,
    vix_index: tuple,
    benchmark_values: tuple | None,
    benchmark_index: tuple | None,
    window: int = 20,
) -> go.Figure:
    """Generate and cache the rolling performance plot with VIX regime backgrounds."""
    strategy_returns = _tuples_to_series(strategy_values, strategy_index)
    vix = _tuples_to_series(vix_values, vix_index)
    benchmark_returns = None
    if benchmark_values is not None and benchmark_index is not None:
        benchmark_returns = _tuples_to_series(benchmark_values, benchmark_index)
    analyzer = VixRegimeAnalyzer(strategy_returns, vix, benchmark_returns)
    return analyzer.plot_rolling_performance(window=window)


def parse_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse an uploaded file to DataFrame."""
    try:
        loader = BacktestLoader()
        df = loader.load_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
        return df
    except Exception as e:
        st.error(f"Error parsing {uploaded_file.name}: {e}")
        return None


def get_workflow_step() -> int:
    """Determine current workflow step based on application state."""
    # Determine step based on actual progress, not tab selection
    if st.session_state.analysis_complete:
        return 4  # Results
    elif st.session_state.backtests:
        return 2  # Configure (data loaded, ready to configure weights)
    return 1  # Upload


def render_metric_with_tooltip(
    label: str,
    value: str,
    metric_key: str,
    delta: str = None,
    delta_color: str = "normal",
):
    """
    Render a metric with an informational tooltip.

    Args:
        label: Display label for the metric.
        value: Formatted value to display.
        metric_key: Key to lookup in METRIC_DEFINITIONS.
        delta: Optional delta value.
        delta_color: Color for delta ("normal", "inverse", "off").
    """
    definition = METRIC_DEFINITIONS.get(metric_key, {})
    help_text = definition.get("short", "")

    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def render_info_expander(metric_key: str):
    """Render a small expandable info section for a metric."""
    definition = METRIC_DEFINITIONS.get(metric_key, {})
    if definition:
        with st.expander(f"ℹ️ About {definition.get('name', metric_key)}", expanded=False):
            st.markdown(definition.get("detailed", definition.get("short", "")))


def render_vix_explanation():
    """Render explanation card for VIX regimes."""
    st.markdown("""
    <div class="info-box" style="margin-bottom: 1.5rem;">
        <div class="info-box-icon">📊</div>
        <div class="info-box-content">
            <div class="info-box-title">Understanding VIX Regimes</div>
            <p class="info-box-text">
                The VIX (Volatility Index) measures market expectations of near-term volatility
                conveyed by S&P 500 option prices. It's often called the "fear gauge" because
                it typically rises during market stress and falls during calm periods.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 VIX Regime Definitions", expanded=False):
        for key, regime in VIX_REGIME_DEFINITIONS.items():
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.5rem; border-radius: 6px; background: {regime['color']}15;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: {regime['color']}; margin-right: 10px;"></div>
                <div>
                    <strong>{regime['name']}</strong><br/>
                    <span style="font-size: 0.85rem; color: var(--text-secondary);">{regime['description']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_workflow_indicator():
    """Render the workflow step indicator.

    Steps are based on actual application state:
    - Step 1 (Upload): Active until data is loaded
    - Step 2 (Configure): Active when data is loaded but analysis not run
    - Step 3 (Analyze): Active when ready to analyze (same as step 2, user clicks Run)
    - Step 4 (Results): Active when analysis is complete
    """
    current_step = get_workflow_step()

    steps = [
        ("1", "Upload Data"),
        ("2", "Configure"),
        ("3", "Analyze"),
        ("4", "Results"),
    ]

    step_items = []
    for i, (num, label) in enumerate(steps, 1):
        # Step is completed if we're past it based on actual progress
        if i < current_step:
            status = "completed"
            icon = "✓"
        elif i == current_step:
            status = "active"
            icon = num
        else:
            status = ""
            icon = num
        step_items.append(f'<div class="step {status}"><div class="step-number">{icon}</div><div class="step-label">{label}</div></div>')

    html = f'<div class="step-indicator">{"".join(step_items)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def create_comparison_plot(
    backtests: dict[str, pd.DataFrame],
    portfolio: CombinedPortfolio,
) -> go.Figure:
    """Create cumulative returns comparison plot."""
    fig = go.Figure()

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
            line=dict(color=PLOT_COLORS[i % len(PLOT_COLORS)], width=2),
            opacity=0.8,
        ))

    combined_cum = portfolio.equity_curve / portfolio.initial_capital
    fig.add_trace(go.Scatter(
        x=combined_cum.index,
        y=combined_cum.values,
        mode="lines",
        name="Combined Portfolio",
        line=dict(color="#1e293b", width=3),
    ))

    fig.update_layout(
        title=dict(text="Strategy Comparison", font=dict(size=16), y=0.98, x=0.5, xanchor="center"),
        xaxis_title="",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template=PLOT_TEMPLATE,
        height=500,
        margin=dict(t=80, b=40, l=60, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    return fig


# =============================================================================
# HEADER COMPONENT
# =============================================================================

def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Performance Analysis Dashboard</h1>
        <p class="app-subtitle">Analyze and combine your trading strategies with VIX regime analysis</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## Configuration")

        # Status summary
        num_backtests = len(st.session_state.backtests)
        if num_backtests > 0:
            st.markdown(f"""
            <div class="status-badge success">
                <span>●</span> {num_backtests} strateg{'ies' if num_backtests > 1 else 'y'} loaded
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge warning">
                <span>○</span> No strategies loaded
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # File upload
        st.markdown("### Upload Files")
        uploaded_files = st.file_uploader(
            "Drag and drop strategy files",
            type=["ods", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Supported: ODS, XLSX, XLS, CSV",
            label_visibility="collapsed",
        )

        if uploaded_files:
            for file in uploaded_files:
                if file.name not in [f"{k}.{file.name.split('.')[-1]}" for k in st.session_state.backtests.keys()]:
                    df = parse_uploaded_file(file)
                    if df is not None:
                        name = Path(file.name).stem
                        st.session_state.backtests[name] = df
                        st.toast(f"Loaded: {name}", icon="✅")

        # Loaded backtests list
        if st.session_state.backtests:
            st.markdown("### Loaded Data")
            for name, df in list(st.session_state.backtests.items()):
                with st.expander(f"📊 {name}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Trading Days")
                        st.markdown(f"**{len(df):,}**")
                    with col2:
                        total_ret = (1 + df["daily_return_decimal"]).prod() - 1
                        st.caption("Total Return")
                        color = "positive" if total_ret >= 0 else "negative"
                        st.markdown(f'<span class="stat-value {color}" style="font-size:1rem">{total_ret:+.1%}</span>', unsafe_allow_html=True)

                    start_date = df['date'].min().strftime('%Y-%m-%d')
                    end_date = df['date'].max().strftime('%Y-%m-%d')
                    st.caption(f"{start_date} → {end_date}")

                    if st.button("Remove", key=f"remove_{name}", type="secondary", use_container_width=True):
                        del st.session_state.backtests[name]
                        st.rerun()

        st.markdown("---")

        # Benchmark selection
        st.markdown("### Benchmark")
        benchmark_ticker = st.selectbox(
            "Select benchmark",
            options=list(BENCHMARK_TICKERS.keys()),
            format_func=lambda x: f"{x} - {BENCHMARK_TICKERS[x]}",
            index=0,
            label_visibility="collapsed",
        )

        # Initial capital
        st.markdown("### Initial Capital")
        initial_capital = st.number_input(
            "Initial capital",
            min_value=1000,
            max_value=100_000_000,
            value=1_000_000,
            step=100_000,
            format="%d",
            label_visibility="collapsed",
        )
        st.caption(f"${initial_capital:,.0f}")

        # Reset button
        st.markdown("---")
        if st.button("Reset All", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return benchmark_ticker, initial_capital


# =============================================================================
# UPLOAD TAB
# =============================================================================

def render_upload_tab():
    """Render the upload and configuration tab."""
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon primary">📁</div>
                <div>
                    <h4 class="card-title">Upload Strategy Files</h4>
                    <p class="card-subtitle">Import your strategy performance data</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        Upload your strategy files using the sidebar. Supported formats:

        | Format | Description |
        |--------|-------------|
        | **CSV** | Comma-separated values |
        | **XLSX** | Microsoft Excel |
        | **XLS** | Legacy Excel |
        | **ODS** | OpenDocument Spreadsheet |

        **Required columns:**
        - `date` - Trading date
        - `daily_return` - Daily return (% or decimal)
        """)

        st.markdown("</div>", unsafe_allow_html=True)

        if not st.session_state.backtests:
            st.markdown("""
            <div class="info-box">
                <div class="info-box-icon">👈</div>
                <div class="info-box-content">
                    <div class="info-box-title">Get Started</div>
                    <p class="info-box-text">Use the sidebar to upload your strategy files.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon success">📈</div>
                <div>
                    <h4 class="card-title">Quick Stats</h4>
                    <p class="card-subtitle">Loaded strategies overview</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.backtests:
            for name, df in st.session_state.backtests.items():
                total_ret = (1 + df["daily_return_decimal"]).prod() - 1
                color_class = "positive" if total_ret >= 0 else "negative"
                st.markdown(f"""
                <div class="stat-card" style="margin-bottom: 0.75rem;">
                    <div class="stat-label">{name}</div>
                    <div class="stat-value {color_class}">{total_ret:+.2%}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p style="color: var(--text-muted); text-align: center; padding: 2rem;">
                No strategies loaded yet
            </p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# WEIGHTS TAB
# =============================================================================

def render_weights_tab():
    """Render the portfolio weights configuration tab."""
    if not st.session_state.backtests:
        st.markdown("""
        <div class="info-box">
            <div class="info-box-icon">⚠️</div>
            <div class="info-box-content">
                <div class="info-box-title">No Data Available</div>
                <p class="info-box-text">Please upload at least one strategy file first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return None

    weights = {}
    backtest_names = list(st.session_state.backtests.keys())
    default_weight = 100.0 / len(backtest_names)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon primary">⚖️</div>
                <div>
                    <h4 class="card-title">Portfolio Allocation</h4>
                    <p class="card-subtitle">Adjust weights for each strategy</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        for name in backtest_names:
            col_name, col_slider = st.columns([1, 3])
            with col_name:
                st.markdown(f"**{name}**")
            with col_slider:
                weights[name] = st.slider(
                    f"Weight for {name}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight,
                    step=1.0,
                    format="%.0f%%",
                    key=f"weight_{name}",
                    label_visibility="collapsed",
                ) / 100.0

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: v / total for k, v in weights.items()}
        else:
            normalized_weights = {k: 1.0 / len(weights) for k in weights}

        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon success">📊</div>
                <div>
                    <h4 class="card-title">Allocation Summary</h4>
                    <p class="card-subtitle">Normalized portfolio weights</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(normalized_weights.keys()),
            values=list(normalized_weights.values()),
            hole=0.5,
            textinfo="percent",
            textposition="outside",
            marker=dict(colors=PLOT_COLORS[:len(normalized_weights)]),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        )])
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=280,
            margin=dict(t=20, b=60, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")

        # Weight details
        for name, weight in normalized_weights.items():
            st.markdown(f"- **{name}**: {weight:.1%}")

        st.markdown("</div>", unsafe_allow_html=True)

    return normalized_weights


# =============================================================================
# ANALYSIS TAB
# =============================================================================

def render_analysis_tab(benchmark_ticker: str, initial_capital: float):
    """Render the analysis results tab."""
    if not st.session_state.backtests:
        st.markdown("""
        <div class="info-box">
            <div class="info-box-icon">⚠️</div>
            <div class="info-box-content">
                <div class="info-box-title">No Data Available</div>
                <p class="info-box-text">Please upload at least one strategy file first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Get weights
    weights = {}
    for name in st.session_state.backtests.keys():
        key = f"weight_{name}"
        if key in st.session_state:
            weights[name] = st.session_state[key] / 100.0
        else:
            weights[name] = 1.0 / len(st.session_state.backtests)

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    # Analysis controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        use_aligned = st.checkbox(
            "Use aligned data (common trading days)",
            value=st.session_state.use_aligned_data,
            help="Metrics calculated only on days where strategy, benchmark, and VIX data all exist.",
        )
        st.session_state.use_aligned_data = use_aligned
    with col3:
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_analysis:
        with st.spinner("Running analysis..."):
            try:
                combiner = BacktestCombiner(st.session_state.backtests, weights)
                portfolio = combiner.combine(initial_capital)
                st.session_state.portfolio = portfolio

                start_date = portfolio.data["date"].min()
                end_date = portfolio.data["date"].max()

                benchmark_returns, vix = download_market_data(benchmark_ticker, start_date, end_date)

                strategy_returns_full = portfolio.returns
                st.session_state.strategy_returns_full = strategy_returns_full

                common_idx = (
                    strategy_returns_full.index
                    .intersection(benchmark_returns.index)
                    .intersection(vix.index)
                )

                st.session_state.strategy_returns = strategy_returns_full.reindex(common_idx)
                st.session_state.benchmark_returns = benchmark_returns.reindex(common_idx)
                st.session_state.vix = vix.reindex(common_idx)

                # Pre-calculate and cache metrics
                use_aligned = st.session_state.use_aligned_data
                strategy_returns = st.session_state.strategy_returns if use_aligned else strategy_returns_full
                aligned_benchmark = benchmark_returns.reindex(common_idx)

                report_gen = ReportGenerator(
                    strategy_returns,
                    aligned_benchmark.reindex(strategy_returns.index) if not use_aligned else aligned_benchmark,
                    strategy_name="Combined Portfolio",
                    benchmark_name=benchmark_ticker,
                )
                st.session_state.cached_metrics = report_gen.calculate_metrics()
                st.session_state.cached_report_gen = report_gen

                if not use_aligned:
                    aligned_report_gen = ReportGenerator(
                        st.session_state.strategy_returns,
                        aligned_benchmark,
                        strategy_name="Combined Portfolio",
                        benchmark_name=benchmark_ticker,
                    )
                    st.session_state.cached_aligned_metrics = aligned_report_gen.calculate_metrics()
                else:
                    st.session_state.cached_aligned_metrics = st.session_state.cached_metrics

                # Cache comparison plot
                st.session_state.cached_comparison_plot = create_comparison_plot(
                    st.session_state.backtests, portfolio
                )

                # Save benchmark ticker for reports
                st.session_state.benchmark_ticker = benchmark_ticker

                st.session_state.analysis_complete = True
                st.toast("Analysis complete!", icon="✅")
                st.rerun()

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    # Display results
    if st.session_state.analysis_complete and st.session_state.portfolio:
        portfolio = st.session_state.portfolio

        if st.session_state.strategy_returns_full is None or st.session_state.strategy_returns is None:
            st.warning("Analysis data not available. Please run analysis again.")
            return

        # Use cached metrics
        metrics = st.session_state.cached_metrics
        aligned_metrics = st.session_state.cached_aligned_metrics
        report_gen = st.session_state.cached_report_gen

        if metrics is None or report_gen is None:
            st.warning("Metrics not calculated. Please run analysis again.")
            return

        # Data info
        use_aligned = st.session_state.use_aligned_data
        full_days = len(st.session_state.strategy_returns_full)
        aligned_days = len(st.session_state.strategy_returns)
        if full_days != aligned_days:
            data_info = f"Using {'aligned' if use_aligned else 'full'} data: {aligned_days if use_aligned else full_days} days"
            st.caption(f"ℹ️ {data_info}")

        # Metrics display with tooltips
        col_header, col_info = st.columns([4, 1])
        with col_header:
            st.markdown("#### Performance Metrics")
        with col_info:
            with st.popover("ℹ️ Metric Guide"):
                st.markdown("**Understanding Performance Metrics**")
                st.markdown("""
                Hover over any metric to see a quick explanation.
                Click below to learn more about each metric.
                """)
                for key in ["total_return", "cagr", "sharpe_ratio", "max_drawdown", "volatility", "alpha", "beta"]:
                    defn = METRIC_DEFINITIONS.get(key, {})
                    with st.expander(defn.get("name", key)):
                        st.markdown(defn.get("detailed", ""))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_with_tooltip("Total Return", f"{metrics.total_return:.2%}", "total_return")
        with m2:
            render_metric_with_tooltip("CAGR", f"{metrics.cagr:.2%}", "cagr")
        with m3:
            render_metric_with_tooltip("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}", "sharpe_ratio")
        with m4:
            render_metric_with_tooltip("Max Drawdown", f"{metrics.max_drawdown:.2%}", "max_drawdown")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            render_metric_with_tooltip("Volatility", f"{metrics.volatility:.2%}", "volatility")
        with m6:
            st.metric("Trading Days", f"{metrics.trading_days:,}", help="Total number of trading days in the analysis period.")
        with m7:
            alpha_val = f"{aligned_metrics.alpha:.2%}" if aligned_metrics.alpha else "N/A"
            render_metric_with_tooltip("Alpha", alpha_val, "alpha")
        with m8:
            beta_val = f"{aligned_metrics.beta:.2f}" if aligned_metrics.beta else "N/A"
            render_metric_with_tooltip("Beta", beta_val, "beta")

        st.markdown("---")

        # Visualization tabs
        viz_tabs = st.tabs(["📈 Returns", "📉 Drawdown", "📊 Distribution", "🔄 Rolling Metrics"])

        with viz_tabs[0]:
            st.caption("💡 Cumulative returns show the growth of $1 invested over time. Compare your strategy vs benchmark.")
            st.plotly_chart(report_gen.plot_cumulative_returns(), use_container_width=True)
            # Use cached comparison plot
            st.caption("💡 Individual strategy comparison shows how each component contributes to the portfolio.")
            if st.session_state.cached_comparison_plot is not None:
                st.plotly_chart(st.session_state.cached_comparison_plot, use_container_width=True)
            else:
                st.plotly_chart(create_comparison_plot(st.session_state.backtests, portfolio), use_container_width=True)

        with viz_tabs[1]:
            st.caption("💡 Drawdown measures peak-to-trough declines. Shallower and shorter drawdowns indicate better risk management.")
            st.plotly_chart(report_gen.plot_drawdown(), use_container_width=True)

        with viz_tabs[2]:
            st.caption("💡 Return distribution shows the frequency of different return levels. A normal distribution centered right of zero is ideal.")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(report_gen.plot_returns_distribution(), use_container_width=True)
            with c2:
                st.plotly_chart(report_gen.plot_monthly_returns_table(), use_container_width=True)

        with viz_tabs[3]:
            st.caption("💡 Rolling metrics show how performance evolves over time. Consistent metrics indicate stable strategy behavior.")
            st.plotly_chart(report_gen.plot_rolling_metrics(), use_container_width=True)


# =============================================================================
# VIX TAB
# =============================================================================

def render_vix_tab():
    """Render the VIX regime analysis tab."""
    if not st.session_state.analysis_complete:
        st.markdown("""
        <div class="info-box">
            <div class="info-box-icon">⚠️</div>
            <div class="info-box-content">
                <div class="info-box-title">Analysis Required</div>
                <p class="info-box-text">Please run the analysis in the Analysis tab first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    strategy_returns = st.session_state.strategy_returns
    benchmark_returns = st.session_state.benchmark_returns
    vix = st.session_state.vix

    if strategy_returns is None or vix is None:
        st.warning("Analysis data not available. Please run analysis again.")
        return

    # Data info
    strategy_returns_full = st.session_state.strategy_returns_full
    if strategy_returns_full is not None:
        full_days = len(strategy_returns_full)
        aligned_days = len(strategy_returns)
        if full_days != aligned_days:
            st.caption(f"ℹ️ VIX analysis uses {aligned_days:,} aligned days (out of {full_days:,} total)")

    # VIX explanation section
    render_vix_explanation()

    # Cache setup
    data_hash = f"{len(strategy_returns)}_{strategy_returns.sum():.6f}"
    strategy_vals, strategy_idx = series_to_cache_args(strategy_returns)
    vix_vals, vix_idx = series_to_cache_args(vix)
    benchmark_vals, benchmark_idx = (
        series_to_cache_args(benchmark_returns)
        if benchmark_returns is not None
        else (None, None)
    )

    vix_analyzer = get_vix_analyzer(
        data_hash, strategy_vals, strategy_idx, vix_vals, vix_idx, benchmark_vals, benchmark_idx
    )

    # Regime stats table
    col_title, col_help = st.columns([4, 1])
    with col_title:
        st.markdown("#### Performance by VIX Regime")
    with col_help:
        with st.popover("ℹ️ Table Guide"):
            st.markdown("""
            **Regime Statistics Columns:**

            - **Days**: Number of trading days in this regime
            - **Ann. Return**: Annualized return during this regime
            - **Ann. Vol**: Annualized volatility during this regime
            - **Sharpe**: Risk-adjusted return for this regime
            - **Win Rate**: Percentage of positive days
            - **Max DD**: Maximum drawdown during this regime

            💡 Compare performance across regimes to understand
            how your strategy behaves in different market conditions.
            """)

    regime_stats = vix_analyzer.calculate_regime_stats()
    display_df = regime_stats.copy()
    display_df.columns = [
        "Regime", "Days", "% Time", "Mean VIX", "Ann. Return (%)",
        "Ann. Vol (%)", "Sharpe", "Win Rate (%)", "Max DD (%)",
        "Total Return (%)", "Bench Return (%)", "Alpha (%)"
    ]
    display_cols = ["Regime", "Days", "Ann. Return (%)", "Ann. Vol (%)", "Sharpe", "Win Rate (%)", "Max DD (%)"]
    st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # Visualizations with explanations
    viz_tabs = st.tabs(["📈 Regime Returns", "📊 Return Distribution", "📉 VIX Timeline", "🔄 Rolling Performance"])

    with viz_tabs[0]:
        st.caption("💡 Cumulative returns with VIX regime backgrounds. Colors indicate market volatility environment.")
        st.plotly_chart(vix_analyzer.plot_cumulative_with_regimes(), use_container_width=True)

    with viz_tabs[1]:
        st.caption("💡 Compare return distributions across different VIX environments.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(vix_analyzer.plot_returns_by_regime(), use_container_width=True)
        with c2:
            st.plotly_chart(vix_analyzer.plot_regime_distribution(), use_container_width=True)

    with viz_tabs[2]:
        st.caption("💡 Visualize the relationship between VIX levels and your strategy returns over time.")
        st.plotly_chart(vix_analyzer.plot_vix_vs_returns(), use_container_width=True)

    with viz_tabs[3]:
        st.caption("💡 Rolling performance metrics overlaid with VIX regime periods.")
        rolling_fig = create_vix_rolling_plot(
            data_hash, _ROLLING_PLOT_VERSION,
            strategy_vals, strategy_idx, vix_vals, vix_idx,
            benchmark_vals, benchmark_idx, window=20
        )
        st.plotly_chart(rolling_fig, use_container_width=True)


# =============================================================================
# EXPORT TAB
# =============================================================================

def render_export_tab():
    """Render the export tab."""
    if not st.session_state.analysis_complete:
        st.markdown("""
        <div class="info-box">
            <div class="info-box-icon">⚠️</div>
            <div class="info-box-content">
                <div class="info-box-title">Analysis Required</div>
                <p class="info-box-text">Please run the analysis first to export results.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon primary">📊</div>
                <div>
                    <h4 class="card-title">Export Data</h4>
                    <p class="card-subtitle">Download raw data and metrics</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.portfolio and st.session_state.strategy_returns is not None:
            # Build export DataFrame with required format
            strategy_returns = st.session_state.strategy_returns
            benchmark_returns = st.session_state.benchmark_returns

            # Create export DataFrame
            export_df = pd.DataFrame({
                "date": strategy_returns.index.strftime("%Y-%m-%d"),
                "strategy_return": strategy_returns.values.astype(float),
                "benchmark_return": benchmark_returns.values.astype(float) if benchmark_returns is not None else None,
                "created_at": datetime.now().isoformat(),
            })

            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="Download Portfolio Data (CSV)",
                data=csv_buffer.getvalue(),
                file_name="combined_portfolio.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.session_state.strategy_returns is not None:
            report_gen = ReportGenerator(
                st.session_state.strategy_returns,
                st.session_state.benchmark_returns,
            )
            metrics = report_gen.calculate_metrics()
            metrics_dict = metrics.to_dict()

            st.download_button(
                label="Download Metrics (JSON)",
                data=pd.Series(metrics_dict).to_json(indent=2),
                file_name="performance_metrics.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon success">📈</div>
                <div>
                    <h4 class="card-title">Generate Reports</h4>
                    <p class="card-subtitle">Create comprehensive analysis reports</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.portfolio is not None:
            include_benchmark = st.checkbox(
                "Include benchmark comparison",
                value=True,
                help="Include SPY or selected benchmark in the report.",
            )

            if st.button("Generate Quantstats Report", type="primary", use_container_width=True):
                try:
                    import tempfile
                    import os
                    import quantstats as qs

                    with st.spinner("Generating report..."):
                        returns = st.session_state.strategy_returns.copy()
                        returns.name = "Strategy"

                        benchmark = None
                        benchmark_name = st.session_state.benchmark_ticker
                        if include_benchmark and st.session_state.benchmark_returns is not None:
                            benchmark = st.session_state.benchmark_returns.copy()
                            benchmark.name = benchmark_name

                        returns.index = pd.to_datetime(returns.index)
                        if benchmark is not None:
                            benchmark.index = pd.to_datetime(benchmark.index)

                        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                            temp_path = f.name

                        qs.reports.html(
                            returns, benchmark=benchmark,
                            output=temp_path, title="Performance Analysis Report",
                            benchmark_title=benchmark_name if include_benchmark else None,
                        )

                        with open(temp_path, 'r', encoding='utf-8') as f:
                            report_html = f.read()
                        os.unlink(temp_path)

                        st.session_state.quantstats_report = report_html
                        st.toast("Report generated!", icon="✅")

                except ImportError:
                    st.error("Quantstats not installed. Run: `pip install quantstats`")
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.session_state.quantstats_report:
                st.download_button(
                    label="Download Quantstats Report (HTML)",
                    data=st.session_state.quantstats_report,
                    file_name="quantstats_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

                with st.expander("Preview Report"):
                    st.components.v1.html(st.session_state.quantstats_report, height=500, scrolling=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("💡 Charts can be exported directly using the camera icon in the plot toolbar.")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    inject_custom_css()

    render_header()
    render_workflow_indicator()

    benchmark_ticker, initial_capital = render_sidebar()

    tabs = st.tabs(["📁 Upload", "⚖️ Weights", "📊 Analysis", "🌡️ VIX Regimes", "💾 Export"])

    with tabs[0]:
        render_upload_tab()
    with tabs[1]:
        render_weights_tab()
    with tabs[2]:
        render_analysis_tab(benchmark_ticker, initial_capital)
    with tabs[3]:
        render_vix_tab()
    with tabs[4]:
        render_export_tab()

    # Footer
    st.markdown("""
    <div class="app-footer">
        Built with Streamlit • Performance Analysis v2.1
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
