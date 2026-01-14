"""
Module for correlation and dependency analysis between strategies.

This module provides:
- ARMA-GARCH filtering to extract standardized residuals (epsilon_t)
- Probability Integral Transform (PIT) for copula analysis
- Rolling correlation analysis
- Dynamic Conditional Correlation (DCC) modeling

The methodology follows the standard approach in financial econometrics:
1. Filter returns with ARMA to remove autocorrelation -> a_t = r_t - mu_t
2. Filter shocks with GARCH to remove heteroskedasticity -> epsilon_t = a_t / sigma_t
3. Apply PIT to get uniform margins -> u_t = F(epsilon_t)
4. Fit copulas on (u1_t, u2_t, ..., un_t)

Usage:
    analyzer = CorrelationAnalyzer(returns_dict)
    residuals = analyzer.get_standardized_residuals()
    uniforms = analyzer.get_pit_transformed()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from config import PLOT_TEMPLATE, PLOT_COLORS

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# =============================================================================

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True, cache=True)
def _rolling_correlation_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    Numba-optimized rolling Pearson correlation.

    Args:
        x: First time series
        y: Second time series
        window: Rolling window size

    Returns:
        Array of rolling correlations
    """
    n = len(x)
    result = np.full(n, np.nan)

    for i in prange(window - 1, n):
        start = i - window + 1
        x_win = x[start:i+1]
        y_win = y[start:i+1]

        # Calculate means
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)

        # Calculate correlation
        num = 0.0
        den_x = 0.0
        den_y = 0.0

        for j in range(window):
            x_diff = x_win[j] - x_mean
            y_diff = y_win[j] - y_mean
            num += x_diff * y_diff
            den_x += x_diff * x_diff
            den_y += y_diff * y_diff

        if den_x > 0 and den_y > 0:
            result[i] = num / np.sqrt(den_x * den_y)
        else:
            result[i] = np.nan

    return result


@jit(nopython=True, cache=True)
def _rolling_correlation_multi_window(x: np.ndarray, y: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Calculate rolling correlation for multiple window sizes.

    Args:
        x: First time series
        y: Second time series
        windows: Array of window sizes

    Returns:
        2D array of shape (len(windows), len(x))
    """
    n = len(x)
    n_windows = len(windows)
    result = np.full((n_windows, n), np.nan)

    for w_idx in range(n_windows):
        window = windows[w_idx]
        for i in range(window - 1, n):
            start = i - window + 1
            x_win = x[start:i+1]
            y_win = y[start:i+1]

            x_mean = np.mean(x_win)
            y_mean = np.mean(y_win)

            num = 0.0
            den_x = 0.0
            den_y = 0.0

            for j in range(window):
                x_diff = x_win[j] - x_mean
                y_diff = y_win[j] - y_mean
                num += x_diff * y_diff
                den_x += x_diff * x_diff
                den_y += y_diff * y_diff

            if den_x > 0 and den_y > 0:
                result[w_idx, i] = num / np.sqrt(den_x * den_y)

    return result


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GARCHResult:
    """Results from GARCH model fitting."""
    strategy_name: str
    returns: pd.Series
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    model_type: str
    params: dict
    aic: float
    bic: float
    ljung_box_resid: float  # p-value for Ljung-Box test on residuals
    ljung_box_resid_sq: float  # p-value for Ljung-Box test on squared residuals

    def is_valid(self) -> bool:
        """Check if the model passes diagnostic tests."""
        return self.ljung_box_resid > 0.05 and self.ljung_box_resid_sq > 0.05


@dataclass
class CorrelationStats:
    """Statistics for correlation analysis."""
    strategy_pair: tuple[str, str]
    pearson: float
    spearman: float
    kendall: float
    lower_tail_dep: float
    upper_tail_dep: float


@dataclass
class RollingCorrelationResult:
    """Results from rolling correlation analysis."""
    dates: pd.DatetimeIndex
    correlations: dict[tuple[str, str], pd.Series]
    mean_correlation: dict[tuple[str, str], float]
    std_correlation: dict[tuple[str, str], float]
    min_correlation: dict[tuple[str, str], float]
    max_correlation: dict[tuple[str, str], float]


@dataclass
class PITValidationResult:
    """Comprehensive validation results for PIT transformation."""
    strategy_name: str
    method: str
    n_obs: int

    # Kolmogorov-Smirnov test
    ks_statistic: float
    ks_pvalue: float
    ks_passed: bool

    # Anderson-Darling test
    ad_statistic: float
    ad_critical_5pct: float
    ad_passed: bool

    # Cramér-von Mises test
    cvm_statistic: float
    cvm_pvalue: float
    cvm_passed: bool

    # Chi-squared test for uniformity (binned)
    chi2_statistic: float
    chi2_pvalue: float
    chi2_passed: bool

    # Summary
    all_tests_passed: bool

    def get_summary_dict(self) -> dict:
        """Return a summary dictionary for display."""
        return {
            "Strategy": self.strategy_name,
            "Method": self.method,
            "N": self.n_obs,
            "KS p-val": self.ks_pvalue,
            "AD stat": self.ad_statistic,
            "CvM p-val": self.cvm_pvalue,
            "χ² p-val": self.chi2_pvalue,
            "Uniform": "Yes" if self.all_tests_passed else "No",
        }


@dataclass
class StepValidationSummary:
    """Summary of validation at each step of the analysis pipeline."""
    garch_valid: dict[str, bool]  # strategy -> valid
    pit_valid: dict[str, bool]  # strategy -> valid
    copula_gof_pvalue: Optional[float] = None
    overall_valid: bool = False


# =============================================================================
# GARCH FILTERING
# =============================================================================

class GARCHFilter:
    """
    GARCH filter for extracting standardized residuals from returns.

    This implements the standard ARMA-GARCH filtering procedure:
    1. Fit ARMA(p,q) to capture autocorrelation in returns
    2. Fit GARCH(1,1) to capture volatility clustering
    3. Extract standardized residuals: epsilon_t = a_t / sigma_t

    Example:
        filter = GARCHFilter(returns)
        result = filter.fit()
        residuals = result.standardized_residuals
    """

    def __init__(
        self,
        returns: pd.Series,
        strategy_name: str = "Strategy",
        arma_order: tuple[int, int] = (1, 0),
        garch_order: tuple[int, int] = (1, 1),
        distribution: Literal["normal", "t", "skewt"] = "t",
    ):
        """
        Initialize the GARCH filter.

        Args:
            returns: Daily returns series (decimal format).
            strategy_name: Name of the strategy for labeling.
            arma_order: (p, q) order for ARMA mean model.
            garch_order: (p, q) order for GARCH volatility model.
            distribution: Distribution for innovations ("normal", "t", "skewt").
        """
        self.returns = returns.dropna()
        self.strategy_name = strategy_name
        self.arma_order = arma_order
        self.garch_order = garch_order
        self.distribution = distribution
        self._result: Optional[GARCHResult] = None

    def fit(self) -> GARCHResult:
        """
        Fit the ARMA-GARCH model and extract standardized residuals.

        Returns:
            GARCHResult with model diagnostics and residuals.
        """
        try:
            from arch import arch_model
            return self._fit_arch()
        except ImportError:
            # Fallback to simple EWMA volatility if arch not available
            return self._fit_ewma_fallback()

    def _fit_arch(self) -> GARCHResult:
        """Fit using the arch library."""
        from arch import arch_model
        from statsmodels.stats.diagnostic import acorr_ljungbox

        returns_scaled = self.returns * 100  # Scale for numerical stability

        # Build ARMA-GARCH model
        model = arch_model(
            returns_scaled,
            mean="ARX" if self.arma_order != (0, 0) else "Constant",
            lags=self.arma_order[0] if self.arma_order[0] > 0 else None,
            vol="GARCH",
            p=self.garch_order[0],
            q=self.garch_order[1],
            dist=self.distribution,
        )

        # Fit with robust standard errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        # Extract conditional volatility and residuals
        cond_vol = result.conditional_volatility / 100  # Scale back
        std_resid = result.std_resid

        # Convert to Series with proper index
        cond_vol_series = pd.Series(cond_vol.values, index=self.returns.index, name="cond_vol")
        std_resid_series = pd.Series(std_resid.values, index=self.returns.index, name="std_resid")

        # Ljung-Box tests
        lb_resid = acorr_ljungbox(std_resid.dropna(), lags=[10], return_df=True)
        lb_resid_sq = acorr_ljungbox(std_resid.dropna() ** 2, lags=[10], return_df=True)

        # Extract parameters
        params = {
            "omega": result.params.get("omega", 0) / 10000,  # Scale back
            "alpha": result.params.get("alpha[1]", 0),
            "beta": result.params.get("beta[1]", 0),
        }
        if self.distribution == "t":
            params["nu"] = result.params.get("nu", 30)

        self._result = GARCHResult(
            strategy_name=self.strategy_name,
            returns=self.returns,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            model_type=f"ARMA{self.arma_order}-GARCH{self.garch_order}",
            params=params,
            aic=result.aic,
            bic=result.bic,
            ljung_box_resid=float(lb_resid["lb_pvalue"].iloc[0]),
            ljung_box_resid_sq=float(lb_resid_sq["lb_pvalue"].iloc[0]),
        )

        return self._result

    def _fit_ewma_fallback(self) -> GARCHResult:
        """Fallback using EWMA volatility when arch is not available."""
        from statsmodels.stats.diagnostic import acorr_ljungbox

        # Simple EWMA volatility (like RiskMetrics)
        lambda_param = 0.94
        returns_sq = self.returns ** 2

        # Initialize with sample variance
        var_t = np.zeros(len(self.returns))
        var_t[0] = returns_sq.iloc[:20].mean() if len(returns_sq) > 20 else returns_sq.iloc[0]

        # EWMA recursion
        for t in range(1, len(self.returns)):
            var_t[t] = lambda_param * var_t[t-1] + (1 - lambda_param) * returns_sq.iloc[t-1]

        cond_vol = np.sqrt(var_t)
        std_resid = self.returns.values / np.maximum(cond_vol, 1e-8)

        cond_vol_series = pd.Series(cond_vol, index=self.returns.index, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=self.returns.index, name="std_resid")

        # Ljung-Box tests
        lb_resid = acorr_ljungbox(std_resid[~np.isnan(std_resid)], lags=[10], return_df=True)
        lb_resid_sq = acorr_ljungbox(std_resid[~np.isnan(std_resid)] ** 2, lags=[10], return_df=True)

        self._result = GARCHResult(
            strategy_name=self.strategy_name,
            returns=self.returns,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            model_type="EWMA(0.94)",
            params={"lambda": lambda_param},
            aic=np.nan,
            bic=np.nan,
            ljung_box_resid=float(lb_resid["lb_pvalue"].iloc[0]),
            ljung_box_resid_sq=float(lb_resid_sq["lb_pvalue"].iloc[0]),
        )

        return self._result

    @property
    def result(self) -> Optional[GARCHResult]:
        """Get the fitted result."""
        return self._result


# =============================================================================
# PROBABILITY INTEGRAL TRANSFORM (PIT)
# =============================================================================

class PITTransformer:
    """
    Probability Integral Transform for standardized residuals.

    Transforms standardized residuals to uniform [0, 1] margins
    for copula analysis.

    Methods:
    - Empirical: Uses rank transform (non-parametric)
    - Parametric: Fits a distribution and applies its CDF

    Example:
        pit = PITTransformer(residuals)
        uniforms = pit.transform(method="empirical")
    """

    def __init__(self, residuals: pd.Series):
        """
        Initialize the PIT transformer.

        Args:
            residuals: Standardized residuals from GARCH filtering.
        """
        self.residuals = residuals.dropna()
        self._uniforms: Optional[pd.Series] = None
        self._method: Optional[str] = None
        self._distribution_params: Optional[dict] = None

    def transform(
        self,
        method: Literal["empirical", "normal", "t"] = "empirical",
    ) -> pd.Series:
        """
        Apply PIT to get uniform margins.

        Args:
            method: Transformation method:
                - "empirical": Rank-based (pseudo-observations)
                - "normal": Fit normal CDF
                - "t": Fit Student-t CDF

        Returns:
            Series of uniform values in (0, 1).
        """
        self._method = method
        n = len(self.residuals)

        if method == "empirical":
            # Pseudo-observations (avoid 0 and 1)
            ranks = self.residuals.rank()
            uniforms = ranks / (n + 1)

        elif method == "normal":
            # Fit normal and apply CDF
            mu, sigma = self.residuals.mean(), self.residuals.std()
            uniforms = stats.norm.cdf(self.residuals, loc=mu, scale=sigma)
            self._distribution_params = {"mu": mu, "sigma": sigma}

        elif method == "t":
            # Fit Student-t and apply CDF
            df, loc, scale = stats.t.fit(self.residuals)
            uniforms = stats.t.cdf(self.residuals, df, loc=loc, scale=scale)
            self._distribution_params = {"df": df, "loc": loc, "scale": scale}

        else:
            raise ValueError(f"Unknown method: {method}")

        # Clip to avoid exact 0 and 1 (problematic for copulas)
        uniforms = np.clip(uniforms, 1e-6, 1 - 1e-6)

        self._uniforms = pd.Series(uniforms, index=self.residuals.index, name="uniform")
        return self._uniforms

    def validate_uniformity(self, strategy_name: str = "Strategy") -> PITValidationResult:
        """
        Comprehensive validation that the transformed data is uniform.

        Performs multiple statistical tests:
        1. Kolmogorov-Smirnov test
        2. Anderson-Darling test
        3. Cramér-von Mises test
        4. Chi-squared goodness-of-fit test

        Args:
            strategy_name: Name for labeling results.

        Returns:
            PITValidationResult with all test statistics.
        """
        if self._uniforms is None:
            raise ValueError("Must call transform() first")

        u = self._uniforms.values
        n = len(u)

        # 1. Kolmogorov-Smirnov test for uniformity
        ks_stat, ks_pval = stats.kstest(u, "uniform")

        # 2. Anderson-Darling test for uniform(0,1)
        # Manual calculation since scipy.stats.anderson doesn't support uniform
        u_sorted = np.sort(u)
        # Clip to avoid log(0)
        u_sorted = np.clip(u_sorted, 1e-10, 1 - 1e-10)
        i_vals = np.arange(1, n + 1)
        # AD statistic for uniform(0,1)
        ad_stat = -n - (1.0/n) * np.sum(
            (2 * i_vals - 1) * (np.log(u_sorted) + np.log(1 - u_sorted[::-1]))
        )
        # Critical value at 5% for uniform(0,1) is approximately 2.492
        ad_critical_5pct = 2.492

        # 3. Cramér-von Mises test
        try:
            cvm_result = stats.cramervonmises(u, "uniform")
            cvm_stat = cvm_result.statistic
            cvm_pval = cvm_result.pvalue
        except AttributeError:
            # Older scipy versions - manual calculation
            u_sorted = np.sort(u)
            i_vals = np.arange(1, n + 1)
            cvm_stat = 1/(12*n) + np.sum((u_sorted - (2*i_vals - 1)/(2*n))**2)
            # Approximate p-value
            cvm_pval = 1.0 if cvm_stat < 0.461 else 0.0  # Simplified

        # 4. Chi-squared test with 10 bins
        n_bins = 10
        observed, _ = np.histogram(u, bins=n_bins, range=(0, 1))
        expected = np.full(n_bins, n / n_bins)
        chi2_stat, chi2_pval = stats.chisquare(observed, expected)

        # Determine if tests pass (alpha = 0.05)
        ks_passed = ks_pval > 0.05
        ad_passed = ad_stat < ad_critical_5pct
        cvm_passed = cvm_pval > 0.05
        chi2_passed = chi2_pval > 0.05

        # All tests must pass for overall validity
        all_passed = ks_passed and ad_passed and cvm_passed and chi2_passed

        return PITValidationResult(
            strategy_name=strategy_name,
            method=self._method,
            n_obs=n,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ks_passed=ks_passed,
            ad_statistic=ad_stat,
            ad_critical_5pct=ad_critical_5pct,
            ad_passed=ad_passed,
            cvm_statistic=cvm_stat,
            cvm_pvalue=cvm_pval,
            cvm_passed=cvm_passed,
            chi2_statistic=chi2_stat,
            chi2_pvalue=chi2_pval,
            chi2_passed=chi2_passed,
            all_tests_passed=all_passed,
        )

    def validate_uniformity_simple(self) -> dict:
        """
        Simple validation returning a dictionary (for backward compatibility).

        Returns:
            Dictionary with test statistics and p-values.
        """
        result = self.validate_uniformity()
        return {
            "ks_statistic": result.ks_statistic,
            "ks_pvalue": result.ks_pvalue,
            "ad_statistic": result.ad_statistic,
            "cvm_statistic": result.cvm_statistic,
            "cvm_pvalue": result.cvm_pvalue,
            "chi2_pvalue": result.chi2_pvalue,
            "is_uniform": result.all_tests_passed,
        }

    @property
    def uniforms(self) -> Optional[pd.Series]:
        """Get the uniform transformed values."""
        return self._uniforms


# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    """
    Comprehensive correlation and dependency analyzer for multiple strategies.

    This class provides:
    - GARCH filtering for all strategies
    - PIT transformation
    - Static and rolling correlations
    - Tail dependence estimation

    Example:
        returns_dict = {"strat1": returns1, "strat2": returns2}
        analyzer = CorrelationAnalyzer(returns_dict)
        analyzer.fit_garch()
        corr_matrix = analyzer.get_correlation_matrix()
    """

    def __init__(
        self,
        returns_dict: dict[str, pd.Series],
        garch_order: tuple[int, int] = (1, 1),
        arma_order: tuple[int, int] = (1, 0),
        distribution: str = "t",
    ):
        """
        Initialize the correlation analyzer.

        Args:
            returns_dict: Dictionary mapping strategy names to return series.
            garch_order: GARCH(p, q) order for all strategies.
            arma_order: ARMA(p, q) order for mean model.
            distribution: Innovation distribution ("normal", "t", "skewt").
        """
        self.strategy_names = list(returns_dict.keys())

        # Align all returns to common dates
        common_idx = returns_dict[self.strategy_names[0]].index
        for name in self.strategy_names[1:]:
            common_idx = common_idx.intersection(returns_dict[name].index)

        self.returns = pd.DataFrame({
            name: returns_dict[name].reindex(common_idx)
            for name in self.strategy_names
        })

        self.garch_order = garch_order
        self.arma_order = arma_order
        self.distribution = distribution

        # Results storage
        self._garch_results: dict[str, GARCHResult] = {}
        self._standardized_residuals: Optional[pd.DataFrame] = None
        self._pit_uniforms: Optional[pd.DataFrame] = None

    def fit_garch(self) -> dict[str, GARCHResult]:
        """
        Fit GARCH models to all strategies.

        Returns:
            Dictionary of GARCHResult for each strategy.
        """
        for name in self.strategy_names:
            filter = GARCHFilter(
                returns=self.returns[name],
                strategy_name=name,
                arma_order=self.arma_order,
                garch_order=self.garch_order,
                distribution=self.distribution,
            )
            self._garch_results[name] = filter.fit()

        # Build standardized residuals DataFrame
        self._standardized_residuals = pd.DataFrame({
            name: result.standardized_residuals
            for name, result in self._garch_results.items()
        })

        return self._garch_results

    def apply_pit(self, method: str = "empirical") -> pd.DataFrame:
        """
        Apply PIT transformation to all standardized residuals.

        Args:
            method: PIT method ("empirical", "normal", "t").

        Returns:
            DataFrame of uniform margins for all strategies.
        """
        if self._standardized_residuals is None:
            self.fit_garch()

        pit_data = {}
        self._pit_transformers = {}  # Store for validation
        for name in self.strategy_names:
            pit = PITTransformer(self._standardized_residuals[name])
            pit_data[name] = pit.transform(method=method)
            self._pit_transformers[name] = pit

        self._pit_uniforms = pd.DataFrame(pit_data)
        return self._pit_uniforms

    def validate_pit(self) -> dict[str, PITValidationResult]:
        """
        Validate PIT transformation for all strategies.

        Returns:
            Dictionary mapping strategy names to PITValidationResult.
        """
        if self._pit_uniforms is None:
            self.apply_pit()

        results = {}
        for name in self.strategy_names:
            if hasattr(self, '_pit_transformers') and name in self._pit_transformers:
                results[name] = self._pit_transformers[name].validate_uniformity(strategy_name=name)
            else:
                # Reconstruct transformer
                pit = PITTransformer(self._standardized_residuals[name])
                pit._uniforms = self._pit_uniforms[name]
                pit._method = "empirical"
                results[name] = pit.validate_uniformity(strategy_name=name)

        return results

    def get_validation_summary(self) -> StepValidationSummary:
        """
        Get a summary of validation at each step.

        Returns:
            StepValidationSummary with validation status for each step.
        """
        # GARCH validation
        if not self._garch_results:
            self.fit_garch()

        garch_valid = {
            name: result.is_valid()
            for name, result in self._garch_results.items()
        }

        # PIT validation
        pit_results = self.validate_pit()
        pit_valid = {
            name: result.all_tests_passed
            for name, result in pit_results.items()
        }

        # Overall
        overall = all(garch_valid.values()) and all(pit_valid.values())

        return StepValidationSummary(
            garch_valid=garch_valid,
            pit_valid=pit_valid,
            overall_valid=overall,
        )

    def get_standardized_residuals(self) -> pd.DataFrame:
        """Get standardized residuals for all strategies."""
        if self._standardized_residuals is None:
            self.fit_garch()
        return self._standardized_residuals.copy()

    def get_pit_uniforms(self) -> pd.DataFrame:
        """Get PIT-transformed uniform margins."""
        if self._pit_uniforms is None:
            self.apply_pit()
        return self._pit_uniforms.copy()

    def get_correlation_matrix(
        self,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        use_residuals: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            method: Correlation method.
            use_residuals: If True, use standardized residuals; else raw returns.

        Returns:
            Correlation matrix as DataFrame.
        """
        data = self._standardized_residuals if use_residuals else self.returns
        if data is None:
            self.fit_garch()
            data = self._standardized_residuals

        return data.corr(method=method)

    def get_rolling_correlation(
        self,
        window: int = 60,
        method: str = "pearson",
        use_residuals: bool = True,
        use_numba: bool = True,
    ) -> RollingCorrelationResult:
        """
        Calculate rolling pairwise correlations.

        Uses Numba-optimized calculation when available for Pearson correlation.

        Args:
            window: Rolling window size in days.
            method: Correlation method.
            use_residuals: If True, use standardized residuals.
            use_numba: If True, use Numba-optimized calculation (Pearson only).

        Returns:
            RollingCorrelationResult with correlation time series.
        """
        data = self._standardized_residuals if use_residuals else self.returns
        if data is None:
            self.fit_garch()
            data = self._standardized_residuals

        correlations = {}
        mean_corr = {}
        std_corr = {}
        min_corr = {}
        max_corr = {}

        for i, name1 in enumerate(self.strategy_names):
            for name2 in self.strategy_names[i+1:]:
                pair = (name1, name2)

                if method == "pearson" and use_numba and NUMBA_AVAILABLE:
                    # Use Numba-optimized calculation
                    x = data[name1].values.astype(np.float64)
                    y = data[name2].values.astype(np.float64)
                    corr_values = _rolling_correlation_numba(x, y, window)
                    rolling_corr = pd.Series(corr_values, index=data.index)
                elif method == "pearson":
                    rolling_corr = data[name1].rolling(window).corr(data[name2])
                elif method == "spearman":
                    # Spearman via rank transform
                    rank1 = data[name1].rolling(window).apply(
                        lambda x: stats.rankdata(x)[-1] / len(x), raw=True
                    )
                    rank2 = data[name2].rolling(window).apply(
                        lambda x: stats.rankdata(x)[-1] / len(x), raw=True
                    )
                    rolling_corr = rank1.rolling(window).corr(rank2)
                else:  # kendall
                    def kendall_corr(x, y):
                        return stats.kendalltau(x, y)[0]

                    rolling_corr = data[name1].rolling(window).apply(
                        lambda x: kendall_corr(
                            x, data[name2].loc[x.index]
                        ) if len(x) == window else np.nan,
                        raw=False
                    )

                correlations[pair] = rolling_corr.dropna()
                mean_corr[pair] = correlations[pair].mean()
                std_corr[pair] = correlations[pair].std()
                min_corr[pair] = correlations[pair].min()
                max_corr[pair] = correlations[pair].max()

        return RollingCorrelationResult(
            dates=data.index,
            correlations=correlations,
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
        )

    def get_rolling_correlation_multiwindow(
        self,
        windows: list[int] = None,
        use_residuals: bool = True,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """
        Calculate rolling correlations for multiple window sizes.

        Uses Numba-optimized calculation for efficiency.

        Args:
            windows: List of window sizes (default: [20, 40, 60, 120, 252]).
            use_residuals: If True, use standardized residuals.

        Returns:
            Dictionary mapping pairs to DataFrames with columns for each window.
        """
        if windows is None:
            windows = [20, 40, 60, 120, 252]

        data = self._standardized_residuals if use_residuals else self.returns
        if data is None:
            self.fit_garch()
            data = self._standardized_residuals

        results = {}
        windows_arr = np.array(windows, dtype=np.int64)

        for i, name1 in enumerate(self.strategy_names):
            for name2 in self.strategy_names[i+1:]:
                pair = (name1, name2)

                x = data[name1].values.astype(np.float64)
                y = data[name2].values.astype(np.float64)

                if NUMBA_AVAILABLE:
                    corr_matrix = _rolling_correlation_multi_window(x, y, windows_arr)
                else:
                    # Fallback
                    corr_matrix = np.zeros((len(windows), len(x)))
                    for w_idx, w in enumerate(windows):
                        corr_matrix[w_idx] = data[name1].rolling(w).corr(data[name2]).values

                results[pair] = pd.DataFrame(
                    corr_matrix.T,
                    index=data.index,
                    columns=[f"{w}d" for w in windows],
                )

        return results

    def estimate_tail_dependence(
        self,
        quantile: float = 0.05,
    ) -> dict[tuple[str, str], dict]:
        """
        Estimate tail dependence coefficients non-parametrically.

        Uses the empirical tail dependence estimator:
        lambda_L = P(U1 <= q | U2 <= q)
        lambda_U = P(U1 > 1-q | U2 > 1-q)

        Args:
            quantile: Quantile threshold for tail (default 5%).

        Returns:
            Dictionary of tail dependence for each pair.
        """
        if self._pit_uniforms is None:
            self.apply_pit()

        results = {}

        for i, name1 in enumerate(self.strategy_names):
            for name2 in self.strategy_names[i+1:]:
                u1 = self._pit_uniforms[name1].values
                u2 = self._pit_uniforms[name2].values

                # Lower tail dependence
                lower_mask = (u1 <= quantile) & (u2 <= quantile)
                lambda_L = lower_mask.sum() / (u1 <= quantile).sum() if (u1 <= quantile).sum() > 0 else 0

                # Upper tail dependence
                upper_mask = (u1 >= 1 - quantile) & (u2 >= 1 - quantile)
                lambda_U = upper_mask.sum() / (u1 >= 1 - quantile).sum() if (u1 >= 1 - quantile).sum() > 0 else 0

                results[(name1, name2)] = {
                    "lower_tail": lambda_L,
                    "upper_tail": lambda_U,
                    "asymmetry": lambda_L - lambda_U,
                }

        return results

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_correlation_matrix(
        self,
        method: str = "pearson",
        use_residuals: bool = True,
    ) -> go.Figure:
        """Plot correlation matrix heatmap."""
        corr = self.get_correlation_matrix(method=method, use_residuals=use_residuals)

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation"),
        ))

        title = f"Correlation Matrix ({method.capitalize()})"
        if use_residuals:
            title += " - Standardized Residuals"

        fig.update_layout(
            title=title,
            xaxis_title="Strategy",
            yaxis_title="Strategy",
            template=PLOT_TEMPLATE,
            height=500,
            width=600,
        )

        return fig

    def plot_rolling_correlation(
        self,
        window: int = 60,
        method: str = "pearson",
        use_residuals: bool = True,
    ) -> go.Figure:
        """Plot rolling correlations over time."""
        result = self.get_rolling_correlation(
            window=window,
            method=method,
            use_residuals=use_residuals,
        )

        fig = go.Figure()

        colors = PLOT_COLORS[:len(result.correlations)]

        for (pair, corr_series), color in zip(result.correlations.items(), colors):
            fig.add_trace(go.Scatter(
                x=corr_series.index,
                y=corr_series.values,
                mode="lines",
                name=f"{pair[0]} vs {pair[1]}",
                line=dict(color=color, width=1.5),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        title = f"Rolling Correlation ({window}-Day Window, {method.capitalize()})"

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Correlation",
            template=PLOT_TEMPLATE,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
        )

        return fig

    def plot_rolling_correlation_interactive(
        self,
        windows: list[int] = None,
        use_residuals: bool = True,
    ) -> go.Figure:
        """
        Create interactive rolling correlation plot with window size slider.

        Uses Numba-optimized calculations for smooth slider interaction.

        Args:
            windows: List of window sizes for slider (default: 20 to 252 by 20).
            use_residuals: If True, use standardized residuals.

        Returns:
            Plotly figure with slider for window selection.
        """
        if windows is None:
            windows = list(range(20, 260, 20))  # 20, 40, 60, ..., 240

        # Get multi-window correlations
        multi_corr = self.get_rolling_correlation_multiwindow(
            windows=windows,
            use_residuals=use_residuals,
        )

        # Get list of pairs
        pairs = list(multi_corr.keys())
        colors = PLOT_COLORS[:len(pairs)]

        fig = go.Figure()

        # Add traces for each window (initially invisible except first)
        for w_idx, window in enumerate(windows):
            for pair_idx, pair in enumerate(pairs):
                corr_df = multi_corr[pair]
                col_name = f"{window}d"

                fig.add_trace(go.Scatter(
                    x=corr_df.index,
                    y=corr_df[col_name].values,
                    mode="lines",
                    name=f"{pair[0][:15]} vs {pair[1][:15]}",
                    line=dict(color=colors[pair_idx], width=1.5),
                    visible=(w_idx == 0),  # Only first window visible
                    showlegend=(w_idx == 0),  # Only show legend for first
                ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Create slider steps
        steps = []
        n_pairs = len(pairs)

        for w_idx, window in enumerate(windows):
            # Create visibility array
            visible = [False] * (len(windows) * n_pairs)
            for pair_idx in range(n_pairs):
                visible[w_idx * n_pairs + pair_idx] = True

            step = dict(
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Rolling Correlation ({window}-Day Window)"}
                ],
                label=str(window),
            )
            steps.append(step)

        # Add slider
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Window: ", "suffix": " days"},
            pad={"t": 50},
            steps=steps,
            x=0.1,
            len=0.8,
        )]

        fig.update_layout(
            title="Rolling Correlation (Use Slider to Change Window)",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis=dict(range=[-1.1, 1.1]),
            template=PLOT_TEMPLATE,
            hovermode="x unified",
            sliders=sliders,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            height=550,
        )

        return fig

    def plot_garch_diagnostics(self, strategy_name: str) -> go.Figure:
        """Plot GARCH model diagnostics for a strategy."""
        if strategy_name not in self._garch_results:
            raise ValueError(f"No GARCH result for {strategy_name}")

        result = self._garch_results[strategy_name]

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Returns",
                "Conditional Volatility",
                "Standardized Residuals",
                "Residuals Distribution",
                "ACF of Residuals",
                "ACF of Squared Residuals",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Returns
        fig.add_trace(
            go.Scatter(
                x=result.returns.index,
                y=result.returns.values * 100,
                mode="lines",
                name="Returns",
                line=dict(color=PLOT_COLORS[0], width=1),
            ),
            row=1, col=1,
        )

        # Conditional volatility
        fig.add_trace(
            go.Scatter(
                x=result.conditional_volatility.index,
                y=result.conditional_volatility.values * 100,
                mode="lines",
                name="Cond. Vol",
                line=dict(color=PLOT_COLORS[1], width=1),
            ),
            row=1, col=2,
        )

        # Standardized residuals
        fig.add_trace(
            go.Scatter(
                x=result.standardized_residuals.index,
                y=result.standardized_residuals.values,
                mode="lines",
                name="Std. Resid",
                line=dict(color=PLOT_COLORS[2], width=1),
            ),
            row=2, col=1,
        )

        # Residuals distribution
        std_resid = result.standardized_residuals.dropna()
        fig.add_trace(
            go.Histogram(
                x=std_resid.values,
                nbinsx=50,
                name="Residuals",
                marker_color=PLOT_COLORS[0],
                opacity=0.7,
            ),
            row=2, col=2,
        )

        # Add normal overlay
        x_norm = np.linspace(std_resid.min(), std_resid.max(), 100)
        y_norm = stats.norm.pdf(x_norm) * len(std_resid) * (std_resid.max() - std_resid.min()) / 50
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode="lines",
                name="Normal",
                line=dict(color="red", dash="dash"),
            ),
            row=2, col=2,
        )

        # ACF of residuals
        from statsmodels.tsa.stattools import acf
        acf_vals = acf(std_resid.values, nlags=20)
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_vals))),
                y=acf_vals,
                name="ACF",
                marker_color=PLOT_COLORS[3],
            ),
            row=3, col=1,
        )
        # Confidence bands
        conf = 1.96 / np.sqrt(len(std_resid))
        fig.add_hline(y=conf, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=-conf, line_dash="dot", line_color="gray", row=3, col=1)

        # ACF of squared residuals
        acf_sq = acf(std_resid.values ** 2, nlags=20)
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_sq))),
                y=acf_sq,
                name="ACF²",
                marker_color=PLOT_COLORS[4],
            ),
            row=3, col=2,
        )
        fig.add_hline(y=conf, line_dash="dot", line_color="gray", row=3, col=2)
        fig.add_hline(y=-conf, line_dash="dot", line_color="gray", row=3, col=2)

        fig.update_layout(
            height=900,
            title=dict(
                text=f"GARCH Diagnostics: {strategy_name} ({result.model_type})",
                y=0.98,
            ),
            showlegend=False,
            template=PLOT_TEMPLATE,
        )

        # Add diagnostic info
        diag_text = (
            f"AIC: {result.aic:.1f} | BIC: {result.bic:.1f}<br>"
            f"LB Resid p-val: {result.ljung_box_resid:.3f} | "
            f"LB Resid² p-val: {result.ljung_box_resid_sq:.3f}"
        )
        fig.add_annotation(
            x=0.5, y=-0.05,
            xref="paper", yref="paper",
            text=diag_text,
            showarrow=False,
            font=dict(size=11),
        )

        return fig

    def plot_pit_diagnostics(self) -> go.Figure:
        """Plot PIT transformation diagnostics."""
        if self._pit_uniforms is None:
            self.apply_pit()

        n_strategies = len(self.strategy_names)
        fig = make_subplots(
            rows=n_strategies, cols=2,
            subplot_titles=[
                item for name in self.strategy_names
                for item in [f"{name} - Histogram", f"{name} - QQ Plot"]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        for i, name in enumerate(self.strategy_names, 1):
            uniforms = self._pit_uniforms[name].dropna().values

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=uniforms,
                    nbinsx=30,
                    marker_color=PLOT_COLORS[i-1],
                    opacity=0.7,
                    showlegend=False,
                ),
                row=i, col=1,
            )

            # Expected uniform line
            fig.add_hline(
                y=len(uniforms) / 30,
                line_dash="dash",
                line_color="red",
                row=i, col=1,
            )

            # QQ plot against uniform
            sorted_u = np.sort(uniforms)
            theoretical = np.linspace(0, 1, len(sorted_u))

            fig.add_trace(
                go.Scatter(
                    x=theoretical,
                    y=sorted_u,
                    mode="markers",
                    marker=dict(size=3, color=PLOT_COLORS[i-1]),
                    showlegend=False,
                ),
                row=i, col=2,
            )

            # 45-degree line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=i, col=2,
            )

        fig.update_layout(
            height=300 * n_strategies,
            title="PIT Transformation Diagnostics (Should be Uniform)",
            template=PLOT_TEMPLATE,
        )

        return fig

    def plot_scatter_matrix(self, use_uniforms: bool = True) -> go.Figure:
        """Plot scatter matrix of PIT uniforms or residuals."""
        data = self._pit_uniforms if use_uniforms else self._standardized_residuals

        if data is None:
            if use_uniforms:
                self.apply_pit()
                data = self._pit_uniforms
            else:
                self.fit_garch()
                data = self._standardized_residuals

        n = len(self.strategy_names)

        fig = make_subplots(
            rows=n, cols=n,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        for i, name_i in enumerate(self.strategy_names, 1):
            for j, name_j in enumerate(self.strategy_names, 1):
                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(
                            x=data[name_i].dropna(),
                            nbinsx=30,
                            marker_color=PLOT_COLORS[i-1],
                            showlegend=False,
                        ),
                        row=i, col=j,
                    )
                else:
                    # Off-diagonal: scatter
                    fig.add_trace(
                        go.Scatter(
                            x=data[name_j],
                            y=data[name_i],
                            mode="markers",
                            marker=dict(
                                size=3,
                                color=PLOT_COLORS[0],
                                opacity=0.5,
                            ),
                            showlegend=False,
                        ),
                        row=i, col=j,
                    )

        # Add axis labels
        for i, name in enumerate(self.strategy_names, 1):
            fig.update_xaxes(title_text=name, row=n, col=i)
            fig.update_yaxes(title_text=name, row=i, col=1)

        title = "Scatter Matrix (PIT Uniforms)" if use_uniforms else "Scatter Matrix (Std. Residuals)"

        fig.update_layout(
            height=200 * n,
            width=200 * n,
            title=title,
            template=PLOT_TEMPLATE,
            showlegend=False,
        )

        return fig

    def plot_conditional_volatility(self) -> go.Figure:
        """Plot conditional volatility for all strategies."""
        if not self._garch_results:
            self.fit_garch()

        fig = go.Figure()

        for i, (name, result) in enumerate(self._garch_results.items()):
            fig.add_trace(go.Scatter(
                x=result.conditional_volatility.index,
                y=result.conditional_volatility.values * 100 * np.sqrt(252),  # Annualized
                mode="lines",
                name=name,
                line=dict(color=PLOT_COLORS[i], width=1.5),
            ))

        fig.update_layout(
            title="Annualized Conditional Volatility (GARCH)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template=PLOT_TEMPLATE,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
        )

        return fig

    def plot_tail_dependence(self, quantiles: list[float] = None) -> go.Figure:
        """Plot tail dependence across different quantiles."""
        if quantiles is None:
            quantiles = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Lower Tail Dependence", "Upper Tail Dependence"),
        )

        pairs = [
            (self.strategy_names[i], self.strategy_names[j])
            for i in range(len(self.strategy_names))
            for j in range(i + 1, len(self.strategy_names))
        ]

        for idx, pair in enumerate(pairs):
            lower_deps = []
            upper_deps = []

            for q in quantiles:
                tail_dep = self.estimate_tail_dependence(quantile=q)
                lower_deps.append(tail_dep[pair]["lower_tail"])
                upper_deps.append(tail_dep[pair]["upper_tail"])

            # Lower tail
            fig.add_trace(
                go.Scatter(
                    x=quantiles,
                    y=lower_deps,
                    mode="lines+markers",
                    name=f"{pair[0]} vs {pair[1]}",
                    line=dict(color=PLOT_COLORS[idx]),
                ),
                row=1, col=1,
            )

            # Upper tail
            fig.add_trace(
                go.Scatter(
                    x=quantiles,
                    y=upper_deps,
                    mode="lines+markers",
                    name=f"{pair[0]} vs {pair[1]}",
                    line=dict(color=PLOT_COLORS[idx]),
                    showlegend=False,
                ),
                row=1, col=2,
            )

        fig.update_layout(
            title="Tail Dependence by Quantile",
            template=PLOT_TEMPLATE,
            height=400,
        )

        fig.update_xaxes(title_text="Quantile", row=1, col=1)
        fig.update_xaxes(title_text="Quantile", row=1, col=2)
        fig.update_yaxes(title_text="Tail Dependence", row=1, col=1)
        fig.update_yaxes(title_text="Tail Dependence", row=1, col=2)

        return fig

    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics for all strategies."""
        if not self._garch_results:
            self.fit_garch()

        stats_data = []

        for name in self.strategy_names:
            result = self._garch_results[name]

            stats_data.append({
                "Strategy": name,
                "Model": result.model_type,
                "AIC": result.aic,
                "BIC": result.bic,
                "LB Resid p-val": result.ljung_box_resid,
                "LB Resid² p-val": result.ljung_box_resid_sq,
                "Valid": result.is_valid(),
                "Mean Vol (Ann.)": result.conditional_volatility.mean() * np.sqrt(252) * 100,
                "Vol of Vol": result.conditional_volatility.std() * np.sqrt(252) * 100,
            })

        return pd.DataFrame(stats_data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_correlation(
    returns_dict: dict[str, pd.Series],
    window: int = 60,
) -> tuple[CorrelationAnalyzer, dict[str, go.Figure]]:
    """
    Quick correlation analysis with default settings.

    Args:
        returns_dict: Dictionary of strategy returns.
        window: Rolling window for correlation.

    Returns:
        Tuple of (analyzer, plots_dict).
    """
    analyzer = CorrelationAnalyzer(returns_dict)
    analyzer.fit_garch()
    analyzer.apply_pit()

    plots = {
        "correlation_matrix": analyzer.plot_correlation_matrix(),
        "rolling_correlation": analyzer.plot_rolling_correlation(window=window),
        "scatter_matrix": analyzer.plot_scatter_matrix(),
        "conditional_volatility": analyzer.plot_conditional_volatility(),
        "tail_dependence": analyzer.plot_tail_dependence(),
        "pit_diagnostics": analyzer.plot_pit_diagnostics(),
    }

    return analyzer, plots


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Test the correlation analysis module."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

    from backtest_loader import BacktestLoader
    from config import BACKTESTS_DIR

    print("=" * 60)
    print("  CORRELATION ANALYSIS MODULE TEST")
    print("=" * 60)

    # Load backtests
    loader = BacktestLoader(BACKTESTS_DIR)
    backtests = loader.load_all()

    if len(backtests) < 2:
        print("Need at least 2 strategies for correlation analysis")
        return

    print(f"\nLoaded {len(backtests)} strategies:")
    for name in backtests:
        print(f"  - {name}")

    # Prepare returns
    returns_dict = {
        name: df.set_index("date")["daily_return_decimal"]
        for name, df in backtests.items()
    }

    # Create analyzer
    print("\nFitting GARCH models...")
    analyzer = CorrelationAnalyzer(returns_dict)
    garch_results = analyzer.fit_garch()

    # Print summary
    print("\nGARCH Model Summary:")
    print("-" * 80)
    summary = analyzer.generate_summary_stats()
    print(summary.to_string(index=False))

    # Correlation matrix
    print("\nCorrelation Matrix (Pearson on Standardized Residuals):")
    corr_matrix = analyzer.get_correlation_matrix()
    print(corr_matrix.round(3))

    # Tail dependence
    print("\nTail Dependence (5% quantile):")
    tail_dep = analyzer.estimate_tail_dependence(quantile=0.05)
    for pair, deps in tail_dep.items():
        print(f"  {pair[0]} vs {pair[1]}: Lower={deps['lower_tail']:.3f}, Upper={deps['upper_tail']:.3f}")

    # Generate plots
    print("\nGenerating plots...")
    plots = {
        "correlation_matrix": analyzer.plot_correlation_matrix(),
        "rolling_correlation": analyzer.plot_rolling_correlation(),
        "conditional_volatility": analyzer.plot_conditional_volatility(),
    }

    print(f"Generated {len(plots)} plots")

    print("\n" + "=" * 60)
    print("  TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
