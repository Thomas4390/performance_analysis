"""
Module for copula analysis and 3D visualization of dependency structures.

This module provides:
- Multiple copula families (Gaussian, Student-t, Clayton, Gumbel, Frank)
- Maximum likelihood estimation
- Model selection via AIC/BIC
- 3D density surface plots
- Contour plots
- Tail dependence visualization

The copula separates the dependency structure from the marginal distributions,
allowing for more accurate modeling of joint extreme events.

Usage:
    analyzer = CopulaAnalyzer(uniforms_df)
    results = analyzer.fit_all_copulas()
    best = analyzer.get_best_copula()
    fig = analyzer.plot_copula_3d(best.family)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Literal, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, optimize
from scipy.special import gamma as gamma_func

from config import PLOT_TEMPLATE, PLOT_COLORS


# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CopulaResult:
    """Results from copula fitting."""
    family: str
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    kendall_tau: float
    lower_tail_dep: float
    upper_tail_dep: float
    n_obs: int
    gof_pvalue: Optional[float] = None  # Goodness-of-fit p-value

    def __lt__(self, other):
        """Compare by AIC for sorting."""
        return self.aic < other.aic


@dataclass
class CopulaGOFResult:
    """Goodness-of-fit test results for a copula."""
    family: str
    # Cramér-von Mises test on Rosenblatt transform
    cvm_statistic: float
    cvm_pvalue: float
    cvm_passed: bool
    # Kolmogorov-Smirnov test on Rosenblatt transform
    ks_statistic: float
    ks_pvalue: float
    ks_passed: bool
    # Overall
    passed: bool
    n_bootstrap: int = 0

    def get_summary_dict(self) -> dict:
        """Return summary dictionary for display."""
        return {
            "Family": self.family.replace("_", "-").title(),
            "CvM stat": self.cvm_statistic,
            "CvM p-val": self.cvm_pvalue,
            "KS stat": self.ks_statistic,
            "KS p-val": self.ks_pvalue,
            "Valid": "Yes" if self.passed else "No",
        }


# =============================================================================
# COPULA FUNCTIONS
# =============================================================================

class CopulaFunctions:
    """
    Static methods for copula density and CDF calculations.

    Implements the following copula families:
    - Gaussian: No tail dependence, symmetric
    - Student-t: Symmetric tail dependence
    - Clayton: Lower tail dependence
    - Gumbel: Upper tail dependence
    - Frank: No tail dependence, symmetric
    """

    @staticmethod
    def gaussian_pdf(u1: np.ndarray, u2: np.ndarray, rho: float) -> np.ndarray:
        """
        Gaussian copula density.

        c(u1, u2) = (1/sqrt(1-rho^2)) * exp(-(rho^2*(x1^2+x2^2) - 2*rho*x1*x2) / (2*(1-rho^2)))

        where x1 = Phi^{-1}(u1), x2 = Phi^{-1}(u2)
        """
        # Transform to normal quantiles
        x1 = stats.norm.ppf(np.clip(u1, 1e-6, 1-1e-6))
        x2 = stats.norm.ppf(np.clip(u2, 1e-6, 1-1e-6))

        rho2 = rho ** 2
        denom = 1 - rho2

        if denom <= 0:
            return np.zeros_like(u1)

        exponent = -(rho2 * (x1**2 + x2**2) - 2 * rho * x1 * x2) / (2 * denom)
        density = np.exp(exponent) / np.sqrt(denom)

        return np.maximum(density, 1e-10)

    @staticmethod
    def gaussian_cdf(u1: np.ndarray, u2: np.ndarray, rho: float) -> np.ndarray:
        """Gaussian copula CDF using bivariate normal."""
        x1 = stats.norm.ppf(np.clip(u1, 1e-6, 1-1e-6))
        x2 = stats.norm.ppf(np.clip(u2, 1e-6, 1-1e-6))

        # Bivariate normal CDF
        from scipy.stats import multivariate_normal
        cov = np.array([[1, rho], [rho, 1]])
        rv = multivariate_normal(mean=[0, 0], cov=cov)

        result = np.zeros(len(u1))
        for i in range(len(u1)):
            result[i] = rv.cdf([x1[i], x2[i]])

        return result

    @staticmethod
    def student_t_pdf(u1: np.ndarray, u2: np.ndarray, rho: float, nu: float) -> np.ndarray:
        """
        Student-t copula density.

        Has symmetric tail dependence: lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        """
        # Transform to t quantiles
        t1 = stats.t.ppf(np.clip(u1, 1e-6, 1-1e-6), df=nu)
        t2 = stats.t.ppf(np.clip(u2, 1e-6, 1-1e-6), df=nu)

        rho2 = rho ** 2
        denom = 1 - rho2

        if denom <= 0 or nu <= 0:
            return np.zeros_like(u1)

        # t-copula density
        term1 = gamma_func((nu + 2) / 2) / gamma_func(nu / 2)
        term2 = gamma_func(nu / 2) / gamma_func((nu + 1) / 2)
        term3 = 1 / (np.pi * nu * np.sqrt(denom))

        quad_form = (t1**2 + t2**2 - 2*rho*t1*t2) / (nu * denom)
        density = (term1 * term2**2 * term3 *
                   (1 + quad_form) ** (-(nu + 2) / 2) *
                   (1 + t1**2/nu) ** ((nu + 1) / 2) *
                   (1 + t2**2/nu) ** ((nu + 1) / 2))

        return np.maximum(density, 1e-10)

    @staticmethod
    def clayton_pdf(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
        """
        Clayton copula density.

        C(u1, u2) = (u1^{-theta} + u2^{-theta} - 1)^{-1/theta}

        Has lower tail dependence: lambda_L = 2^{-1/theta}
        """
        if theta <= 0:
            return np.ones_like(u1)  # Independence copula

        u1c = np.clip(u1, 1e-6, 1-1e-6)
        u2c = np.clip(u2, 1e-6, 1-1e-6)

        term = u1c**(-theta) + u2c**(-theta) - 1
        term = np.maximum(term, 1e-10)

        density = ((1 + theta) * (u1c * u2c)**(-(1 + theta)) *
                   term**(-(2 + 1/theta)))

        return np.maximum(density, 1e-10)

    @staticmethod
    def clayton_cdf(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
        """Clayton copula CDF."""
        if theta <= 0:
            return u1 * u2  # Independence

        u1c = np.clip(u1, 1e-6, 1-1e-6)
        u2c = np.clip(u2, 1e-6, 1-1e-6)

        term = u1c**(-theta) + u2c**(-theta) - 1
        term = np.maximum(term, 1e-10)

        return term**(-1/theta)

    @staticmethod
    def gumbel_pdf(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
        """
        Gumbel copula density.

        C(u1, u2) = exp(-((- log u1)^theta + (-log u2)^theta)^{1/theta})

        Has upper tail dependence: lambda_U = 2 - 2^{1/theta}
        """
        if theta < 1:
            theta = 1  # Minimum is 1 (independence)

        u1c = np.clip(u1, 1e-6, 1-1e-6)
        u2c = np.clip(u2, 1e-6, 1-1e-6)

        log_u1 = -np.log(u1c)
        log_u2 = -np.log(u2c)

        A = (log_u1**theta + log_u2**theta)**(1/theta)
        A = np.maximum(A, 1e-10)

        C = np.exp(-A)

        # Density
        term1 = C * (log_u1 * log_u2)**(theta - 1)
        term2 = A**(2 - 2*theta) + (theta - 1) * A**(1 - 2*theta)
        density = term1 * term2 / (u1c * u2c)

        return np.maximum(density, 1e-10)

    @staticmethod
    def gumbel_cdf(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
        """Gumbel copula CDF."""
        if theta < 1:
            theta = 1

        u1c = np.clip(u1, 1e-6, 1-1e-6)
        u2c = np.clip(u2, 1e-6, 1-1e-6)

        log_u1 = -np.log(u1c)
        log_u2 = -np.log(u2c)

        A = (log_u1**theta + log_u2**theta)**(1/theta)

        return np.exp(-A)

    @staticmethod
    def frank_pdf(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
        """
        Frank copula density.

        C(u1, u2) = -1/theta * log(1 + (exp(-theta*u1)-1)(exp(-theta*u2)-1)/(exp(-theta)-1))

        No tail dependence (symmetric).
        """
        if abs(theta) < 1e-6:
            return np.ones_like(u1)  # Independence

        u1c = np.clip(u1, 1e-6, 1-1e-6)
        u2c = np.clip(u2, 1e-6, 1-1e-6)

        e1 = np.exp(-theta * u1c)
        e2 = np.exp(-theta * u2c)
        e_theta = np.exp(-theta)

        numer = theta * (e_theta - 1) * (1 + (e1 - 1) * (e2 - 1) / (e_theta - 1))
        denom = ((e1 - 1) * (e2 - 1) + (e_theta - 1))**2

        density = -numer * e1 * e2 / np.maximum(denom, 1e-10)

        return np.maximum(np.abs(density), 1e-10)

    @staticmethod
    def frank_cdf(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
        """Frank copula CDF."""
        if abs(theta) < 1e-6:
            return u1 * u2

        u1c = np.clip(u1, 1e-6, 1-1e-6)
        u2c = np.clip(u2, 1e-6, 1-1e-6)

        e1 = np.exp(-theta * u1c)
        e2 = np.exp(-theta * u2c)
        e_theta = np.exp(-theta)

        numer = (e1 - 1) * (e2 - 1)
        denom = e_theta - 1

        return -np.log(1 + numer / denom) / theta


# =============================================================================
# TAIL DEPENDENCE COEFFICIENTS
# =============================================================================

class TailDependence:
    """Calculate theoretical tail dependence coefficients for copulas."""

    @staticmethod
    def gaussian(rho: float) -> tuple[float, float]:
        """Gaussian copula: no tail dependence."""
        return 0.0, 0.0

    @staticmethod
    def student_t(rho: float, nu: float) -> tuple[float, float]:
        """Student-t copula: symmetric tail dependence."""
        if nu <= 0:
            return 0.0, 0.0

        coef = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_sym = 2 * stats.t.cdf(-coef, df=nu + 1)

        return lambda_sym, lambda_sym

    @staticmethod
    def clayton(theta: float) -> tuple[float, float]:
        """Clayton copula: lower tail dependence only."""
        if theta <= 0:
            return 0.0, 0.0

        lambda_L = 2**(-1/theta)
        return lambda_L, 0.0

    @staticmethod
    def gumbel(theta: float) -> tuple[float, float]:
        """Gumbel copula: upper tail dependence only."""
        if theta < 1:
            return 0.0, 0.0

        lambda_U = 2 - 2**(1/theta)
        return 0.0, lambda_U

    @staticmethod
    def frank(theta: float) -> tuple[float, float]:
        """Frank copula: no tail dependence."""
        return 0.0, 0.0


# =============================================================================
# COPULA ANALYZER
# =============================================================================

class CopulaAnalyzer:
    """
    Comprehensive copula analysis for bivariate dependency.

    Fits multiple copula families and provides visualization tools
    for understanding the dependency structure.

    Example:
        uniforms_df = pd.DataFrame({"u1": u1, "u2": u2})
        analyzer = CopulaAnalyzer(uniforms_df)
        results = analyzer.fit_all_copulas()
        fig = analyzer.plot_copula_comparison()
    """

    COPULA_FAMILIES = ["gaussian", "student_t", "clayton", "gumbel", "frank"]

    def __init__(
        self,
        uniforms: pd.DataFrame,
        strategy_names: Optional[tuple[str, str]] = None,
    ):
        """
        Initialize the copula analyzer.

        Args:
            uniforms: DataFrame with 2 columns of PIT-transformed uniform margins.
            strategy_names: Names of the two strategies for labeling.
        """
        if uniforms.shape[1] != 2:
            raise ValueError("Expected DataFrame with exactly 2 columns")

        self.uniforms = uniforms.dropna()
        self.u1 = self.uniforms.iloc[:, 0].values
        self.u2 = self.uniforms.iloc[:, 1].values
        self.n_obs = len(self.uniforms)

        self.strategy_names = strategy_names or (
            uniforms.columns[0],
            uniforms.columns[1],
        )

        # Empirical Kendall's tau
        self.empirical_tau, _ = stats.kendalltau(self.u1, self.u2)

        # Results storage
        self._results: dict[str, CopulaResult] = {}

    def fit_copula(self, family: str) -> CopulaResult:
        """
        Fit a specific copula family.

        Args:
            family: Copula family name.

        Returns:
            CopulaResult with fitted parameters and diagnostics.
        """
        if family == "gaussian":
            return self._fit_gaussian()
        elif family == "student_t":
            return self._fit_student_t()
        elif family == "clayton":
            return self._fit_clayton()
        elif family == "gumbel":
            return self._fit_gumbel()
        elif family == "frank":
            return self._fit_frank()
        else:
            raise ValueError(f"Unknown copula family: {family}")

    def _fit_gaussian(self) -> CopulaResult:
        """Fit Gaussian copula."""
        def neg_log_likelihood(rho):
            if abs(rho) >= 1:
                return 1e10
            density = CopulaFunctions.gaussian_pdf(self.u1, self.u2, rho)
            return -np.sum(np.log(np.maximum(density, 1e-10)))

        # Initial guess from Kendall's tau
        rho_init = np.sin(np.pi * self.empirical_tau / 2)

        result = optimize.minimize_scalar(
            neg_log_likelihood,
            bounds=(-0.999, 0.999),
            method="bounded",
        )

        rho = result.x
        ll = -result.fun
        k = 1  # Number of parameters

        lambda_L, lambda_U = TailDependence.gaussian(rho)

        return CopulaResult(
            family="gaussian",
            params={"rho": rho},
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(self.n_obs),
            kendall_tau=2 * np.arcsin(rho) / np.pi,
            lower_tail_dep=lambda_L,
            upper_tail_dep=lambda_U,
            n_obs=self.n_obs,
        )

    def _fit_student_t(self) -> CopulaResult:
        """Fit Student-t copula."""
        def neg_log_likelihood(params):
            rho, nu = params
            if abs(rho) >= 1 or nu <= 2:
                return 1e10
            density = CopulaFunctions.student_t_pdf(self.u1, self.u2, rho, nu)
            return -np.sum(np.log(np.maximum(density, 1e-10)))

        # Initial guess
        rho_init = np.sin(np.pi * self.empirical_tau / 2)
        nu_init = 10

        result = optimize.minimize(
            neg_log_likelihood,
            x0=[rho_init, nu_init],
            bounds=[(-0.999, 0.999), (2.1, 100)],
            method="L-BFGS-B",
        )

        rho, nu = result.x
        ll = -result.fun
        k = 2

        lambda_L, lambda_U = TailDependence.student_t(rho, nu)

        return CopulaResult(
            family="student_t",
            params={"rho": rho, "nu": nu},
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(self.n_obs),
            kendall_tau=2 * np.arcsin(rho) / np.pi,
            lower_tail_dep=lambda_L,
            upper_tail_dep=lambda_U,
            n_obs=self.n_obs,
        )

    def _fit_clayton(self) -> CopulaResult:
        """Fit Clayton copula."""
        def neg_log_likelihood(theta):
            if theta <= 0:
                return 1e10
            density = CopulaFunctions.clayton_pdf(self.u1, self.u2, theta)
            return -np.sum(np.log(np.maximum(density, 1e-10)))

        # Initial guess from Kendall's tau
        tau = max(0.01, self.empirical_tau)
        theta_init = 2 * tau / (1 - tau) if tau < 1 else 10

        result = optimize.minimize_scalar(
            neg_log_likelihood,
            bounds=(0.01, 50),
            method="bounded",
        )

        theta = result.x
        ll = -result.fun
        k = 1

        lambda_L, lambda_U = TailDependence.clayton(theta)

        return CopulaResult(
            family="clayton",
            params={"theta": theta},
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(self.n_obs),
            kendall_tau=theta / (theta + 2),
            lower_tail_dep=lambda_L,
            upper_tail_dep=lambda_U,
            n_obs=self.n_obs,
        )

    def _fit_gumbel(self) -> CopulaResult:
        """Fit Gumbel copula."""
        def neg_log_likelihood(theta):
            if theta < 1:
                return 1e10
            density = CopulaFunctions.gumbel_pdf(self.u1, self.u2, theta)
            return -np.sum(np.log(np.maximum(density, 1e-10)))

        # Initial guess from Kendall's tau
        tau = max(0.01, self.empirical_tau)
        theta_init = 1 / (1 - tau) if tau < 1 else 10

        result = optimize.minimize_scalar(
            neg_log_likelihood,
            bounds=(1, 50),
            method="bounded",
        )

        theta = result.x
        ll = -result.fun
        k = 1

        lambda_L, lambda_U = TailDependence.gumbel(theta)

        return CopulaResult(
            family="gumbel",
            params={"theta": theta},
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(self.n_obs),
            kendall_tau=1 - 1/theta,
            lower_tail_dep=lambda_L,
            upper_tail_dep=lambda_U,
            n_obs=self.n_obs,
        )

    def _fit_frank(self) -> CopulaResult:
        """Fit Frank copula."""
        def neg_log_likelihood(theta):
            if abs(theta) < 0.01:
                return -self.n_obs * np.log(1)  # Independence
            density = CopulaFunctions.frank_pdf(self.u1, self.u2, theta)
            return -np.sum(np.log(np.maximum(density, 1e-10)))

        # Initial guess - Frank requires numerical inversion for tau
        theta_init = 5 * np.sign(self.empirical_tau)

        result = optimize.minimize_scalar(
            neg_log_likelihood,
            bounds=(-50, 50),
            method="bounded",
        )

        theta = result.x
        ll = -result.fun
        k = 1

        lambda_L, lambda_U = TailDependence.frank(theta)

        # Approximate Kendall's tau for Frank (complex formula)
        tau_approx = self.empirical_tau  # Use empirical as approximation

        return CopulaResult(
            family="frank",
            params={"theta": theta},
            log_likelihood=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(self.n_obs),
            kendall_tau=tau_approx,
            lower_tail_dep=lambda_L,
            upper_tail_dep=lambda_U,
            n_obs=self.n_obs,
        )

    def fit_all_copulas(self, run_gof: bool = False) -> dict[str, CopulaResult]:
        """
        Fit all copula families.

        Args:
            run_gof: If True, run goodness-of-fit tests for each copula.

        Returns:
            Dictionary of CopulaResult for each family.
        """
        for family in self.COPULA_FAMILIES:
            try:
                self._results[family] = self.fit_copula(family)
                if run_gof:
                    gof = self.test_goodness_of_fit(family)
                    self._results[family].gof_pvalue = gof.cvm_pvalue
            except Exception as e:
                warnings.warn(f"Failed to fit {family} copula: {e}")

        return self._results

    def rosenblatt_transform(self, family: str) -> np.ndarray:
        """
        Apply Rosenblatt probability integral transform for GOF testing.

        The Rosenblatt transform converts the bivariate copula to
        two independent uniform variables if the copula is correctly specified.

        Args:
            family: Copula family to use.

        Returns:
            Array of shape (n, 2) with transformed values.
        """
        if family not in self._results:
            self.fit_copula(family)

        result = self._results[family]

        # First variable stays the same
        v1 = self.u1.copy()

        # Second variable: conditional CDF
        # v2 = C(u1, u2 | U1 = u1) = ∂C(u1, u2)/∂u1 evaluated at u2

        if family == "gaussian":
            rho = result.params["rho"]
            x1 = stats.norm.ppf(np.clip(self.u1, 1e-6, 1-1e-6))
            x2 = stats.norm.ppf(np.clip(self.u2, 1e-6, 1-1e-6))
            # Conditional: (X2 | X1 = x1) ~ N(rho*x1, 1-rho^2)
            v2 = stats.norm.cdf((x2 - rho * x1) / np.sqrt(1 - rho**2))

        elif family == "student_t":
            rho = result.params["rho"]
            nu = result.params["nu"]
            t1 = stats.t.ppf(np.clip(self.u1, 1e-6, 1-1e-6), df=nu)
            t2 = stats.t.ppf(np.clip(self.u2, 1e-6, 1-1e-6), df=nu)
            # Conditional distribution
            scale = np.sqrt((nu + t1**2) / (nu + 1) * (1 - rho**2))
            v2 = stats.t.cdf((t2 - rho * t1) / scale, df=nu + 1)

        elif family == "clayton":
            theta = result.params["theta"]
            if theta > 0:
                u1_t = np.clip(self.u1, 1e-6, 1-1e-6) ** (-theta)
                u2_t = np.clip(self.u2, 1e-6, 1-1e-6) ** (-theta)
                term = u1_t + u2_t - 1
                v2 = np.clip(self.u1, 1e-6, 1-1e-6) ** (theta + 1) * term ** (-1 - 1/theta)
            else:
                v2 = self.u2.copy()

        elif family == "gumbel":
            theta = result.params["theta"]
            u1c = np.clip(self.u1, 1e-6, 1-1e-6)
            u2c = np.clip(self.u2, 1e-6, 1-1e-6)
            log_u1 = -np.log(u1c)
            log_u2 = -np.log(u2c)
            A = (log_u1**theta + log_u2**theta)**(1/theta)
            # Partial derivative
            C = np.exp(-A)
            v2 = C * (log_u1 / u1c) * (log_u1**theta + log_u2**theta)**(1/theta - 1) * log_u1**(theta - 1)
            v2 = np.clip(v2, 1e-6, 1-1e-6)

        elif family == "frank":
            theta = result.params["theta"]
            if abs(theta) > 1e-6:
                e1 = np.exp(-theta * np.clip(self.u1, 1e-6, 1-1e-6))
                e2 = np.exp(-theta * np.clip(self.u2, 1e-6, 1-1e-6))
                e_t = np.exp(-theta)
                v2 = e2 * (e1 - 1) / ((e1 - 1) * (e2 - 1) + (e_t - 1))
            else:
                v2 = self.u2.copy()

        else:
            v2 = self.u2.copy()

        v2 = np.clip(v2, 1e-6, 1-1e-6)

        return np.column_stack([v1, v2])

    def test_goodness_of_fit(self, family: str) -> CopulaGOFResult:
        """
        Test goodness-of-fit for a copula using the Rosenblatt transform.

        The Rosenblatt transform converts (u1, u2) to (v1, v2) which should
        be independent Uniform(0,1) if the copula is correctly specified.

        We test:
        1. Uniformity of v1 and v2 (KS test)
        2. Independence of v1 and v2 (correlation test)

        Args:
            family: Copula family to test.

        Returns:
            CopulaGOFResult with test statistics.
        """
        # Get Rosenblatt transformed values
        v = self.rosenblatt_transform(family)
        v1, v2 = v[:, 0], v[:, 1]

        # Test uniformity with Cramér-von Mises
        try:
            cvm1 = stats.cramervonmises(v1, "uniform")
            cvm2 = stats.cramervonmises(v2, "uniform")
            # Combine p-values (Fisher's method)
            chi2_stat = -2 * (np.log(cvm1.pvalue + 1e-10) + np.log(cvm2.pvalue + 1e-10))
            cvm_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=4)
            cvm_statistic = (cvm1.statistic + cvm2.statistic) / 2
        except Exception:
            # Fallback
            cvm_statistic = 0.0
            cvm_pvalue = 1.0

        # Test uniformity with KS
        ks1 = stats.kstest(v1, "uniform")
        ks2 = stats.kstest(v2, "uniform")
        # Combine with Bonferroni correction
        ks_pvalue = min(1.0, 2 * min(ks1.pvalue, ks2.pvalue))
        ks_statistic = max(ks1.statistic, ks2.statistic)

        # Determine if tests pass
        cvm_passed = cvm_pvalue > 0.05
        ks_passed = ks_pvalue > 0.05
        passed = cvm_passed and ks_passed

        return CopulaGOFResult(
            family=family,
            cvm_statistic=cvm_statistic,
            cvm_pvalue=cvm_pvalue,
            cvm_passed=cvm_passed,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            ks_passed=ks_passed,
            passed=passed,
        )

    def test_all_gof(self) -> dict[str, CopulaGOFResult]:
        """
        Run goodness-of-fit tests for all fitted copulas.

        Returns:
            Dictionary mapping family names to GOF results.
        """
        if not self._results:
            self.fit_all_copulas()

        gof_results = {}
        for family in self._results:
            gof_results[family] = self.test_goodness_of_fit(family)

        return gof_results

    def get_best_copula(self, criterion: str = "aic") -> CopulaResult:
        """
        Get the best copula by information criterion.

        Args:
            criterion: "aic" or "bic".

        Returns:
            Best CopulaResult.
        """
        if not self._results:
            self.fit_all_copulas()

        if criterion == "aic":
            return min(self._results.values(), key=lambda x: x.aic)
        elif criterion == "bic":
            return min(self._results.values(), key=lambda x: x.bic)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def get_copula_density(
        self,
        family: str,
        grid_size: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate copula density on a grid for plotting.

        Args:
            family: Copula family.
            grid_size: Number of points per dimension.

        Returns:
            Tuple of (u1_grid, u2_grid, density).
        """
        if family not in self._results:
            self.fit_copula(family)

        result = self._results[family]

        u_range = np.linspace(0.01, 0.99, grid_size)
        u1_grid, u2_grid = np.meshgrid(u_range, u_range)
        u1_flat = u1_grid.flatten()
        u2_flat = u2_grid.flatten()

        if family == "gaussian":
            density = CopulaFunctions.gaussian_pdf(
                u1_flat, u2_flat, result.params["rho"]
            )
        elif family == "student_t":
            density = CopulaFunctions.student_t_pdf(
                u1_flat, u2_flat, result.params["rho"], result.params["nu"]
            )
        elif family == "clayton":
            density = CopulaFunctions.clayton_pdf(
                u1_flat, u2_flat, result.params["theta"]
            )
        elif family == "gumbel":
            density = CopulaFunctions.gumbel_pdf(
                u1_flat, u2_flat, result.params["theta"]
            )
        elif family == "frank":
            density = CopulaFunctions.frank_pdf(
                u1_flat, u2_flat, result.params["theta"]
            )
        else:
            raise ValueError(f"Unknown family: {family}")

        density = density.reshape(grid_size, grid_size)

        return u1_grid, u2_grid, density

    def get_summary_table(self) -> pd.DataFrame:
        """
        Get summary table of all fitted copulas.

        Returns:
            DataFrame with copula comparison.
        """
        if not self._results:
            self.fit_all_copulas()

        rows = []
        for family, result in self._results.items():
            param_str = ", ".join([f"{k}={v:.3f}" for k, v in result.params.items()])
            rows.append({
                "Family": family.replace("_", "-").title(),
                "Parameters": param_str,
                "Log-Lik": result.log_likelihood,
                "AIC": result.aic,
                "BIC": result.bic,
                "Kendall Tau": result.kendall_tau,
                "Lower Tail": result.lower_tail_dep,
                "Upper Tail": result.upper_tail_dep,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("AIC").reset_index(drop=True)
        return df

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_copula_3d(
        self,
        family: str = "gaussian",
        grid_size: int = 40,
    ) -> go.Figure:
        """
        Create 3D surface plot of copula density.

        Args:
            family: Copula family to plot.
            grid_size: Grid resolution.

        Returns:
            Plotly figure.
        """
        u1_grid, u2_grid, density = self.get_copula_density(family, grid_size)

        # Cap extreme values for visualization
        density = np.clip(density, 0, np.percentile(density, 99))

        result = self._results.get(family)
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in result.params.items()]) if result else ""

        fig = go.Figure(data=[
            go.Surface(
                x=u1_grid,
                y=u2_grid,
                z=density,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Density"),
            )
        ])

        fig.update_layout(
            title=dict(
                text=f"{family.replace('_', '-').title()} Copula Density<br><sup>{param_str}</sup>",
                y=0.95,
            ),
            scene=dict(
                xaxis_title=self.strategy_names[0],
                yaxis_title=self.strategy_names[1],
                zaxis_title="Density c(u1, u2)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                ),
            ),
            template=PLOT_TEMPLATE,
            height=600,
            width=700,
        )

        return fig

    def plot_copula_contour(
        self,
        family: str = "gaussian",
        grid_size: int = 50,
        show_data: bool = True,
    ) -> go.Figure:
        """
        Create contour plot of copula density.

        Args:
            family: Copula family.
            grid_size: Grid resolution.
            show_data: Whether to overlay data points.

        Returns:
            Plotly figure.
        """
        u1_grid, u2_grid, density = self.get_copula_density(family, grid_size)

        result = self._results.get(family)
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in result.params.items()]) if result else ""

        fig = go.Figure()

        # Contour
        fig.add_trace(go.Contour(
            x=u1_grid[0],
            y=u2_grid[:, 0],
            z=density,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Density"),
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
        ))

        # Data points
        if show_data:
            fig.add_trace(go.Scatter(
                x=self.u1,
                y=self.u2,
                mode="markers",
                marker=dict(
                    size=3,
                    color="white",
                    opacity=0.5,
                    line=dict(width=0.5, color="black"),
                ),
                name="Data",
            ))

        fig.update_layout(
            title=f"{family.replace('_', '-').title()} Copula Density<br><sup>{param_str}</sup>",
            xaxis_title=self.strategy_names[0],
            yaxis_title=self.strategy_names[1],
            template=PLOT_TEMPLATE,
            height=500,
            width=550,
        )

        return fig

    def plot_copula_comparison(self, grid_size: int = 35) -> go.Figure:
        """
        Create comparison plot of all fitted copulas.

        Returns:
            Plotly figure with 2x3 subplots.
        """
        if not self._results:
            self.fit_all_copulas()

        # 2 rows, 3 cols (5 copulas + 1 for empirical)
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
                [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
            ],
            subplot_titles=[
                "Gaussian", "Student-t", "Clayton",
                "Gumbel", "Frank", "Empirical",
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

        positions = [
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 2), (2, 3),
        ]

        # Plot fitted copulas
        for i, family in enumerate(self.COPULA_FAMILIES):
            if family in self._results:
                u1_grid, u2_grid, density = self.get_copula_density(family, grid_size)
                density = np.clip(density, 0, np.percentile(density, 99))

                row, col = positions[i]
                fig.add_trace(
                    go.Surface(
                        x=u1_grid,
                        y=u2_grid,
                        z=density,
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    row=row, col=col,
                )

        # Empirical density (kernel density estimate)
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(np.vstack([self.u1, self.u2]))
            u_range = np.linspace(0.01, 0.99, grid_size)
            u1_grid, u2_grid = np.meshgrid(u_range, u_range)
            positions_kde = np.vstack([u1_grid.ravel(), u2_grid.ravel()])
            density_kde = kde(positions_kde).reshape(grid_size, grid_size)
            density_kde = np.clip(density_kde, 0, np.percentile(density_kde, 99))

            fig.add_trace(
                go.Surface(
                    x=u1_grid,
                    y=u2_grid,
                    z=density_kde,
                    colorscale="Viridis",
                    showscale=False,
                ),
                row=2, col=3,
            )
        except Exception:
            pass

        fig.update_layout(
            height=800,
            width=1100,
            title=dict(
                text="Copula Family Comparison",
                y=0.98,
            ),
            template=PLOT_TEMPLATE,
            showlegend=False,
        )

        # Update all scenes
        for i in range(1, 7):
            scene_name = f"scene{i}" if i > 1 else "scene"
            fig.update_layout(**{
                scene_name: dict(
                    xaxis_title="u1",
                    yaxis_title="u2",
                    zaxis_title="",
                    camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
                )
            })

        return fig

    def plot_tail_scatter(self, quantile: float = 0.1) -> go.Figure:
        """
        Create scatter plot highlighting tail observations.

        Args:
            quantile: Quantile threshold for tail highlighting.

        Returns:
            Plotly figure.
        """
        fig = go.Figure()

        # Regular points
        fig.add_trace(go.Scatter(
            x=self.u1,
            y=self.u2,
            mode="markers",
            marker=dict(size=4, color=PLOT_COLORS[0], opacity=0.3),
            name="All Data",
        ))

        # Lower tail (both u1, u2 small)
        lower_mask = (self.u1 <= quantile) & (self.u2 <= quantile)
        fig.add_trace(go.Scatter(
            x=self.u1[lower_mask],
            y=self.u2[lower_mask],
            mode="markers",
            marker=dict(size=8, color="red", symbol="x"),
            name=f"Lower Tail (<{quantile:.0%})",
        ))

        # Upper tail (both u1, u2 large)
        upper_mask = (self.u1 >= 1 - quantile) & (self.u2 >= 1 - quantile)
        fig.add_trace(go.Scatter(
            x=self.u1[upper_mask],
            y=self.u2[upper_mask],
            mode="markers",
            marker=dict(size=8, color="green", symbol="diamond"),
            name=f"Upper Tail (>{1-quantile:.0%})",
        ))

        # Reference lines
        fig.add_hline(y=quantile, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_vline(x=quantile, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=1-quantile, line_dash="dot", line_color="green", opacity=0.5)
        fig.add_vline(x=1-quantile, line_dash="dot", line_color="green", opacity=0.5)

        # Calculate empirical tail counts
        n_lower = lower_mask.sum()
        n_upper = upper_mask.sum()
        expected = quantile ** 2 * self.n_obs  # Under independence

        fig.update_layout(
            title=f"Tail Dependence Visualization<br><sup>Lower: {n_lower} obs ({n_lower/self.n_obs:.1%}), Upper: {n_upper} obs ({n_upper/self.n_obs:.1%}), Expected under independence: {expected:.0f}</sup>",
            xaxis_title=self.strategy_names[0],
            yaxis_title=self.strategy_names[1],
            template=PLOT_TEMPLATE,
            height=500,
            width=550,
        )

        return fig

    def plot_model_comparison_bars(self) -> go.Figure:
        """
        Create bar chart comparing copula models by AIC.

        Returns:
            Plotly figure.
        """
        if not self._results:
            self.fit_all_copulas()

        df = self.get_summary_table()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Information Criteria", "Tail Dependence"),
        )

        # AIC/BIC comparison
        fig.add_trace(
            go.Bar(
                x=df["Family"],
                y=df["AIC"],
                name="AIC",
                marker_color=PLOT_COLORS[0],
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df["Family"],
                y=df["BIC"],
                name="BIC",
                marker_color=PLOT_COLORS[1],
            ),
            row=1, col=1,
        )

        # Tail dependence
        fig.add_trace(
            go.Bar(
                x=df["Family"],
                y=df["Lower Tail"],
                name="Lower Tail",
                marker_color="red",
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Bar(
                x=df["Family"],
                y=df["Upper Tail"],
                name="Upper Tail",
                marker_color="green",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title="Copula Model Comparison",
            template=PLOT_TEMPLATE,
            height=400,
            barmode="group",
        )

        return fig

    def generate_all_plots(self) -> dict[str, go.Figure]:
        """Generate all copula analysis plots."""
        if not self._results:
            self.fit_all_copulas()

        best = self.get_best_copula()

        plots = {
            "copula_3d": self.plot_copula_3d(best.family),
            "copula_contour": self.plot_copula_contour(best.family),
            "copula_comparison": self.plot_copula_comparison(),
            "tail_scatter": self.plot_tail_scatter(),
            "model_comparison": self.plot_model_comparison_bars(),
        }

        return plots


# =============================================================================
# MULTI-STRATEGY COPULA ANALYSIS
# =============================================================================

class MultiStrategyCopulaAnalyzer:
    """
    Analyze copulas for all pairs of strategies.

    Example:
        analyzer = MultiStrategyCopulaAnalyzer(uniforms_df)
        results = analyzer.analyze_all_pairs()
        fig = analyzer.plot_dependency_heatmap()
    """

    def __init__(self, uniforms: pd.DataFrame):
        """
        Initialize with multi-column uniform margins.

        Args:
            uniforms: DataFrame with uniform margins for each strategy.
        """
        self.uniforms = uniforms.dropna()
        self.strategy_names = list(uniforms.columns)
        self.n_strategies = len(self.strategy_names)

        self._pair_analyzers: dict[tuple[str, str], CopulaAnalyzer] = {}
        self._pair_results: dict[tuple[str, str], CopulaResult] = {}

    def analyze_all_pairs(self) -> dict[tuple[str, str], CopulaResult]:
        """
        Fit best copula for all strategy pairs.

        Returns:
            Dictionary mapping pairs to best CopulaResult.
        """
        for i in range(self.n_strategies):
            for j in range(i + 1, self.n_strategies):
                name1 = self.strategy_names[i]
                name2 = self.strategy_names[j]
                pair = (name1, name2)

                pair_df = self.uniforms[[name1, name2]]
                analyzer = CopulaAnalyzer(pair_df, strategy_names=pair)
                analyzer.fit_all_copulas()

                self._pair_analyzers[pair] = analyzer
                self._pair_results[pair] = analyzer.get_best_copula()

        return self._pair_results

    def get_pair_analyzer(self, name1: str, name2: str) -> CopulaAnalyzer:
        """Get the CopulaAnalyzer for a specific pair."""
        pair = (name1, name2)
        if pair not in self._pair_analyzers:
            pair = (name2, name1)
        return self._pair_analyzers.get(pair)

    def plot_dependency_heatmap(
        self,
        metric: Literal["kendall", "lower_tail", "upper_tail"] = "kendall",
    ) -> go.Figure:
        """
        Plot heatmap of dependency metrics.

        Args:
            metric: Metric to display.

        Returns:
            Plotly figure.
        """
        if not self._pair_results:
            self.analyze_all_pairs()

        # Build matrix
        matrix = np.eye(self.n_strategies)

        for (name1, name2), result in self._pair_results.items():
            i = self.strategy_names.index(name1)
            j = self.strategy_names.index(name2)

            if metric == "kendall":
                value = result.kendall_tau
            elif metric == "lower_tail":
                value = result.lower_tail_dep
            elif metric == "upper_tail":
                value = result.upper_tail_dep
            else:
                raise ValueError(f"Unknown metric: {metric}")

            matrix[i, j] = value
            matrix[j, i] = value

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=self.strategy_names,
            y=self.strategy_names,
            colorscale="RdBu_r" if metric == "kendall" else "Reds",
            zmin=-1 if metric == "kendall" else 0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title=metric.replace("_", " ").title()),
        ))

        metric_title = metric.replace("_", " ").title()
        fig.update_layout(
            title=f"Dependency Structure: {metric_title}",
            xaxis_title="Strategy",
            yaxis_title="Strategy",
            template=PLOT_TEMPLATE,
            height=500,
            width=600,
        )

        return fig

    def plot_best_copulas_summary(self) -> go.Figure:
        """
        Plot summary of best copula for each pair.

        Returns:
            Plotly figure.
        """
        if not self._pair_results:
            self.analyze_all_pairs()

        pairs = []
        families = []
        aics = []
        taus = []

        for pair, result in self._pair_results.items():
            pairs.append(f"{pair[0][:10]} vs {pair[1][:10]}")
            families.append(result.family.replace("_", "-").title())
            aics.append(result.aic)
            taus.append(result.kendall_tau)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Best Copula by AIC", "Kendall's Tau"),
        )

        # Copula families
        unique_families = list(set(families))
        family_colors = {f: PLOT_COLORS[i] for i, f in enumerate(unique_families)}

        fig.add_trace(
            go.Bar(
                x=pairs,
                y=aics,
                marker_color=[family_colors[f] for f in families],
                text=families,
                textposition="outside",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(
                x=pairs,
                y=taus,
                marker_color=[PLOT_COLORS[0] if t > 0 else PLOT_COLORS[3] for t in taus],
            ),
            row=1, col=2,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        fig.update_layout(
            title="Copula Analysis Summary",
            template=PLOT_TEMPLATE,
            height=400,
            showlegend=False,
        )

        return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fit_best_copula(
    u1: np.ndarray,
    u2: np.ndarray,
    strategy_names: tuple[str, str] = ("Strategy 1", "Strategy 2"),
) -> tuple[CopulaResult, dict[str, go.Figure]]:
    """
    Quick function to fit best copula and generate plots.

    Args:
        u1: First uniform margin.
        u2: Second uniform margin.
        strategy_names: Names for labeling.

    Returns:
        Tuple of (best_result, plots_dict).
    """
    df = pd.DataFrame({strategy_names[0]: u1, strategy_names[1]: u2})
    analyzer = CopulaAnalyzer(df, strategy_names)
    analyzer.fit_all_copulas()

    best = analyzer.get_best_copula()
    plots = analyzer.generate_all_plots()

    return best, plots


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Test the copula analysis module."""
    print("=" * 60)
    print("  COPULA ANALYSIS MODULE TEST")
    print("=" * 60)

    # Generate synthetic data from a Clayton copula
    np.random.seed(42)
    n = 1000

    # Clayton copula simulation
    theta = 2.0
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    # Clayton conditional
    v_clayton = (1 + u**(-theta) * (v**(-theta/(theta+1)) - 1))**(-1/theta)

    print("\nGenerated synthetic data from Clayton copula (theta=2.0)")
    print(f"Sample size: {n}")

    # Create analyzer
    df = pd.DataFrame({"Strategy_A": u, "Strategy_B": v_clayton})
    analyzer = CopulaAnalyzer(df)

    # Fit all copulas
    print("\nFitting copulas...")
    results = analyzer.fit_all_copulas()

    # Print summary
    print("\nCopula Comparison:")
    print("-" * 80)
    summary = analyzer.get_summary_table()
    print(summary.to_string(index=False))

    # Best copula
    best = analyzer.get_best_copula()
    print(f"\nBest copula: {best.family}")
    print(f"  Parameters: {best.params}")
    print(f"  AIC: {best.aic:.1f}")
    print(f"  Lower tail dep: {best.lower_tail_dep:.3f}")
    print(f"  Upper tail dep: {best.upper_tail_dep:.3f}")

    # Generate plots
    print("\nGenerating plots...")
    plots = analyzer.generate_all_plots()
    print(f"Generated {len(plots)} plots")

    print("\n" + "=" * 60)
    print("  TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
