"""Estimator A: remaining life from the fleet's lifetime law, age only.

This estimator is told how long an engine has been running and nothing
else. No sensor ever reaches it. It fits a lifetime distribution to the
fleet, and then answers "how much longer?" by conditioning that
distribution on survival to the engine's current age.

It is deliberately the weaker of the two estimators in this project, and
deliberately the one that cannot be accused of leaking anything. Its
predictions depend on a single number per engine, so if the sensor-driven
estimator in `degradation` cannot beat it, the sensors are not carrying
usable degradation signal — which makes this the floor the whole evaluation
is measured against rather than a straw man.

**The lifetime law.** A two-parameter Weibull,

    S(t) = exp( -(t / eta) ** beta )

whose shape parameter is a claim about the physics and not a curve-fitting
nuisance:

    beta < 1   hazard falls with age: infant mortality
    beta = 1   hazard is constant: failures arrive without memory of age
    beta > 1   hazard rises with age: wear-out

For a fleet of engines degrading through a growing fault, beta should come
out well above 1. If it does not, either the model or the data handling is
wrong, and that is worth knowing before any RUL number is quoted.

**The censored likelihood.** With right-censored observations,

    log L = sum over failures [ ln(beta/eta) + (beta-1) ln(t/eta) ]
          - sum over ALL units [ (t / eta) ** beta ]

The first sum runs over the engines that failed; the second runs over
every engine, because a survivor contributes its survival probability
rather than a density. That second sum is the whole of how "still running
at cycle 134" gets used.

**How it is solved.** For any fixed beta the scale has a closed form,

    eta(beta) = [ sum over all t ** beta / d ] ** (1 / beta)

with d the number of failures. Substituting it back leaves one equation in
beta alone,

    1/beta + (1/d) sum over failures ln t
           - [ sum over all t**beta ln t ] / [ sum over all t**beta ]  =  0

monotone decreasing in beta, so it has exactly one root and bisection
finds it without the wandering a general optimiser can do. At beta = 1 the
scale equation collapses to eta = (total time on test) / d, the textbook
exponential estimator, which is what the tests pin the general fitter
against.

**From a lifetime law to a remaining-life prediction.** Conditioning on
survival to age a,

    P(T > a + r | T > a) = exp( (a/eta)**beta - ((a+r)/eta)**beta )

which inverts in closed form, so the q-quantile of remaining life is

    r_q(a) = eta * [ (a/eta)**beta - ln(1-q) ] ** (1/beta)  -  a

That single expression gives both the point prediction (q = 0.5, the median
residual life) and the prediction interval (q = 0.05 and 0.95), with no
simulation anywhere. The mean residual life also has a closed form, through
the upper incomplete gamma function,

    MRL(a) = (eta/beta) * Gamma(1/beta, (a/eta)**beta) * exp((a/eta)**beta)

and at a = 0 it reduces to eta * Gamma(1 + 1/beta), the ordinary Weibull
mean, which the tests use as an anchor.

**What this estimator structurally cannot do.** Two engines of the same age
get the same prediction, always. Real engines of the same age differ -- in
this dataset lifetimes at one subset range over a factor of three -- so the
error of this estimator has an irreducible floor set by the spread of the
lifetime distribution itself, and no amount of fitting reduces it. Quoting
a single RMSE for it would hide exactly that. `evaluate` therefore reports
error against true remaining life rather than pooled, and the interval
coverage is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class AgeLaw:
    """A fitted Weibull lifetime law, with remaining-life prediction."""

    name: str
    beta: float               # shape
    eta: float                # scale, in cycles
    n: int
    failures: int
    log_likelihood: float
    beta_se: float = float("nan")
    eta_se: float = float("nan")
    method: str = "censored MLE"

    # -- the distribution -------------------------------------------------

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-np.power(np.maximum(t, 0.0) / self.eta, self.beta))

    def hazard(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-12)
        return (self.beta / self.eta) * np.power(t / self.eta,
                                                 self.beta - 1.0)

    def quantile(self, p: float) -> float:
        """Lifetime by which a fraction p of the fleet has failed."""
        return float(self.eta * (-np.log1p(-p)) ** (1.0 / self.beta))

    @property
    def mean_life(self) -> float:
        return float(self.eta * special.gamma(1.0 + 1.0 / self.beta))

    @property
    def median_life(self) -> float:
        return self.quantile(0.5)

    @property
    def b10_life(self) -> float:
        """Cycle by which a tenth of the fleet has failed."""
        return self.quantile(0.10)

    @property
    def censoring_rate(self) -> float:
        return 1.0 - self.failures / self.n if self.n else float("nan")

    @property
    def regime(self) -> str:
        low, high = self.beta_confidence()
        if not np.isfinite(low):
            return "wear-out" if self.beta > 1 else "infant mortality"
        if high < 1.0:
            return "infant mortality"
        if low > 1.0:
            return "wear-out"
        return "indistinguishable from constant hazard"

    def beta_confidence(self, alpha: float = 0.05) -> tuple[float, float]:
        """Interval for beta, built on the log scale.

        beta is positive and its sampling distribution is skewed, so the
        interval is formed for ln beta and transformed back; a symmetric
        interval on beta itself can reach below zero.
        """
        if not np.isfinite(self.beta_se) or self.beta_se <= 0:
            return (float("nan"), float("nan"))
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        spread = z * self.beta_se / self.beta
        return (self.beta * np.exp(-spread), self.beta * np.exp(spread))

    # -- remaining life ---------------------------------------------------

    def conditional_survival(self, age, remaining):
        """P(survives `remaining` more cycles | alive at `age`)."""
        age = np.asarray(age, dtype=float)
        remaining = np.asarray(remaining, dtype=float)
        scaled_age = np.power(np.maximum(age, 0.0) / self.eta, self.beta)
        scaled_end = np.power(np.maximum(age + remaining, 0.0) / self.eta,
                              self.beta)
        return np.exp(scaled_age - scaled_end)

    def rul_quantile(self, age, q: float):
        """The q-quantile of remaining life at `age`, in closed form.

        q = 0.5 is the median residual life, the point prediction. q = 0.05
        and 0.95 bracket a 90% prediction interval. Because -ln(1-q) is
        positive for every q in (0, 1), the bracket is always valid and
        never needs clipping at zero -- which matters, since an estimator
        that silently clips reports better coverage than it earns.
        """
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie strictly between 0 and 1")
        age = np.asarray(age, dtype=float)
        scaled = np.power(np.maximum(age, 0.0) / self.eta, self.beta)
        end = self.eta * np.power(scaled - np.log1p(-q), 1.0 / self.beta)
        return end - age

    def mean_residual_life(self, age):
        """E[T - a | T > a], through the upper incomplete gamma.

        `gammaincc` is the regularised upper incomplete gamma, so the
        unregularised one is recovered by multiplying by Gamma(1/beta).
        At a = 0 this reduces to eta * Gamma(1 + 1/beta) exactly.
        """
        age = np.asarray(age, dtype=float)
        shape = 1.0 / self.beta
        scaled = np.power(np.maximum(age, 0.0) / self.eta, self.beta)
        upper = special.gamma(shape) * special.gammaincc(shape, scaled)
        return (self.eta / self.beta) * upper * np.exp(scaled)

    def predict(self, age, *, alpha: float = 0.10) -> dict:
        """Remaining life with a (1 - alpha) prediction interval.

        The point prediction is the *median* residual life rather than the
        mean. For a right-skewed lifetime distribution the mean sits above
        the median, and since this estimator is judged on absolute error
        the median is the quantity that minimises it. Both are returned, so
        the choice is visible rather than buried.
        """
        age = np.asarray(age, dtype=float)
        return {
            "age": age,
            "median": self.rul_quantile(age, 0.5),
            "mean": self.mean_residual_life(age),
            "lower": self.rul_quantile(age, alpha / 2.0),
            "upper": self.rul_quantile(age, 1.0 - alpha / 2.0),
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
        }


# -- fitting --------------------------------------------------------------

def log_likelihood(time, failed, beta: float, eta: float) -> float:
    """Right-censored Weibull log-likelihood."""
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    if beta <= 0 or eta <= 0:
        return -np.inf
    scaled = np.power(time / eta, beta)
    event = np.sum(np.log(beta / eta)
                   + (beta - 1.0) * np.log(time[failed] / eta))
    return float(event - np.sum(scaled))


def _scale_for(time: np.ndarray, failures: int, beta: float) -> float:
    return float((np.sum(np.power(time, beta)) / failures) ** (1.0 / beta))


def _shape_equation(beta: float, time: np.ndarray, log_failed_sum: float,
                    failures: int) -> float:
    powered = np.power(time, beta)
    return (1.0 / beta + log_failed_sum / failures
            - np.sum(powered * np.log(time)) / np.sum(powered))


def fit(time, failed, *, name: str = "", floor: float = 1e-6) -> AgeLaw:
    """Maximum likelihood with right censoring.

    `floor` keeps a lifetime of exactly zero out of the logarithms. No
    C-MAPSS engine runs for zero cycles, but a caller passing simulated
    data should not get a silent NaN.
    """
    time = np.maximum(np.asarray(time, dtype=float), floor)
    failed = np.asarray(failed, dtype=bool)
    failures = int(failed.sum())
    if failures < 2:
        raise ValueError("need at least two failures to identify a shape")

    log_failed_sum = float(np.sum(np.log(time[failed])))
    args = (time, log_failed_sum, failures)

    low, high = 1e-3, 1.0
    while _shape_equation(high, *args) > 0 and high < 1e4:
        low, high = high, high * 2.0
    beta = float(optimize.brentq(_shape_equation, low, high, args=args,
                                 xtol=1e-12, rtol=1e-14))
    eta = _scale_for(time, failures, beta)

    beta_se, eta_se = _standard_errors(time, failed, beta, eta)
    return AgeLaw(name=name, beta=beta, eta=eta, n=int(time.size),
                  failures=failures,
                  log_likelihood=log_likelihood(time, failed, beta, eta),
                  beta_se=beta_se, eta_se=eta_se)


def fit_exponential(time, failed, *, name: str = "") -> AgeLaw:
    """The beta = 1 case, in closed form and with no iteration.

    eta is the total time on test over the number of failures. This is the
    anchor the general fitter is checked against, since it involves no
    numerics at all.
    """
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    failures = int(failed.sum())
    if failures == 0:
        raise ValueError("no failures")
    eta = float(np.sum(time) / failures)
    return AgeLaw(name=name, beta=1.0, eta=eta, n=int(time.size),
                  failures=failures,
                  log_likelihood=log_likelihood(time, failed, 1.0, eta),
                  method="exponential, closed form")


def _standard_errors(time, failed, beta: float,
                     eta: float) -> tuple[float, float]:
    """From the observed information, by numerical second derivatives.

    The analytic Hessian exists, but the numerical one is taken from the
    same likelihood the fit maximises, so it cannot drift out of agreement
    with the fit if the model changes.
    """
    step_beta = max(beta * 1e-4, 1e-7)
    step_eta = max(eta * 1e-4, 1e-7)

    def value(b, e):
        return log_likelihood(time, failed, b, e)

    centre = value(beta, eta)
    d_bb = (value(beta + step_beta, eta) - 2 * centre
            + value(beta - step_beta, eta)) / step_beta ** 2
    d_ee = (value(beta, eta + step_eta) - 2 * centre
            + value(beta, eta - step_eta)) / step_eta ** 2
    d_be = ((value(beta + step_beta, eta + step_eta)
             - value(beta + step_beta, eta - step_eta)
             - value(beta - step_beta, eta + step_eta)
             + value(beta - step_beta, eta - step_eta))
            / (4 * step_beta * step_eta))

    information = -np.array([[d_bb, d_be], [d_be, d_ee]])
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    if covariance[0, 0] <= 0 or covariance[1, 1] <= 0:
        return float("nan"), float("nan")
    return float(np.sqrt(covariance[0, 0])), float(np.sqrt(covariance[1, 1]))


def shape_test(time, failed) -> dict:
    """Is beta different from 1 by more than sampling noise?

    A likelihood ratio against the exponential, which is this model with
    beta pinned at one. Two nested models, one degree of freedom.
    """
    full = fit(time, failed)
    reduced = fit_exponential(time, failed)
    statistic = 2.0 * (full.log_likelihood - reduced.log_likelihood)
    return {
        "beta": full.beta,
        "statistic": float(statistic),
        "p_value": float(stats.chi2.sf(max(statistic, 0.0), 1)),
        "log_likelihood_full": full.log_likelihood,
        "log_likelihood_exponential": reduced.log_likelihood,
    }


def fit_treatments(treated: dict) -> dict[str, AgeLaw]:
    """Fit the same fleet under each censoring treatment.

    Takes the output of `lifetimes.treatments`. The `oracle` entry is the
    target, not a method.
    """
    return {label: fit(time, failed, name=label)
            for label, (time, failed) in treated.items()}
