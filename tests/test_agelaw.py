"""Tests for the age-law estimator and the censoring argument.

A survival fitter is easy to write and easy to get subtly wrong, and a
wrong one returns a plausible curve rather than an error. So almost nothing
here compares against stored output. Instead:

* **Closed forms.** With the shape pinned at 1 the Weibull scale is the
  total time on test over the number of failures, with no iteration at all;
  that exact answer pins the general fitter. Mean residual life at age zero
  must equal the ordinary Weibull mean. The remaining-life quantile must
  invert the conditional survivor function exactly.
* **Hand-checkable examples.** The small Kaplan-Meier worked below can be
  done on paper.
* **Recovery of a planted truth**, under each of three censoring schemes,
  with the direction of each failure asserted rather than just its size.
  These are the tests behind the report's censoring claims.
* **Agreement between independent routes to the same number.** Parametric
  mean residual life against the same quantity read off a Kaplan-Meier
  curve; the fitted survivor function against the non-parametric one.
* **No leakage.** The estimator must give an identical answer whether or
  not the true lifetimes are present in the records it is handed. That is
  the test that stops `truth` quietly becoming an input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import special

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agelaw       # noqa: E402
import data         # noqa: E402
import lifetimes    # noqa: E402
import simulate     # noqa: E402
import survival     # noqa: E402


@pytest.fixture(scope="module")
def lives():
    return lifetimes.pooled("FD001")


@pytest.fixture(scope="module")
def fitted(lives):
    time, failed = lifetimes.arrays(lives)
    return agelaw.fit(time, failed, name="FD001 pooled")


# -- closed forms ---------------------------------------------------------

def test_the_exponential_scale_is_total_time_over_failures():
    """No iteration involved: this one can be done by hand."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    failed = np.array([True, False, True, False, True])
    fit = agelaw.fit_exponential(time, failed)
    assert fit.eta == pytest.approx(15.0 / 3.0)
    assert fit.beta == 1.0


def test_the_general_fitter_reproduces_the_exponential_when_the_data_is_one():
    """Data generated with shape 1 must pull the free shape back to 1, and
    the scale back to the closed-form answer."""
    rng = np.random.default_rng(7)
    time = rng.exponential(120.0, 60000)
    failed = np.ones_like(time, dtype=bool)
    free = agelaw.fit(time, failed)
    closed = agelaw.fit_exponential(time, failed)
    assert free.beta == pytest.approx(1.0, abs=0.02)
    assert free.eta == pytest.approx(closed.eta, rel=0.02)


def test_mean_residual_life_at_age_zero_is_the_weibull_mean():
    """MRL(0) = eta * Gamma(1 + 1/beta), independently computed."""
    law = agelaw.AgeLaw(name="", beta=3.3, eta=210.0, n=10, failures=10,
                        log_likelihood=0.0)
    expected = 210.0 * special.gamma(1.0 + 1.0 / 3.3)
    assert float(law.mean_residual_life(0.0)) == pytest.approx(expected,
                                                               rel=1e-10)
    assert law.mean_life == pytest.approx(expected, rel=1e-12)


def test_the_remaining_life_quantile_inverts_conditional_survival():
    """r_q(a) is defined by S(a+r)/S(a) = 1-q, so putting it back must
    return 1-q. Checked across ages and quantiles because this is the one
    expression both the point prediction and the interval come from."""
    law = agelaw.AgeLaw(name="", beta=4.2, eta=230.0, n=10, failures=10,
                        log_likelihood=0.0)
    for age in (0.0, 50.0, 150.0, 300.0, 600.0):
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            r = law.rul_quantile(age, q)
            back = float(law.conditional_survival(age, r))
            assert back == pytest.approx(1.0 - q, rel=1e-9)


def test_remaining_life_quantiles_are_ordered_and_positive():
    law = agelaw.AgeLaw(name="", beta=4.2, eta=230.0, n=10, failures=10,
                        log_likelihood=0.0)
    ages = np.array([0.0, 100.0, 250.0, 500.0])
    low = law.rul_quantile(ages, 0.05)
    mid = law.rul_quantile(ages, 0.5)
    high = law.rul_quantile(ages, 0.95)
    assert np.all(low > 0)
    assert np.all(low < mid) and np.all(mid < high)


def test_a_wear_out_law_has_shrinking_remaining_life():
    """With beta > 1 the hazard rises, so an older engine must have less
    life left, not more. The property that makes the estimator a model of
    wear rather than of chance."""
    law = agelaw.AgeLaw(name="", beta=4.5, eta=225.0, n=10, failures=10,
                        log_likelihood=0.0)
    ages = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    assert np.all(np.diff(law.mean_residual_life(ages)) < 0)
    assert np.all(np.diff(law.rul_quantile(ages, 0.5)) < 0)


def test_a_constant_hazard_law_forgets_its_age():
    """beta = 1 is the memoryless case: remaining life must not depend on
    age at all. A fitter that drifts here has the conditioning wrong."""
    law = agelaw.AgeLaw(name="", beta=1.0, eta=180.0, n=10, failures=10,
                        log_likelihood=0.0)
    ages = np.array([0.0, 100.0, 500.0, 2000.0])
    assert law.mean_residual_life(ages) == pytest.approx(180.0, rel=1e-9)
    assert law.rul_quantile(ages, 0.5) == pytest.approx(
        180.0 * np.log(2.0), rel=1e-9)


def test_the_censored_likelihood_uses_the_survivors():
    """The direct check that a censored observation is not simply ignored:
    moving one survivor's censoring time must change the likelihood."""
    time = np.array([100.0, 150.0, 200.0, 250.0])
    failed = np.array([True, True, False, False])
    near = agelaw.log_likelihood(time, failed, 2.0, 200.0)
    far = agelaw.log_likelihood(np.array([100.0, 150.0, 200.0, 900.0]),
                                failed, 2.0, 200.0)
    assert near != far


# -- Kaplan-Meier, by hand ------------------------------------------------

def test_kaplan_meier_on_six_observations():
    """Six units at 10, 20, 25, 30, 40, 50; failures at 10 and 30, the
    rest censored. By hand:

        t=10   risk set is all six, one fails          S = 1 - 1/6 = 5/6
        t=30   the units censored at 20 and 25 have left, so the risk set
               is {30, 40, 50}, three units, one fails
                                            S = 5/6 * (1 - 1/3) = 5/9

    The censored units are the reason the second step is a third and not a
    sixth: they contribute their time at risk and then leave without a
    death."""
    time = np.array([10.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    failed = np.array([True, False, False, True, False, False])
    curve = survival.kaplan_meier(time, failed)
    assert np.array_equal(curve.times, np.array([10.0, 30.0]))
    assert curve.at_risk.tolist() == [6.0, 3.0]
    assert curve.values[0] == pytest.approx(5.0 / 6.0)
    assert curve.values[1] == pytest.approx(5.0 / 9.0)


def test_kaplan_meier_without_censoring_is_the_empirical_survivor():
    rng = np.random.default_rng(3)
    time = rng.integers(1, 60, 400).astype(float)
    failed = np.ones_like(time, dtype=bool)
    curve = survival.kaplan_meier(time, failed)
    for t in (5.0, 20.0, 40.0, 55.0):
        assert curve.at(t) == pytest.approx(np.mean(time > t), abs=1e-12)


def test_the_confidence_band_stays_inside_zero_and_one(fitted, lives):
    time, failed = lifetimes.arrays(lives)
    curve = survival.kaplan_meier(time, failed)
    assert np.all(curve.lower >= 0.0) and np.all(curve.lower <= 1.0)
    assert np.all(curve.upper >= 0.0) and np.all(curve.upper <= 1.0)
    assert np.all(curve.lower <= curve.values + 1e-12)
    assert np.all(curve.upper >= curve.values - 1e-12)


# -- the two routes to mean residual life ---------------------------------

def test_parametric_and_nonparametric_residual_life_agree(lives, fitted):
    """Independent routes to one number: closed form from the fitted
    Weibull, against a trapezoid over the Kaplan-Meier curve. They are
    compared only where the curve is still informative, and the
    non-parametric one is a lower bound wherever it is truncated."""
    time, failed = lifetimes.arrays(lives)
    curve = survival.kaplan_meier(time, failed)
    for age in (140.0, 170.0, 200.0):
        empirical = survival.residual_life(curve, age)
        modelled = float(fitted.mean_residual_life(age))
        if empirical["lower_bound"]:
            assert modelled > empirical["mean"]
        else:
            assert modelled == pytest.approx(empirical["mean"], rel=0.35)


def test_the_nonparametric_estimate_reports_its_own_truncation(lives):
    """It must say when it has had to leave part of the integral out,
    rather than return a quietly short answer."""
    time, failed = lifetimes.arrays(lives)
    curve = survival.kaplan_meier(time, failed)
    far = survival.residual_life(curve, curve.last_event - 1.0)
    assert far["truncated_at"] == curve.last_event


# -- recovery of a planted truth ------------------------------------------

@pytest.fixture(scope="module")
def recovery():
    rows = simulate.censoring_recovery(n=40000, seed=0)
    return {(row["scheme"], row["treatment"]): row for row in rows}


def test_the_oracle_recovers_the_planted_truth(recovery):
    """The sanity check on the whole simulation: given the real lifetimes,
    the fitter must return the parameters they were generated from."""
    for scheme in simulate.SCHEMES:
        row = recovery[(scheme, "oracle")]
        assert abs(row["beta_error"]) < 0.01
        assert abs(row["eta_error"]) < 0.01


def test_under_independent_censoring_the_mle_is_right_and_the_rest_are_short():
    """The textbook case, and the one the likelihood is built for."""
    rows = {r["treatment"]: r for r in simulate.censoring_recovery(n=40000)
            if r["scheme"] == "independent"}
    assert abs(rows["censored MLE"]["beta_error"]) < 0.02
    assert abs(rows["censored MLE"]["eta_error"]) < 0.02
    for label in ("drop censored", "as failures"):
        assert rows[label]["eta_error"] < -0.02, label
        assert rows[label]["beta_error"] < -0.02, label


def test_lifetime_proportional_censoring_biases_the_mle_long(recovery):
    """The C-MAPSS case. The censoring time knows the lifetime, which the
    likelihood assumes it cannot, so the MLE is no longer consistent -- and
    it errs long, which is the direction that matters because it makes the
    fleet look healthier than it is."""
    mle = recovery[("proportional", "censored MLE")]
    assert mle["beta_error"] > 0.03
    assert mle["eta_error"] > 0.02


def test_lifetime_proportional_censoring_leaves_dropping_them_unbiased(
        recovery):
    """Why the real data shows what it shows: with the censored units a
    random subsample, discarding them costs precision and nothing else."""
    dropped = recovery[("proportional", "drop censored")]
    assert abs(dropped["beta_error"]) < 0.02
    assert abs(dropped["eta_error"]) < 0.02


def test_at_a_common_cutoff_dropping_the_censored_units_is_catastrophic(
        recovery):
    """And why that is no licence to do it. Under the censoring an operator
    actually faces, the censored units are the long-lived tail, and
    discarding them inflates the shape badly while the MLE holds."""
    mle = recovery[("end_of_test", "censored MLE")]
    dropped = recovery[("end_of_test", "drop censored")]
    assert abs(mle["beta_error"]) < 0.02
    assert abs(mle["eta_error"]) < 0.02
    assert dropped["beta_error"] > 0.30
    assert dropped["eta_error"] < -0.15


def test_counting_survivors_as_failures_is_never_acceptable(recovery):
    """The one treatment with no scheme in which it is defensible."""
    for scheme in simulate.SCHEMES:
        row = recovery[(scheme, "as failures")]
        assert abs(row["beta_error"]) > 0.05 or abs(row["eta_error"]) > 0.05


def test_the_mle_is_the_only_treatment_never_badly_wrong(recovery):
    """The actual conclusion of the censoring section, as a test."""
    worst = {}
    for treatment in ("censored MLE", "drop censored", "as failures"):
        worst[treatment] = max(
            max(abs(recovery[(scheme, treatment)]["beta_error"]),
                abs(recovery[(scheme, treatment)]["eta_error"]))
            for scheme in simulate.SCHEMES)
    assert worst["censored MLE"] < 0.10
    assert worst["drop censored"] > 0.30
    assert worst["as failures"] > 0.30


# -- the real fleet -------------------------------------------------------

def test_the_fleet_is_in_wear_out(fitted):
    """Engines degrading through a growing fault must show a rising
    hazard. If this came back near 1, something upstream is broken."""
    low, high = fitted.beta_confidence()
    assert fitted.beta > 1.0
    assert low > 1.0
    assert fitted.regime == "wear-out"


def test_the_shape_is_distinguishable_from_constant_hazard(lives):
    time, failed = lifetimes.arrays(lives)
    test = agelaw.shape_test(time, failed)
    assert test["p_value"] < 1e-6


def test_the_real_censoring_is_measurably_dependent(lives):
    """The diagnosis behind the censoring section, on the real data."""
    found = simulate.measured_censoring_dependence(lives)
    assert found["correlation"] > 0.5
    assert found["p_value"] < 1e-6
    assert 0.3 < found["fraction_mean"] < 0.9


def test_every_subset_shows_the_same_dependence():
    for subset in data.SUBSETS:
        found = simulate.measured_censoring_dependence(
            lifetimes.pooled(subset))
        assert found["correlation"] > 0.5, subset


# -- no leakage -----------------------------------------------------------

def test_the_estimator_cannot_see_the_true_lifetimes(lives):
    """`truth` is for evaluation only. Blanking it must not move the fit by
    so much as a float, which is the only way to be sure it is not an
    input."""
    time, failed = lifetimes.arrays(lives)
    honest = agelaw.fit(time, failed)

    blinded = [lifetimes.Life(subset=one.subset, split=one.split,
                              unit=one.unit, observed=one.observed,
                              failed=one.failed, truth=float("nan"))
               for one in lives]
    time_blind, failed_blind = lifetimes.arrays(blinded)
    again = agelaw.fit(time_blind, failed_blind)

    assert again.beta == honest.beta
    assert again.eta == honest.eta


def test_the_censored_units_contribute_only_a_lower_bound(lives):
    """A censored record must enter as its observed cycles, never as its
    true failure cycle -- the mistake that would make the oracle and the
    MLE agree for the wrong reason."""
    treated = lifetimes.treatments(lives)
    time, _ = treated["censored MLE"]
    truth = lifetimes.truth(lives)
    censored = ~np.array([one.failed for one in lives], dtype=bool)
    assert np.all(time[censored] < truth[censored])
    oracle_time, _ = treated["oracle"]
    assert np.array_equal(oracle_time, truth)
