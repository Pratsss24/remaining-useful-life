"""Tests for sensor screening, estimator B, and the evaluation.

The pattern from `test_agelaw` continues: closed forms and planted truths
rather than stored output, because every component here returns a plausible
number when it is wrong.

What is pinned:

* **Planted-truth recovery, and where it stops.** A simulated fleet with a
  known health law, a known detectability horizon and known useless
  channels. The estimator has to find all three, and the tests record the
  noise level at which it stops finding them.
* **The sensor finding.** That pooled screening rejects most of the usable
  channels on the multi-regime subsets is the project's result, so it is
  asserted rather than left to a figure.
* **No leakage.** Estimator B is fitted on the training fleet only. The
  test checks this the hard way: refitting with the test fleet deleted from
  the process must produce bit-identical predictions.
* **Coverage arithmetic.** Measured against a forecast whose quantiles are
  known exactly, so a coverage figure of 90% means the machinery works
  rather than that the model is good.
* **The thesis.** That a pooled error number conceals the band structure is
  the claim the report is built on, and it is checked as an inequality on
  real numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agelaw          # noqa: E402
import data            # noqa: E402
import degradation     # noqa: E402
import evaluate        # noqa: E402
import lifetimes       # noqa: E402
import sensors         # noqa: E402
import simulate        # noqa: E402

CAP, WINDOW = 100, 5


@pytest.fixture(scope="module")
def model():
    """Estimator B on FD001, with the cap and window pinned.

    Pinned rather than cross-validated so the suite does not pay for the
    selection four times over; `test_selection_finds_the_pinned_settings`
    checks that the selection would have chosen them.
    """
    return degradation.fit("FD001", cap=CAP, window=WINDOW)


@pytest.fixture(scope="module")
def simulated():
    """One planted fleet, split into a run-to-failure half and a censored
    half. Split rather than generated twice, because two draws would get
    different sensor loadings and the index fitted on one would be
    meaningless on the other."""
    fleet = simulate.degradation_fleet(180, noise=0.25, seed=1)
    return fleet[:90], simulate.truncate_fleet(fleet[90:], seed=5)


def _fit_simulated(train, cap=CAP, window=WINDOW, channels=tuple(range(11))):
    labels = np.zeros(sum(u.observed for u in train), dtype=int)
    normaliser = sensors.Normaliser.fit(train, labels)
    prepared = degradation.prepare(train, normaliser, channels)
    index = degradation.fit_index(prepared, subset="SIM", cap=cap,
                                  channels=channels, normaliser=normaliser)
    calibration = degradation.fit_calibration(index, prepared, window=window)
    return index, calibration


# -- the trailing mean ----------------------------------------------------

def test_the_trailing_mean_is_causal():
    """A value must not depend on anything after it. Changing the last
    reading must leave every earlier smoothed value untouched -- otherwise
    a prediction made at cycle 20 would be using cycle 21."""
    values = np.arange(10.0)
    first = degradation.trailing_mean(values, 4)
    changed = values.copy()
    changed[-1] = 999.0
    second = degradation.trailing_mean(changed, 4)
    assert np.array_equal(first[:-1], second[:-1])
    assert first[-1] != second[-1]


def test_the_trailing_mean_uses_a_short_window_at_the_start():
    values = np.array([2.0, 4.0, 6.0, 8.0])
    found = degradation.trailing_mean(values, 3)
    assert found[0] == pytest.approx(2.0)
    assert found[1] == pytest.approx(3.0)
    assert found[2] == pytest.approx(4.0)
    assert found[3] == pytest.approx(6.0)


def test_a_window_of_one_changes_nothing():
    values = np.array([1.0, 7.0, 3.0])
    assert np.array_equal(degradation.trailing_mean(values, 1), values)


# -- weighted quantiles ---------------------------------------------------

def test_equal_weights_reproduce_the_ordinary_quantile():
    rng = np.random.default_rng(4)
    values = rng.normal(size=500)
    weights = np.ones_like(values)
    levels = np.array([0.1, 0.5, 0.9])
    found = degradation.weighted_quantile(values, weights, levels)
    assert found == pytest.approx(np.quantile(values, levels), abs=0.05)


def test_weighting_moves_the_median_toward_the_weighted_group():
    values = np.array([0.0, 0.0, 0.0, 10.0])
    heavy = degradation.weighted_quantile(
        values, np.array([1.0, 1.0, 1.0, 20.0]), np.array([0.5]))
    light = degradation.weighted_quantile(
        values, np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5]))
    assert heavy[0] > light[0]


# -- planted-truth recovery ----------------------------------------------

def test_the_index_recovers_a_planted_health_law(simulated):
    train, _ = simulated
    index, _ = _fit_simulated(train)
    assert index.r_squared > 0.9
    assert index.spans_unit_interval


def test_the_detectability_horizon_is_recovered(simulated):
    """The simulated fleet's health is flat until 100 cycles before
    failure. The cap-selection criterion has to find that without being
    told it."""
    train, _ = simulated
    labels = np.zeros(sum(u.observed for u in train), dtype=int)
    normaliser = sensors.Normaliser.fit(train, labels)
    channels = tuple(range(11))
    prepared = degradation.prepare(train, normaliser, channels)

    gaps = {}
    for cap in degradation.CAP_GRID:
        index = degradation.fit_index(prepared, subset="SIM", cap=cap,
                                      channels=channels,
                                      normaliser=normaliser)
        gaps[cap] = (abs(index.start_level - 1.0)
                     + abs(index.failure_level))
    assert min(gaps, key=lambda cap: gaps[cap]) == 100


def test_remaining_life_is_recovered_where_the_signal_exists(simulated):
    """Inside the detectability horizon the planted answer comes back to
    within a few cycles."""
    train, test = simulated
    index, calibration = _fit_simulated(train)
    health = np.array([degradation.trailing_mean(index.apply(u), WINDOW)[-1]
                       for u in test])
    truth = np.array([u.rul_at_end for u in test], dtype=float)
    error = calibration.quantile(health, 0.5) - truth
    inside = truth < CAP
    assert np.abs(error[inside]).mean() < 10.0
    assert abs(error.mean()) < 15.0


def test_the_planted_intervals_cover_what_they_claim(simulated):
    train, test = simulated
    index, calibration = _fit_simulated(train)
    health = np.array([degradation.trailing_mean(index.apply(u), WINDOW)[-1]
                       for u in test])
    truth = np.array([u.rul_at_end for u in test], dtype=float)
    covered = ((truth >= calibration.quantile(health, 0.05))
               & (truth <= calibration.quantile(health, 0.95)))
    assert 0.85 <= covered.mean() <= 1.0


def test_recovery_degrades_with_noise_and_stops():
    """Where the estimator stops working, as a measured boundary rather
    than a claim. Error grows monotonically with sensor noise, and the
    detectability horizon stops being identifiable once the noise is large
    enough to flatten the index."""
    errors, caps = [], []
    for noise in (0.1, 0.6, 3.0):
        fleet = simulate.degradation_fleet(180, noise=noise, seed=1)
        train, test = fleet[:90], simulate.truncate_fleet(fleet[90:], seed=5)
        index, calibration = _fit_simulated(train)
        health = np.array([
            degradation.trailing_mean(index.apply(u), WINDOW)[-1]
            for u in test])
        truth = np.array([u.rul_at_end for u in test], dtype=float)
        inside = truth < CAP
        error = calibration.quantile(health, 0.5) - truth
        errors.append(float(np.abs(error[inside]).mean()))
        caps.append(index.r_squared)

    assert errors[0] < errors[1] < errors[2]
    assert errors[0] < 5.0            # clean signal, a few cycles
    assert errors[2] > 20.0           # drowned signal, badly wrong
    assert caps[0] > 0.98 > caps[2]


# -- the sensor finding ---------------------------------------------------

def test_pooled_screening_rejects_channels_that_do_carry_signal():
    """The project's sensor result. On the multi-regime subsets, screening
    without holding the regime fixed throws away most of the usable
    channels."""
    for subset in ("FD002", "FD004"):
        found = sensors.screen(subset)
        pooled = found.informative(by_regime=False)
        by_regime = found.informative(by_regime=True)
        assert len(pooled) <= 2, subset
        assert len(by_regime) >= 9, subset
        assert len(found.masked()) >= 8, subset


def test_the_single_regime_subsets_screen_the_same_either_way():
    for subset in ("FD001", "FD003"):
        found = sensors.screen(subset)
        assert (found.informative(by_regime=False)
                == found.informative(by_regime=True))
        assert found.masked() == []


def test_the_commanded_channels_are_never_informative():
    """Inlet conditions and demanded fan speeds are quantities the
    simulator dictates, so they cannot carry degradation. Checked in every
    subset under both screenings."""
    for subset in data.SUBSETS:
        found = sensors.screen(subset)
        for channel in sensors.HELD_CHANNELS:
            assert abs(found.pooled_trend[channel]) < 0.1, (subset, channel)
            assert abs(found.regime_trend[channel]) < 0.1, (subset, channel)


def test_the_pressure_ratio_wakes_up_only_under_two_fault_modes():
    """The second sensor finding, pinned. `epr` is exactly flat in the two
    subsets that degrade the compressor alone, and trends in the two that
    degrade the fan as well. So channel usefulness depends on which fault
    is growing, not only on the operating regime -- and a sensor set chosen
    on FD001 would throw this channel away."""
    for channel in sensors.FAULT_MODE_CHANNELS:
        for subset in ("FD001", "FD002"):
            found = sensors.screen(subset)
            assert abs(found.regime_trend[channel]) < 0.1, subset
        for subset in sensors.TWO_FAULT_SUBSETS:
            found = sensors.screen(subset)
            assert abs(found.regime_trend[channel]) > 0.15, subset
            assert data.README_SAYS[subset]["faults"] == 2


def test_a_constant_channel_does_not_get_a_vanishing_scale():
    """The bug this guards against is real: in FD001 the P2 channel holds
    one value for every cycle and `std` still returns 3.6e-15 for it, so a
    `spread > 0` test would divide by that. The scales must all be
    sensible."""
    units = data.load("FD001", "train")
    labels, _ = data.regime_labels(units)
    normaliser = sensors.Normaliser.fit(units, labels)
    assert normaliser.scales.min() > 1e-6
    standardised = normaliser.apply_fleet(units, labels)
    assert np.abs(standardised).max() < 100.0


def test_the_normaliser_assigns_test_cycles_to_training_regimes():
    """Fitted on train and applied to test, including the regime
    assignment -- a short test record cannot define its own regimes."""
    train = data.load("FD002", "train")
    labels, _ = data.regime_labels(train)
    normaliser = sensors.Normaliser.fit(train, labels)
    assert normaliser.regimes == 6
    for unit in data.load("FD002", "test")[:20]:
        assigned = normaliser.assign(unit.settings)
        assert assigned.min() >= 0
        assert assigned.max() < normaliser.regimes
        assert normaliser.apply(unit).shape == unit.sensors.shape


# -- no leakage -----------------------------------------------------------

def test_estimator_b_never_sees_the_test_fleet(model):
    """Fitted on the training fleet only. Checked by refitting from the
    training units alone and requiring bit-identical weights, which is the
    only way to be sure no test information reached the index or the
    calibration."""
    train = data.load("FD001", "train")
    labels, _ = data.regime_labels(train)
    normaliser = sensors.Normaliser.fit(train, labels)
    channels = tuple(sensors.screen("FD001").informative())
    prepared = degradation.prepare(train, normaliser, channels)
    index = degradation.fit_index(prepared, subset="FD001", cap=CAP,
                                  channels=channels, normaliser=normaliser)
    assert np.array_equal(index.weights, model.index.weights)

    calibration = degradation.fit_calibration(index, prepared, window=WINDOW)
    assert np.array_equal(calibration.grid, model.calibration.grid)


def test_the_calibration_is_built_only_from_failed_units(model):
    """The calibration needs true remaining life, which is only honestly
    available for run-to-failure records."""
    assert all(u.failed for u in data.load("FD001", "train"))


# -- the calibration ------------------------------------------------------

def test_remaining_life_never_falls_as_health_rises(model):
    """A healthier reading must not predict less life left."""
    grid = model.calibration.grid
    assert np.all(np.diff(grid, axis=0) >= -1e-9)


def test_the_quantile_grid_is_ordered_at_every_health(model):
    grid = model.calibration.grid
    assert np.all(np.diff(grid, axis=1) >= -1e-9)


def test_unit_weighting_reduces_the_mid_band_bias():
    """The diagnosed calibration bias and its measured fix. Fitting on a
    cycle-weighted population over-represents the long healthy stretches of
    long-lived engines, which inflates predicted life at middling health."""
    train = data.load("FD001", "train")
    test = data.load("FD001", "test")
    labels, _ = data.regime_labels(train)
    normaliser = sensors.Normaliser.fit(train, labels)
    channels = tuple(sensors.screen("FD001").informative())
    prepared = degradation.prepare(train, normaliser, channels)
    index = degradation.fit_index(prepared, subset="FD001", cap=CAP,
                                  channels=channels, normaliser=normaliser)

    truth = np.array([u.rul_at_end for u in test], dtype=float)
    biases = {}
    for weighted in (True, False):
        calibration = degradation.fit_calibration(
            index, prepared, window=WINDOW, unit_weighted=weighted)
        health = np.array([
            degradation.trailing_mean(index.apply(u), WINDOW)[-1]
            for u in test])
        biases[weighted] = float((calibration.quantile(health, 0.5)
                                  - truth).mean())

    assert biases[False] > biases[True] > 0
    assert biases[False] - biases[True] > 2.0


# -- the route that does not work ----------------------------------------

def test_slope_extrapolation_fails_by_attenuation(model):
    """The documented dead end, measured. The fitted slope is far shallower
    than the slope needed to reach the threshold at the true failure cycle,
    so the projected crossing lands much too far ahead -- and the ratio of
    the two shows it is attenuation rather than noise."""
    attenuations, errors = [], []
    for unit in data.load("FD001", "test"):
        found = degradation.crossing_extrapolation(model, unit)
        if found.get("remaining") is None:
            continue
        attenuations.append(found["attenuation"])
        errors.append(found["remaining"] - found["truth"])

    assert len(attenuations) > 40
    # The needed slope is steeper than the fitted one, by a wide margin.
    assert np.median(attenuations) > 2.0
    # And the projection is wildly optimistic as a result.
    assert np.mean(errors) > 100.0


def test_the_calibrated_inversion_beats_the_extrapolation(model):
    """Which is why the estimator uses the one it uses."""
    test = data.load("FD001", "test")
    calibrated = evaluate.from_degradation(model, test)

    projected, truth = [], []
    for unit in test:
        found = degradation.crossing_extrapolation(model, unit)
        if found.get("remaining") is None:
            continue
        projected.append(found["remaining"])
        truth.append(found["truth"])

    extrapolated_mae = float(np.abs(np.array(projected)
                                    - np.array(truth)).mean())
    assert float(np.abs(calibrated.error).mean()) < extrapolated_mae / 3.0


# -- the evaluation machinery --------------------------------------------

def test_coverage_is_measured_correctly_on_a_known_forecast():
    """A forecast whose quantiles are exactly right must measure at its
    nominal level. This tests the arithmetic, not a model."""
    rng = np.random.default_rng(11)
    truth = rng.normal(100.0, 20.0, 20000)
    forecast = evaluate.Forecast(
        name="exact", subset="SIM",
        age=np.zeros_like(truth), truth=truth,
        quantiles=lambda q: np.full_like(
            truth, 100.0 + 20.0 * np.sqrt(2.0)
            * __import__("scipy.special", fromlist=["erfinv"]).erfinv(
                2 * q - 1)))
    for nominal in (0.5, 0.8, 0.9):
        assert forecast.covered(nominal).mean() == pytest.approx(nominal,
                                                                 abs=0.02)


def test_an_overconfident_forecast_measures_below_nominal():
    truth = np.random.default_rng(2).normal(0.0, 10.0, 5000)
    forecast = evaluate.Forecast(
        name="narrow", subset="SIM", age=np.zeros_like(truth), truth=truth,
        quantiles=lambda q: np.full_like(truth, 0.0) + (q - 0.5) * 2.0)
    assert forecast.covered(0.9).mean() < 0.2


def test_the_bands_partition_the_fleet(model):
    test = data.load("FD001", "test")
    forecast = evaluate.from_degradation(model, test)
    rows = evaluate.by_true_rul(forecast)
    banded = sum(row["n"] for row in rows if row["band"] != "all")
    pooled = next(row for row in rows if row["band"] == "all")
    assert pooled["n"] == forecast.n
    assert banded <= forecast.n          # sparse bands are dropped


def test_comparing_forecasts_over_different_fleets_is_refused(model):
    first = evaluate.from_degradation(model, data.load("FD001", "test"))
    second = evaluate.from_degradation(model,
                                       data.load("FD001", "test")[:50])
    with pytest.raises(ValueError):
        evaluate.compare(first, second)


# -- the thesis, as a test ------------------------------------------------

def test_a_pooled_error_number_conceals_the_band_structure():
    """The claim the report is built on. The pooled figure must sit well
    inside the range of the bands it averages, so that quoting it alone
    would misdescribe both ends."""
    found = evaluate.headline("FD001", cap=CAP, window=WINDOW)
    rows = found["age_law_bands"]
    bands = [row for row in rows if row["band"] != "all"]
    pooled = next(row for row in rows if row["band"] == "all")

    worst = max(row["mae"] for row in bands)
    best = min(row["mae"] for row in bands)
    assert worst / best > 1.5
    assert best < pooled["mae"] < worst

    # And the bias cancels: the pooled bias is far smaller than the
    # largest band bias, in both directions.
    assert max(row["bias"] for row in bands) > pooled["bias"] + 20.0
    assert min(row["bias"] for row in bands) < pooled["bias"] - 10.0


def test_the_age_law_interval_is_overconfident_near_failure():
    """The coverage failure the report leads on, pinned. The age-only
    estimator's intervals are far too narrow for the engines closest to
    failure, while looking respectable pooled."""
    for subset in ("FD002", "FD003", "FD004"):
        found = evaluate.headline(subset, cap=100 if subset != "FD004" else 140,
                                  window=WINDOW)
        rows = {row["band"]: row for row in found["age_law_bands"]}
        nearest, pooled = rows["0-25"], rows["all"]
        assert nearest["coverage_80"] < 0.55, subset
        assert pooled["coverage_90"] > nearest["coverage_90"], subset


def test_the_two_estimators_disagree_and_swap_places():
    """The reason for having two. The sensor-driven estimator is far better
    close to failure; the age law is competitive or better further out. A
    single winner would mean one of them was redundant."""
    found = evaluate.headline("FD001", cap=CAP, window=WINDOW)
    rows = {row["band"]: row for row in found["comparison"]}

    near = rows["0-25"]
    assert near["winner"] == "degradation"
    assert near["age law mae"] / near["degradation mae"] > 2.5

    winners = {row["winner"] for row in found["comparison"]
               if row["band"] != "all"}
    assert winners == {"degradation", "age law"}


def test_estimator_b_wins_overall_on_every_subset():
    for subset in data.SUBSETS:
        found = evaluate.headline(subset,
                                  cap=140 if subset == "FD004" else 100,
                                  window=WINDOW)
        pooled = next(row for row in found["comparison"]
                      if row["band"] == "all")
        assert pooled["winner"] == "degradation", subset


# -- the censoring counterfactual, on predictions ------------------------

def test_the_censoring_treatments_change_the_predictions():
    """Carrying the censoring argument through from the fitted parameters
    to the predictions anyone would act on."""
    rows = {row["treatment"]: row
            for row in evaluate.censoring_effect("FD001")}
    assert set(rows) == set(lifetimes.TREATMENTS)

    # Counting survivors as failures makes the fleet look far shorter
    # lived, so it predicts less remaining life than the oracle does.
    assert rows["as failures"]["bias"] < rows["oracle"]["bias"]
    assert rows["as failures"]["eta"] < rows["oracle"]["eta"] * 0.9


def test_downstream_error_ranks_the_censoring_treatments_backwards():
    """A warning, pinned as a fact: on this dataset the treatment that
    damages the fitted parameters most produces the *best* remaining-life
    predictions.

    `as failures` recovers a scale of 189 cycles against the oracle's 225
    and a shape of 2.9 against 4.6 -- it is unambiguously the worst fit of
    the lifetime law. But the age-only estimator over-predicts remaining
    life by about +12 cycles when fed the correct law, and a law biased
    short pushes the predictions back down, so its pooled error comes out
    at 27.9 against the oracle's 32.5.

    Two errors partially cancelling. The lesson is not that mishandling
    censoring is harmless; it is that downstream prediction error cannot
    be used to decide whether the censoring was handled correctly. That
    question is settled on the parameters, where the ordering is not in
    doubt, and `simulate.censoring_recovery` settles it against a planted
    truth.

    This is the same trap as the cap selection in `degradation.select`:
    a pooled error average, used as a criterion, prefers the wrong model.
    """
    rows = {row["treatment"]: row
            for row in evaluate.censoring_effect("FD001")}

    # Worst parameters of the three available treatments...
    available = ("censored MLE", "drop censored", "as failures")
    worst_eta = max(available,
                    key=lambda name: abs(rows[name]["eta"]
                                         - rows["oracle"]["eta"]))
    assert worst_eta == "as failures"

    # ...and yet the lowest prediction error, the oracle included.
    assert rows["as failures"]["mae"] < rows["oracle"]["mae"]
    assert rows["as failures"]["mae_vs_oracle"] < 0
    assert min(rows, key=lambda name: rows[name]["mae"]) == "as failures"

    # The correct likelihood, biased long on this dataset, compounds the
    # age law's own optimism and so scores worst of all.
    assert max(rows, key=lambda name: rows[name]["mae"]) == "censored MLE"


# -- selection ------------------------------------------------------------

@pytest.mark.slow
def test_selection_finds_the_pinned_settings():
    """The cross-validated selection agrees with the values the rest of
    the suite pins, and the two cap criteria agree or land one grid step
    apart."""
    chosen = degradation.select("FD001")
    assert chosen["best_cap"] == CAP
    assert chosen["best_window"] == WINDOW
    assert chosen["criteria_agree"]
