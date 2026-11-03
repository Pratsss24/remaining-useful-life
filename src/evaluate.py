"""Scoring remaining-life predictions honestly.

A single RMSE over a test fleet is close to meaningless for remaining-life
prediction, and this module exists to avoid quoting one.

**Why one number will not do.** The difficulty of the question depends
entirely on how far ahead it is being asked. Predicting that an engine with
twelve cycles left has about twelve cycles left is a different task from
saying the same about one with two hundred, and a pooled average over a
fleet silently weights those by however many of each the test set happens
to contain. Worse, the two errors point in opposite directions and cancel:
on FD001 the age-only estimator over-predicts by 35 cycles on the engines
closest to failure and under-predicts on the ones furthest from it, and the
pooled bias comes out at +12, which describes neither group.

So every result here is reported against the true remaining life, in bands.
The pooled figure is computed too, and printed beside the bands rather than
instead of them, because its purpose is to show how much it conceals.

**Coverage is measured, not asserted.** A prediction interval is a claim
with a number attached: a 90% interval should contain the truth 90% of the
time. Whether it does is an empirical question, and the answer is reported
as found. If a nominal 90% interval covers 70% of the cases in some band,
that is what appears, with the band it happened in.

The nominal levels are fixed in advance -- `degradation.LEVELS` stores a
whole quantile grid for exactly this reason -- so the level can never be
chosen after seeing which one covered well.

**Asymmetry matters more than magnitude.** Over-predicting remaining life
means an engine is left on the wing longer than it should be; under-
predicting means it comes off early. These are not equally bad, and an
absolute error treats them as though they were. `late_share` reports the
fraction of predictions that were optimistic, because for maintenance that
is the number with consequences attached.

Nothing in here is specific to either estimator. Both are wrapped as a
`Forecast`, which is a set of predicted quantiles and the truth, and
everything else is derived from that -- so the two are scored by identical
code and any difference between them is a difference in the estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import agelaw
import data
import degradation
import lifetimes

# Bands of true remaining life, in cycles. The first is the one that
# matters operationally and the last is where no sensor says anything.
BANDS = (0, 25, 50, 75, 100, 125, 150, 10_000)
MIN_IN_BAND = 5

# Nominal levels at which coverage is measured. Fixed here, before any
# coverage is computed.
NOMINAL = (0.50, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class Forecast:
    """Predicted quantiles of remaining life, with the truth beside them.

    `quantiles` maps a level to one predicted value per unit. Holding the
    whole quantile function rather than a single interval is what lets
    coverage be measured at several nominal levels from one object.
    """

    name: str
    subset: str
    age: np.ndarray
    truth: np.ndarray
    quantiles: Callable[[float], np.ndarray]
    declined: np.ndarray | None = None      # unit cannot be predicted

    @property
    def median(self) -> np.ndarray:
        return np.asarray(self.quantiles(0.5), dtype=float)

    @property
    def error(self) -> np.ndarray:
        """Predicted minus true. Positive is optimistic."""
        return self.median - self.truth

    def interval(self, nominal: float) -> tuple[np.ndarray, np.ndarray]:
        alpha = 1.0 - nominal
        return (np.asarray(self.quantiles(alpha / 2.0), dtype=float),
                np.asarray(self.quantiles(1.0 - alpha / 2.0), dtype=float))

    def covered(self, nominal: float) -> np.ndarray:
        low, high = self.interval(nominal)
        return (self.truth >= low) & (self.truth <= high)

    @property
    def n(self) -> int:
        return int(self.truth.size)


# -- building forecasts ---------------------------------------------------

def from_agelaw(law: agelaw.AgeLaw, units: list[data.Unit], *,
                subset: str = "") -> Forecast:
    """Estimator A as a forecast: conditional quantiles at each age."""
    age = np.array([u.observed for u in units], dtype=float)
    truth = np.array([u.rul_at_end for u in units], dtype=float)
    return Forecast(name="age law", subset=subset or units[0].subset,
                    age=age, truth=truth,
                    quantiles=lambda q: law.rul_quantile(age, q))


def from_degradation(model: degradation.DegradationModel,
                     units: list[data.Unit], *,
                     subset: str = "") -> Forecast:
    """Estimator B as a forecast: calibrated quantiles at each health."""
    age = np.array([u.observed for u in units], dtype=float)
    truth = np.array([u.rul_at_end for u in units], dtype=float)
    health = np.array([model.health(u) for u in units], dtype=float)
    ceiling = health >= model.calibration.ceiling
    return Forecast(name="degradation", subset=subset or units[0].subset,
                    age=age, truth=truth,
                    quantiles=lambda q: model.calibration.quantile(health, q),
                    declined=ceiling)


# -- scoring --------------------------------------------------------------

def _row(label: str, error: np.ndarray, truth: np.ndarray,
         coverage: dict) -> dict:
    return {
        "band": label,
        "n": int(error.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "bias": float(error.mean()),
        "late_share": float((error > 0).mean()),
        "median_truth": float(np.median(truth)),
        **{f"coverage_{int(100 * level)}": value
           for level, value in coverage.items()},
    }


def by_true_rul(forecast: Forecast, *, bands=BANDS,
                nominal=NOMINAL) -> list[dict]:
    """Error and measured coverage, banded by true remaining life.

    The central table of the project. A band with fewer than `MIN_IN_BAND`
    units is dropped rather than reported, because a coverage figure from
    three units is noise with a percentage sign on it.
    """
    error = forecast.error
    rows = []
    for low, high in zip(bands[:-1], bands[1:]):
        mask = (forecast.truth >= low) & (forecast.truth < high)
        if mask.sum() < MIN_IN_BAND:
            continue
        label = f"{low}-{high}" if high < 10_000 else f"{low}+"
        coverage = {level: float(forecast.covered(level)[mask].mean())
                    for level in nominal}
        rows.append(_row(label, error[mask], forecast.truth[mask], coverage))

    coverage = {level: float(forecast.covered(level).mean())
                for level in nominal}
    rows.append(_row("all", error, forecast.truth, coverage))
    return rows


def calibration_curve(forecast: Forecast, *,
                      levels=None) -> list[dict]:
    """Measured coverage against nominal, across the whole range.

    A perfectly calibrated estimator traces the diagonal. Departures are
    reported in both directions: an interval that covers more than it
    claims is not "safe", it is uninformative, and one that covers less is
    overconfident.
    """
    if levels is None:
        levels = (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
    rows = []
    for nominal in levels:
        measured = float(forecast.covered(nominal).mean())
        low, high = forecast.interval(nominal)
        rows.append({
            "nominal": float(nominal),
            "measured": measured,
            "gap": measured - float(nominal),
            "mean_width": float(np.mean(high - low)),
        })
    return rows


def compare(first: Forecast, second: Forecast, *, bands=BANDS) -> list[dict]:
    """The two estimators band by band, and which one wins where.

    Both forecasts must cover the same units in the same order, which is
    checked rather than assumed: comparing two estimators scored on
    different fleets is the easiest way to produce a confident wrong
    answer.
    """
    if not np.array_equal(first.truth, second.truth):
        raise ValueError("the two forecasts are not over the same units")

    rows = []
    for low, high in zip(bands[:-1], bands[1:]):
        mask = (first.truth >= low) & (first.truth < high)
        if mask.sum() < MIN_IN_BAND:
            continue
        one = float(np.abs(first.error[mask]).mean())
        two = float(np.abs(second.error[mask]).mean())
        rows.append({
            "band": f"{low}-{high}" if high < 10_000 else f"{low}+",
            "n": int(mask.sum()),
            f"{first.name} mae": one,
            f"{second.name} mae": two,
            "winner": first.name if one < two else second.name,
            "ratio": two / one if one else float("nan"),
        })
    one = float(np.abs(first.error).mean())
    two = float(np.abs(second.error).mean())
    rows.append({"band": "all", "n": first.n,
                 f"{first.name} mae": one, f"{second.name} mae": two,
                 "winner": first.name if one < two else second.name,
                 "ratio": two / one if one else float("nan")})
    return rows


def disagreement(first: Forecast, second: Forecast, *,
                 count: int = 10) -> dict:
    """Where the two estimators disagree most, and what the truth was.

    The point of having two independent estimators is that their
    disagreement is informative. This returns the units where the gap is
    widest, together with which one was closer, so the disagreement can be
    diagnosed rather than averaged away.
    """
    gap = second.median - first.median
    order = np.argsort(-np.abs(gap))[:count]
    rows = []
    for index in order:
        rows.append({
            "age": float(first.age[index]),
            "truth": float(first.truth[index]),
            first.name: float(first.median[index]),
            second.name: float(second.median[index]),
            "gap": float(gap[index]),
            "closer": (first.name
                       if abs(first.error[index]) < abs(second.error[index])
                       else second.name),
        })

    correlation = (float(np.corrcoef(first.median, second.median)[0, 1])
                   if first.n > 2 else float("nan"))
    return {
        "mean_absolute_gap": float(np.abs(gap).mean()),
        "correlation": correlation,
        "widest": rows,
        # When they agree, is the agreement right? Two independent
        # estimators landing on the same wrong answer is a different
        # situation from two that disagree, and worth separating.
        "agreement_mae": float(np.abs(first.error[np.abs(gap) < 15]).mean())
        if (np.abs(gap) < 15).any() else float("nan"),
    }


# -- the censoring counterfactual, carried through to predictions --------

def censoring_effect(subset: str = "FD001", *,
                     nominal: float = 0.90) -> list[dict]:
    """What each censoring treatment does to the predictions themselves.

    `lifetimes.treatments` and `simulate.censoring_recovery` show what the
    four treatments do to the fitted lifetime law. This carries that
    through to the quantity anyone actually cares about: the remaining-life
    predictions and their error.

    The lifetime law is fitted on the pooled fleet under each treatment and
    then used to predict the test fleet. The oracle row is the unobtainable
    target, not a method.

    One honest caveat about this particular table. The test units appear in
    the pooled fleet that the law is fitted to, and are then predicted --
    which would be circular if the purpose were to estimate out-of-sample
    accuracy. It is not: the purpose is to isolate the effect of the
    censoring treatment, and every row shares the same circularity, so the
    differences between rows are attributable to the treatment alone. The
    honest out-of-sample numbers are the train-fitted ones in
    `headline`.
    """
    lives = lifetimes.pooled(subset)
    units = data.load(subset, "test")
    fits = agelaw.fit_treatments(lifetimes.treatments(lives))

    rows = []
    for label in lifetimes.TREATMENTS:
        law = fits[label]
        forecast = from_agelaw(law, units, subset=subset)
        error = forecast.error
        rows.append({
            "treatment": label,
            "beta": law.beta,
            "eta": law.eta,
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "bias": float(error.mean()),
            "late_share": float((error > 0).mean()),
            "coverage": float(forecast.covered(nominal).mean()),
            "nominal": nominal,
        })

    oracle = next(row for row in rows if row["treatment"] == "oracle")
    for row in rows:
        row["mae_vs_oracle"] = row["mae"] - oracle["mae"]
        row["bias_vs_oracle"] = row["bias"] - oracle["bias"]
    return rows


# -- the whole thing for one subset --------------------------------------

def headline(subset: str = "FD001", *, seed: int = 0,
             cap: int | None = None, window: int | None = None) -> dict:
    """Fit both estimators on train, score both on test, compare them.

    Both are fitted on the run-to-failure fleet only and scored on the
    censored fleet, which is the honest protocol: nothing either estimator
    sees during fitting comes from the units it is asked about.
    """
    train = data.load(subset, "train")
    test = data.load(subset, "test")

    time, failed = lifetimes.arrays(lifetimes.build(train))
    law = agelaw.fit(time, failed, name=f"{subset} train")
    model = degradation.fit(subset, seed=seed, cap=cap, window=window)

    first = from_agelaw(law, test, subset=subset)
    second = from_degradation(model, test, subset=subset)

    return {
        "subset": subset,
        "law": law,
        "model": model,
        "age_law": first,
        "degradation": second,
        "age_law_bands": by_true_rul(first),
        "degradation_bands": by_true_rul(second),
        "age_law_calibration": calibration_curve(first),
        "degradation_calibration": calibration_curve(second),
        "comparison": compare(first, second),
        "disagreement": disagreement(first, second),
    }


def _print_bands(name: str, rows: list[dict]) -> None:
    print(f"  {name}")
    print(f"    {'band':>8}  {'n':>4}  {'MAE':>6}  {'bias':>7}  "
          f"{'late':>5}   coverage at nominal 50/80/90/95")
    for row in rows:
        print(f"    {row['band']:>8}  {row['n']:4d}  {row['mae']:6.1f}  "
              f"{row['bias']:+7.1f}  {row['late_share']:5.0%}   "
              f"{row['coverage_50']:5.0%} {row['coverage_80']:5.0%} "
              f"{row['coverage_90']:5.0%} {row['coverage_95']:5.0%}")


if __name__ == "__main__":
    for subset in data.SUBSETS:
        found = headline(subset)
        model = found["model"]
        print(f"=== {subset}   age law beta {found['law'].beta:.2f}   "
              f"degradation cap {model.cap} window {model.window}")
        _print_bands("estimator A, age only", found["age_law_bands"])
        _print_bands("estimator B, degradation signal",
                     found["degradation_bands"])
        print("  head to head, mean absolute error")
        for row in found["comparison"]:
            print(f"    {row['band']:>8}  {row['n']:4d}  "
                  f"A {row['age law mae']:6.1f}   "
                  f"B {row['degradation mae']:6.1f}   -> {row['winner']}")
        gap = found["disagreement"]
        print(f"  mean absolute gap between them "
              f"{gap['mean_absolute_gap']:.1f} cycles, "
              f"correlation {gap['correlation']:+.3f}")
        print()
