"""Non-parametric survival estimation, assuming no shape at all.

This exists to check the parametric work in `agelaw` against something that
does not share its assumptions. A Weibull age law that disagrees with the
Kaplan-Meier curve of the same fleet is wrong about the data, not merely
imprecise, and without a distribution-free comparison there is no way to
tell the two apart.

`kaplan_meier`
    The survivor function. At each cycle where an engine failed, the
    fraction of the engines still at risk that got through it. A censored
    unit leaves the risk set without contributing a death, which is how
    "still running at cycle 134" enters the estimate.
`residual_life`
    Mean residual life read off the Kaplan-Meier curve by trapezoid, with
    no lifetime law involved. `agelaw` computes the same quantity in closed
    form from a fitted Weibull; the two have to agree wherever the curve is
    well determined, and where they stop agreeing is informative about the
    fit rather than about the estimator.

Both are standard. Variance for the survivor function is Greenwood's, and
the confidence band is built on the log-log scale so it cannot leave
[0, 1] — a symmetric band on S itself does, near the ends, which is where
these curves spend much of their time.

**The tail problem, stated here because it is load-bearing later.** Mean
residual life is an integral of the survivor function out to infinity, and
Kaplan-Meier says nothing beyond the last observed failure. So the
non-parametric version is only honest up to that point, and `residual_life`
reports how much of the integral it had to leave out rather than silently
truncating. The parametric version has no such limit, which is not a
virtue: it is extrapolating on the strength of an assumed shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Curve:
    """A survivor function as a step function, with a confidence band."""

    name: str
    times: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    extras: dict = field(default_factory=dict)

    def at(self, t: float) -> float:
        """S(t), right-continuous."""
        if self.times.size == 0 or t < self.times[0]:
            return 1.0
        index = int(np.searchsorted(self.times, t, side="right") - 1)
        return float(self.values[index])

    @property
    def total_events(self) -> int:
        return int(self.events.sum())

    @property
    def last_event(self) -> float:
        """Beyond this the curve is uninformative, not flat."""
        return float(self.times[-1]) if self.times.size else float("nan")


def kaplan_meier(time, failed, *, name: str = "",
                 alpha: float = 0.05) -> Curve:
    """The product-limit estimator.

    A censored observation at exactly an event time is conventionally held
    to have survived that time, so it stays in the risk set for it.
    """
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    event_times = np.unique(time[failed])

    events = np.array([np.sum(failed & (time == t)) for t in event_times],
                      dtype=float)
    at_risk = np.array([np.sum(time >= t) for t in event_times], dtype=float)

    survival = np.cumprod(1.0 - events / at_risk)

    # Greenwood's variance, on the log-log scale for the band.
    with np.errstate(divide="ignore", invalid="ignore"):
        increments = events / (at_risk * (at_risk - events))
        variance = np.cumsum(np.where(np.isfinite(increments), increments, 0.0))
        log_log_se = np.sqrt(variance) / np.abs(np.log(survival))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.exp(z * log_log_se)
        lower = survival ** factor
        upper = survival ** (1.0 / factor)

    lower = np.clip(np.nan_to_num(lower, nan=0.0), 0.0, 1.0)
    upper = np.clip(np.nan_to_num(upper, nan=1.0), 0.0, 1.0)

    return Curve(name=name, times=event_times, values=survival,
                 lower=lower, upper=upper, at_risk=at_risk, events=events,
                 extras={"n": int(time.size),
                         "censored": int((~failed).sum())})


def residual_life(curve: Curve, age: float) -> dict:
    """Mean residual life at `age`, straight off the curve.

    E[T - a | T > a] is the area under S beyond a, divided by S(a). The
    area is taken by trapezoid over the curve's own steps, which is exact
    for a step function between event times.

    Kaplan-Meier carries no information past its last observed failure, so
    the integral stops there. `truncated_at` and `survival_remaining` say
    how much was left out: if S is still well above zero at the last event,
    a large part of the integral is simply missing and the returned mean is
    a lower bound. Reported rather than patched, because the usual patches
    -- extending flat, or extending exponentially -- are assumptions
    dressed as arithmetic.
    """
    start = curve.at(age)
    if start <= 0:
        return {"mean": float("nan"), "lower_bound": True,
                "truncated_at": curve.last_event,
                "survival_remaining": 0.0, "age": age}

    keep = curve.times > age
    times = np.concatenate(([age], curve.times[keep]))
    values = np.concatenate(([start], curve.values[keep]))

    # Left-continuous steps: S holds its value until the next event.
    area = float(np.sum(np.diff(times) * values[:-1]))

    tail = float(values[-1])
    return {
        "mean": area / start,
        "age": age,
        "truncated_at": curve.last_event,
        # What fraction of the fleet is still alive where the curve runs
        # out. Anything much above zero means `mean` is a lower bound.
        "survival_remaining": tail / start,
        "lower_bound": tail / start > 1e-9,
    }


def median_survival(curve: Curve) -> float:
    """First time at which S drops to or below one half."""
    below = np.flatnonzero(curve.values <= 0.5)
    if below.size == 0:
        return float("nan")
    return float(curve.times[below[0]])
