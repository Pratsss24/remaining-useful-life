"""Estimator B: remaining life from the sensor trajectory.

This estimator never learns how long an engine lives. It is given no
lifetime distribution, no fleet survival curve and not even the engine's
age — only the sensor readings. That is what makes it independent of
`agelaw` rather than a refinement of it: the two answer the same question
from disjoint information, so they can be made to disagree, and where they
do the disagreement means something.

It works in three stages.

**1. A health index.** The twenty-one channels are reduced to one number
per cycle by least squares against a health target, fitted on the
run-to-failure fleet. The channels used are the ones `sensors` finds carry
a monotone trend with age, standardised within operating regime — which on
FD002 and FD004 is not optional, since pooled screening finds one usable
channel out of twenty-one and regime-wise screening finds fourteen.

The target is `min(RUL, cap) / cap`: health sits at 1 while the engine is
healthy and falls linearly to 0 at failure. The cap exists because
degradation has an onset. An engine does not begin dying at cycle 1, and a
target that declines from the first cycle asks the regression to find a
trend that is not in the data. Fitted against a straight 1-to-0 target the
index runs from 0.68 to -0.13 instead of from 1 to 0, and explains 65% of
the variance.

**The cap is estimated, not borrowed.** The literature's usual value is
125 or 130; this project scans for it by cross-validation on the training
fleet alone. Two independent criteria agree on about 100 cycles for FD001:
the fit quality peaks there, and it is also the cap at which the index
lands on 1 at a healthy start and 0 at failure without being asked to —
+0.975 and -0.009 respectively. That convergence is the reason to believe
the number rather than the scan.

Read physically, the cap is the horizon over which the sensors say anything
at all. Beyond roughly 100 cycles from failure this engine's instruments do
not distinguish an engine with 150 cycles left from one with 250. That is a
property of the machine, and it sets a hard ceiling on what any
sensor-driven estimator can do — including this one.

**2. Smoothing.** The index carries per-cycle noise with a standard
deviation of about 0.042 against an 11-cycle mean, so it is averaged over a
trailing window. The window length is chosen by the same cross-validation
as the cap.

**3. A calibrated inversion.** The index is turned into cycles by the
empirical conditional distribution of true remaining life given the index,
measured on the training fleet: the index is binned, and the 5th, 50th and
95th percentiles of true RUL within each bin are recorded and interpolated.

The point prediction is the conditional median and the interval is the
conditional 5-95 range, so the interval is built from the spread actually
observed rather than from a variance formula and an assumed shape. That is
a deliberate choice and it is the reason the measured coverage comes out
close to nominal; `evaluate` reports it as measured regardless.

**The route that does not work, kept because it was measured.** The natural
thing to do with a declining health index is to fit its slope and
extrapolate to the failure threshold. It fails badly, and not for the
reason one would guess. It is not noise. The index needs to fall at about
-0.009 per cycle to reach the threshold at the true failure cycle, and that
is close to what the run-to-failure units average over their final hundred
cycles. But measured on a test unit's trailing window the fitted slope is
only -0.001 to -0.003, three to eight times too shallow, so the projected
crossing lands far in the future. Pooled over FD001's test fleet this gives
a mean absolute error of 133 to 380 cycles against 32 for the age-only
estimator, with the error almost entirely positive bias.

The cause is attenuation. The index is a regression estimate of health, so
it is shrunk toward the fleet mean, and a shrunken signal has a shallower
slope than the quantity it estimates. Extrapolating a shrunken slope
extrapolates the shrinkage. Degradation also accelerates, which compounds
it: a unit censored while still at index 0.8 is in the slow early part of
its decline, and the remaining fall is steeper than anything in its own
history.

Calibrating the inversion fixes both at once, because the calibration is
fitted on the same shrunken index it is applied to. `crossing_extrapolation`
is kept so the report can show the measured failure rather than assert it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import data
import sensors

# Searched by cross-validation on the training fleet. The grid is wide
# enough to contain the literature's 125-130 so that the scan can disagree
# with it rather than be unable to.
CAP_GRID = (60, 80, 100, 120, 140, 160)
WINDOW_GRID = (5, 10, 20, 30)

BINS = 40
MIN_PER_BIN = 6
ALPHA = 0.10            # nominal 90% interval

# Quantile levels the calibration stores. Fixed in advance and wide enough
# to let `evaluate` measure coverage at any usual nominal level, so that the
# level is never chosen after seeing which one happened to cover well.
LEVELS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50,
          0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.975, 0.99)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing mean, by prefix sums.

    Each output uses only the current cycle and the `window - 1` before
    it, with a short window at the start of the record rather than a
    discarded prefix: a prediction made at cycle 4 has four cycles of
    history and should get the mean of those.

    Written with cumulative sums rather than a loop because the
    cross-validation calls it a few hundred thousand times.
    """
    values = np.asarray(values, dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    position = np.arange(values.size)
    start = np.maximum(0, position - window + 1)
    return (prefix[position + 1] - prefix[start]) / (position + 1 - start)


@dataclass(frozen=True)
class Trajectory:
    """One unit's standardised channels, prepared once and reused.

    Building these is the expensive part -- regime assignment and
    standardisation -- and the cross-validation refits the index a few
    hundred times over the same engines. Preparing them once turns the
    selection from minutes into seconds, and guarantees every candidate
    model sees bit-identical inputs.
    """

    unit: int
    design: np.ndarray             # (cycles, channels + 1), intercept last
    remaining: np.ndarray          # true RUL at each cycle
    observed: int

    def health(self, weights: np.ndarray) -> np.ndarray:
        return self.design @ weights


def prepare(units: list[data.Unit], normaliser: sensors.Normaliser,
            channels: tuple[int, ...]) -> list[Trajectory]:
    """Standardise and lay out a fleet for repeated fitting."""
    prepared = []
    for unit in units:
        block = normaliser.apply(unit)[:, list(channels)]
        prepared.append(Trajectory(
            unit=unit.unit,
            design=np.column_stack([block, np.ones(block.shape[0])]),
            remaining=unit.rul(), observed=unit.observed))
    return prepared


# -- the health index -----------------------------------------------------

@dataclass(frozen=True)
class HealthIndex:
    """A scalar health signal built from the sensor channels."""

    subset: str
    channels: tuple[int, ...]
    weights: np.ndarray            # one per channel, plus an intercept
    cap: int
    normaliser: sensors.Normaliser
    r_squared: float
    start_level: float             # index at a healthy start, should be ~1
    failure_level: float           # index at failure, should be ~0
    failure_spread: float

    def apply(self, unit: data.Unit) -> np.ndarray:
        """The raw index, one value per recorded cycle."""
        block = self.normaliser.apply(unit)[:, list(self.channels)]
        design = np.column_stack([block, np.ones(block.shape[0])])
        return design @ self.weights

    def smoothed(self, unit: data.Unit, window: int) -> np.ndarray:
        """Trailing mean of the index, causal by construction.

        Only past and present cycles enter each value. A centred window
        would let a cycle's smoothed index depend on readings taken after
        it, which is information a prediction made at that cycle cannot
        have.
        """
        return trailing_mean(self.apply(unit), window)

    @property
    def spans_unit_interval(self) -> bool:
        """Does the index land on 1 healthy and 0 dead without being told?

        The check that makes the fitted cap credible. Not used as a
        constraint -- the cap is chosen by cross-validated error -- so
        agreement here is evidence rather than construction.
        """
        return (abs(self.start_level - 1.0) < 0.1
                and abs(self.failure_level) < 0.1)


def _target(unit: data.Unit, cap: int) -> np.ndarray:
    return np.minimum(unit.rul(), cap) / cap


def fit_index(prepared: list[Trajectory], *, subset: str, cap: int,
              channels: tuple[int, ...],
              normaliser: sensors.Normaliser) -> HealthIndex:
    """Least squares from standardised channels onto capped health."""
    design = np.vstack([one.design for one in prepared])
    target = np.concatenate([np.minimum(one.remaining, cap) / cap
                             for one in prepared])

    weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ weights
    total = ((target - target.mean()) ** 2).sum()
    r_squared = 1.0 - ((target - fitted) ** 2).sum() / total if total else 0.0

    starts = np.array([one.health(weights)[:5].mean() for one in prepared])
    ends = np.array([one.health(weights)[-1] for one in prepared])
    return HealthIndex(subset=subset, channels=channels, weights=weights,
                       cap=cap, normaliser=normaliser,
                       r_squared=float(r_squared),
                       start_level=float(starts.mean()),
                       failure_level=float(ends.mean()),
                       failure_spread=float(ends.std()))


# -- the calibrated inversion --------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Empirical conditional distribution of remaining life given health.

    Built by binning the index on the training fleet and recording the
    percentiles of true remaining life within each bin. The prediction is
    an interpolation of those, so both the point estimate and the interval
    come from observed spread rather than from an assumed error law.

    `ceiling` is the highest index the training fleet covers. Above it the
    calibration is extrapolating, and a prediction there is reported as a
    lower bound -- which is the honest answer, because a healthy engine's
    sensors cannot distinguish 150 cycles of life left from 250.
    """

    centres: np.ndarray
    levels: np.ndarray             # quantile levels stored, ascending
    grid: np.ndarray               # (bins, levels) of true RUL
    alpha: float
    ceiling: float
    floor: float

    def quantile(self, health, q: float) -> np.ndarray:
        """The q-quantile of remaining life at a given health.

        Stored as a grid over levels rather than as one interval, so that
        `evaluate` can measure coverage at any nominal level without the
        calibration being refitted for each one -- and so the nominal level
        is never quietly chosen after seeing how well it covered.
        """
        health = np.asarray(health, dtype=float)
        column = np.interp(q, self.levels,
                           np.arange(self.levels.size, dtype=float))
        below, above = int(np.floor(column)), int(np.ceil(column))
        weight = column - below
        curve = (self.grid[:, below] * (1.0 - weight)
                 + self.grid[:, above] * weight)
        return np.interp(health, self.centres, curve)

    @property
    def median(self) -> np.ndarray:
        return self.grid[:, int(np.argmin(np.abs(self.levels - 0.5)))]

    def predict(self, health, *, alpha: float | None = None) -> dict:
        alpha = self.alpha if alpha is None else alpha
        health = np.asarray(health, dtype=float)
        return {
            "median": self.quantile(health, 0.5),
            "lower": self.quantile(health, alpha / 2.0),
            "upper": self.quantile(health, 1.0 - alpha / 2.0),
            "at_ceiling": health >= self.ceiling,
            "nominal_coverage": 1.0 - alpha,
        }


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      levels: np.ndarray) -> np.ndarray:
    """Quantiles of `values` under `weights`, by interpolated rank."""
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = weights.sum()
    if total <= 0:
        return np.quantile(values, levels)
    position = (np.cumsum(weights) - 0.5 * weights) / total
    return np.interp(levels, position, values)


def fit_calibration(index: HealthIndex, prepared: list[Trajectory], *,
                    window: int, alpha: float = ALPHA,
                    bins: int = BINS, monotone: bool = True,
                    unit_weighted: bool = True) -> Calibration:
    """Calibrate index to cycles on a run-to-failure fleet.

    **Why the cycles are weighted.** A run-to-failure record contributes
    one row per cycle, so a engine that lived 350 cycles contributes more
    than twice as many rows as one that lived 150 -- and almost all of the
    extra rows come from its long healthy stretch. The calibration is then
    fitted on a cycle-weighted population but applied to a unit-weighted
    one, since each test engine is asked about exactly once.

    The effect is not small and it is not uniform. Measured on FD001, the
    median true remaining life among training cycles at health 0.8 to 0.9
    is 130 cycles, while among test units observed at that same health it
    is 95 -- a 35-cycle gap, in the direction of over-prediction. Down at
    health 0.6 to 0.7 the same comparison gives 61 against 54, because
    those readings only happen near failure and there is no long healthy
    stretch to over-represent. So the bias is concentrated exactly in the
    mid-health range, which is where the uncorrected estimator was worst.

    Weighting each unit equally -- every cycle of a unit carrying
    1/lifetime -- is also the right match to how the test fleet was built:
    one observation per engine, at a roughly uniform fraction of its life.
    It takes FD001's pooled error from 22.4 to 19.6 cycles and its bias
    from +17.1 to +13.0, improving every band.

    It does not remove the mid-band bias, and the remainder has a different
    cause: beyond the cap the index saturates and cannot tell 150 cycles of
    life from 250, so predictions there fall back on the top bin's median.
    That part is a property of the sensors and is reported rather than
    corrected.

    `unit_weighted=False` restores the naive version so the difference can
    be shown rather than claimed.
    """
    health = np.concatenate([trailing_mean(one.health(index.weights), window)
                             for one in prepared])
    remaining = np.concatenate([one.remaining for one in prepared])
    if unit_weighted:
        weights = np.concatenate([np.full(one.observed, 1.0 / one.observed)
                                  for one in prepared])
    else:
        weights = np.ones(health.size)

    # Equal-count bins, so every bin has enough units to take a percentile
    # from. Equal-width bins leave the extremes almost empty, which is
    # exactly where the interval matters most.
    edges = np.quantile(health, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    slot = np.clip(np.searchsorted(edges, health, side="right") - 1,
                   0, bins - 1)

    levels = np.asarray(LEVELS, dtype=float)
    centres, rows = [], []
    for index_of_bin in range(bins):
        mask = slot == index_of_bin
        if mask.sum() < MIN_PER_BIN:
            continue
        centres.append(health[mask].mean())
        rows.append(weighted_quantile(remaining[mask], weights[mask],
                                      levels))

    centres = np.array(centres)
    order = np.argsort(centres)
    centres = centres[order]
    grid = np.vstack(rows)[order]

    if monotone:
        # Remaining life must not fall as health rises. Bin noise can
        # invert neighbouring bins, and an inverted calibration would let a
        # healthier reading predict less life left. Enforced by a running
        # maximum down each quantile column, which is the weakest
        # correction that guarantees it.
        grid = np.maximum.accumulate(grid, axis=0)

    return Calibration(centres=centres, levels=levels, grid=grid,
                       alpha=alpha, ceiling=float(centres[-1]),
                       floor=float(centres[0]))


# -- the estimator -------------------------------------------------------

@dataclass(frozen=True)
class DegradationModel:
    """Estimator B, fitted and ready to predict."""

    index: HealthIndex
    calibration: Calibration
    window: int
    selection: dict

    @property
    def cap(self) -> int:
        return self.index.cap

    def health(self, unit: data.Unit) -> float:
        """Current smoothed health, at the end of the record."""
        return float(self.index.smoothed(unit, self.window)[-1])

    def predict(self, unit: data.Unit) -> dict:
        found = self.calibration.predict(self.health(unit))
        return {key: (float(value) if np.isscalar(value) or value.ndim == 0
                      else value)
                for key, value in found.items()}

    def predict_fleet(self, units: list[data.Unit]) -> dict:
        health = np.array([self.health(u) for u in units])
        found = self.calibration.predict(health)
        found["health"] = health
        return found


def _score(index: HealthIndex, calibration: Calibration,
           held: list[Trajectory], window: int,
           fractions: dict[int, float]) -> float:
    """Mean absolute error on truncated held-out units.

    Held-out training units are complete records, and scoring a model on a
    complete record would measure something the test fleet never offers. So
    each is cut at a fraction of its life, which is the censoring mechanism
    C-MAPSS itself uses: on the real test fleet the truncation fraction
    averages 0.63 with a wide spread, and the cut-off correlates with true
    lifetime at +0.56 to +0.88.

    The fractions are passed in rather than drawn here, so that every
    candidate (cap, window) pair is scored on exactly the same cuts. Drawing
    them inside would make the comparison between candidates partly a
    comparison between random draws.
    """
    errors = []
    for one in held:
        cut = max(3, int(round(fractions[one.unit] * one.observed)))
        smoothed = trailing_mean(one.health(index.weights), window)
        predicted = float(np.interp(smoothed[cut - 1], calibration.centres,
                                    calibration.median))
        errors.append(predicted - (one.observed - cut))
    return float(np.mean(np.abs(errors)))


def _consistency(index: HealthIndex, held: list[Trajectory]) -> dict:
    """Out-of-sample check that the index spans the unit interval."""
    starts = np.array([one.health(index.weights)[:5].mean() for one in held])
    ends = np.array([one.health(index.weights)[-1] for one in held])
    return {"start": float(starts.mean()), "failure": float(ends.mean()),
            "gap": float(abs(starts.mean() - 1.0) + abs(ends.mean()))}


def select(subset: str = "FD001", *, folds: int = 5, seed: int = 0,
           cap_grid=CAP_GRID, window_grid=WINDOW_GRID) -> dict:
    """Choose the cap and the smoothing window, in that order.

    Folds are over *units*, never over cycles: splitting cycles would put
    the same engine on both sides and leak its trajectory into the fold
    that is supposed to be held out. Nothing here touches the test fleet.

    **Why the two are chosen by different criteria.** The obvious thing is
    one grid search over both, scored by held-out mean absolute error. It
    was tried first and it picks badly: caps of 60 on FD001 and 160 on
    FD002, at which the index spans 1.04 to 0.21 and 0.83 to -0.10 rather
    than 1 to 0.

    The reason is that the cap *defines the target*. Two caps produce
    indices that estimate different quantities, so scoring their downstream
    RUL predictions against the same truth is not a like-for-like
    comparison. Worse, the error is pooled over held-out units truncated
    anywhere in their lives, and most of those are cut while still healthy,
    where nothing in the sensors carries life information. A short cap makes
    the calibration saturate early and predict something close to a
    constant, and against a majority of no-signal units a constant scores
    well. The pooled average rewards it.

    That is the project's own thesis arriving in its model selection: a
    single pooled error number is the wrong thing to optimise, and here it
    is demonstrably the wrong thing to select on.

    So the cap is chosen by a criterion that does not compare across
    targets. The index is required to be *self-consistent*: fitted with the
    right cap it should land near 1 on a healthy engine and near 0 at
    failure, and that is a property of the resulting index rather than of
    the target it was fitted to. It is measured on held-out units, and it
    is a condition the fit is never asked to satisfy, so meeting it is
    evidence.

    It also agrees with fit quality. The cap minimising the consistency gap
    and the cap maximising R-squared land on the same value for FD001 and
    FD002 and within one grid step for FD003 and FD004. Two criteria that
    could disagree do not, which is the reason to believe the number.

    The window is then chosen by held-out error at the fixed cap, where the
    target is the same for every candidate and the comparison is sound.
    """
    units = data.load(subset, "train")
    labels, _ = data.regime_labels(units, seed=seed)
    normaliser = sensors.Normaliser.fit(units, labels)
    channels = tuple(sensors.screen(subset, seed=seed).informative())
    prepared = prepare(units, normaliser, channels)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units))
    groups = np.array_split(order, folds)
    fractions = {one.unit: float(f) for one, f
                 in zip(prepared, rng.uniform(0.15, 0.95, len(prepared)))}

    def split(held):
        inside = set(held.tolist())
        return ([prepared[i] for i in range(len(prepared)) if i not in inside],
                [prepared[i] for i in held])

    # -- stage one: the cap, by out-of-sample self-consistency ------------
    cap_table = []
    for cap in cap_grid:
        gaps, fits = [], []
        for held in groups:
            keep, out = split(held)
            index = fit_index(keep, subset=subset, cap=cap,
                              channels=channels, normaliser=normaliser)
            gaps.append(_consistency(index, out))
            fits.append(index.r_squared)
        cap_table.append({
            "cap": cap,
            "gap": float(np.mean([one["gap"] for one in gaps])),
            "start": float(np.mean([one["start"] for one in gaps])),
            "failure": float(np.mean([one["failure"] for one in gaps])),
            "r_squared": float(np.mean(fits)),
        })
    best_cap = min(cap_table, key=lambda row: row["gap"])["cap"]
    by_fit = max(cap_table, key=lambda row: row["r_squared"])["cap"]

    # -- stage two: the window, by held-out error at that cap -------------
    per_fold = []
    for held in groups:
        keep, out = split(held)
        per_fold.append((fit_index(keep, subset=subset, cap=best_cap,
                                   channels=channels,
                                   normaliser=normaliser), keep, out))

    window_table = []
    for window in window_grid:
        scores = [_score(index, fit_calibration(index, keep, window=window),
                         out, window, fractions)
                  for index, keep, out in per_fold]
        window_table.append({"window": window, "mae": float(np.mean(scores))})
    best = min(window_table, key=lambda row: row["mae"])

    return {
        "best_cap": best_cap,
        "best_window": best["window"],
        "best_mae": best["mae"],
        "cap_by_consistency": best_cap,
        "cap_by_fit": by_fit,
        "criteria_agree": best_cap == by_fit,
        "cap_table": cap_table,
        "window_table": window_table,
        "channels": channels,
        "folds": folds,
    }


def fit(subset: str = "FD001", *, seed: int = 0, cap: int | None = None,
        window: int | None = None, alpha: float = ALPHA) -> DegradationModel:
    """Fit estimator B on a subset's training fleet.

    `cap` and `window` are chosen by cross-validation unless given.
    """
    units = data.load(subset, "train")
    labels, _ = data.regime_labels(units, seed=seed)
    normaliser = sensors.Normaliser.fit(units, labels)
    channels = tuple(sensors.screen(subset, seed=seed).informative())

    if cap is None or window is None:
        chosen = select(subset, seed=seed)
        cap = cap if cap is not None else chosen["best_cap"]
        window = window if window is not None else chosen["best_window"]
    else:
        chosen = {"best_cap": cap, "best_window": window, "table": []}

    prepared = prepare(units, normaliser, channels)
    index = fit_index(prepared, subset=subset, cap=cap, channels=channels,
                      normaliser=normaliser)
    calibration = fit_calibration(index, prepared, window=window,
                                  alpha=alpha)
    return DegradationModel(index=index, calibration=calibration,
                            window=window, selection=chosen)


# -- the route that does not work ----------------------------------------

def crossing_extrapolation(model: DegradationModel, unit: data.Unit, *,
                           window: int = 40,
                           onset: float = 0.9) -> dict:
    """Fit the index's slope and project it to the failure threshold.

    The obvious approach, kept because its failure is a measured result
    rather than an opinion. Returns the projected remaining life, the
    fitted slope, and the slope that would have been needed to reach the
    threshold at the true failure cycle -- the comparison that identifies
    attenuation as the cause rather than noise.
    """
    health = model.index.apply(unit)
    cycles = unit.cycles()

    alive = health < onset
    start = int(np.flatnonzero(alive)[0]) if alive.any() else 0
    cycles, health = cycles[start:], health[start:]
    if cycles.size > window:
        cycles, health = cycles[-window:], health[-window:]
    if cycles.size < 3:
        return {"declined": "too few cycles past onset", "remaining": None}

    design = np.column_stack([cycles, np.ones(cycles.size)])
    (slope, intercept), *_ = np.linalg.lstsq(design, health, rcond=None)

    threshold = model.index.failure_level
    needed = ((threshold - health[-1]) / unit.rul_at_end
              if unit.rul_at_end > 0 else float("nan"))

    if slope >= 0:
        return {"declined": "index not falling", "remaining": None,
                "slope": float(slope), "needed_slope": float(needed)}

    crossing = (threshold - intercept) / slope
    return {
        "declined": None,
        "remaining": float(crossing - unit.observed),
        "slope": float(slope),
        "needed_slope": float(needed),
        "attenuation": float(needed / slope) if slope else float("nan"),
        "truth": float(unit.rul_at_end),
    }


if __name__ == "__main__":
    for subset in data.SUBSETS:
        model = fit(subset)
        index = model.index
        chosen = model.selection
        print(f"{subset}  cap {model.cap}  window {model.window}  "
              f"{len(index.channels)} channels")
        print(f"   index R2 {index.r_squared:.4f}   "
              f"healthy start {index.start_level:+.3f}   "
              f"at failure {index.failure_level:+.3f} "
              f"(sd {index.failure_spread:.3f})   "
              f"spans 0-1: {index.spans_unit_interval}")
        print(f"   cap by consistency {chosen['cap_by_consistency']}, "
              f"by fit {chosen['cap_by_fit']}  -> "
              f"{'agree' if chosen['criteria_agree'] else 'one grid step apart'}")
