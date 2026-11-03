"""Which sensor channels carry degradation signal, and where.

This is analysis, not preprocessing. The obvious thing to do with C-MAPSS
is to drop the channels that look flat and get on with the modelling, and
doing that produces a result that is wrong in a way nothing downstream
reveals. So the screening is run deliberately, both ways, and the
difference between the two answers is one of the project's findings.

**The finding.** Screening by how strongly a channel moves with age, as the
mean within-unit Spearman correlation between the channel and the cycle
number:

    subset  regimes   informative pooled   informative within regime
    FD001      1        14 of 21               14 of 21
    FD002      6         1 of 21               14 of 21
    FD003      1        10 of 21               10 of 21
    FD004      6         1 of 21                9 of 21

In the two single-regime subsets the two screenings agree exactly. In the
two multi-regime subsets, pooled screening finds **one** usable channel out
of twenty-one: no correlation anywhere in FD002 exceeds 0.28, and the fleet
looks as though it has essentially no degradation signal in it. Holding the
regime fixed recovers fourteen, and thirteen of those were rejected by the
pooled screening. FD004 is the same story at eight.

The channels are the same channels. Ps30, the strongest single indicator in
the single-regime subsets, reads +0.18 pooled in FD002 and +0.79 once the
operating regime is held fixed. T50 goes +0.18 to +0.74, BPR +0.17 to
+0.72, htBleed +0.12 to +0.62.

What is happening is not subtle once stated: moving the engine between six
flight conditions swings every sensor by far more than degradation does, so
pooling the regimes buries a real monotone drift under a much larger
between-regime variance. A screening step run on pooled data would reject
thirteen of the channels that carry the signal, and would do it while
reporting perfectly reasonable-looking correlations near zero.

**Why this is a result and not a footnote.** It means a sensor-selection
step is not transferable between subsets of the same dataset, and that
"which sensors matter" is not a property of the engine but of the engine
*and* how it is being flown. It also means the regime-wise standardisation
in `Normaliser` is load-bearing rather than cosmetic: without it the
degradation estimator in `degradation` has nothing to work with on FD002
and FD004, and the failure would look like a modelling failure rather than
a preprocessing one.

**What stays excluded either way.** Four channels -- T2, P2, Nf_dmd and
PCNfR_dmd -- are flat in every subset under both screenings. These are
inlet conditions and commanded fan speeds: quantities the simulator holds
or dictates rather than measures off a degrading engine. Excluding them is
justified by what they are, which is a better reason than a threshold.

**A second finding, from a channel that refused to stay excluded.** `epr`,
the engine pressure ratio, belongs on that list for FD001 and FD002, where
it reads exactly 0.000 under both screenings. It does not belong there for
FD003 and FD004, where it trends +0.308 and +0.206. Those are precisely the
two subsets with a *second fault mode*: FD001 and FD002 degrade the
high-pressure compressor alone, while FD003 and FD004 degrade the fan as
well.

That is physically the right place for it to appear. The engine pressure
ratio is a ratio of pressures across the machine, and fan degradation
changes it in a way that compressor degradation on its own does not. So
which channels carry signal turns out to depend on *two* things, not one:
the operating regime, and which fault is actually growing. A sensor set
chosen on FD001 would discard the one channel that distinguishes FD003's
second failure mode, and would do so while looking perfectly well
justified.

`farB` is a further borderline case -- exactly flat in FD001 and FD003,
weakly alive in FD002 and FD004 -- and the screening treats it on its
measured trend like anything else. It contributes almost nothing either
way.

**On thresholds.** `== 0` is not used as a screening rule anywhere here.
In FD001 six channels are exactly constant to the recorded precision, but
another is flat to within 1e-5 relative and would be missed by an equality
test on a dataset rounded one digit differently. Every cut below is a
stated threshold on a reported statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

import data

# A channel counts as carrying signal if its mean within-unit rank
# correlation with age reaches this. Chosen to sit in the gap between the
# flat channels (below 0.1) and the informative ones (above 0.35); nothing
# in the analysis is sensitive to its exact value, and `screen` reports the
# statistic so the cut can be re-drawn.
TREND_FLOOR = 0.25

# Channels the simulator holds or commands rather than measures off the
# engine. Flat in all four subsets under both screenings.
HELD_CHANNELS = (0, 4, 17, 18)          # T2, P2, Nf_dmd, PCNfR_dmd

# Flat under one fault mode and informative under two. `epr` is dead in
# FD001 and FD002, which degrade the compressor alone, and trends at +0.31
# and +0.21 in FD003 and FD004, which also degrade the fan. Kept separate
# from HELD_CHANNELS because the distinction is the finding.
FAULT_MODE_CHANNELS = (9,)              # epr
TWO_FAULT_SUBSETS = ("FD003", "FD004")


@dataclass(frozen=True)
class Screening:
    """Per-channel screening of one fleet."""

    subset: str
    pooled_trend: np.ndarray        # mean within-unit Spearman, raw
    regime_trend: np.ndarray        # the same, after regime standardising
    spread: np.ndarray              # pooled standard deviation
    relative_spread: np.ndarray     # spread over |mean|
    regimes: int

    def informative(self, *, floor: float = TREND_FLOOR,
                    by_regime: bool = True) -> list[int]:
        """Channel indices that reach the trend floor."""
        trend = self.regime_trend if by_regime else self.pooled_trend
        return [int(j) for j in np.flatnonzero(np.abs(trend) >= floor)]

    def masked(self, *, floor: float = TREND_FLOOR) -> list[int]:
        """Channels that only show their trend once the regime is fixed.

        The project's sensor finding, as a list. Empty for the
        single-regime subsets and thirteen channels long for FD002.
        """
        return [j for j in self.informative(floor=floor, by_regime=True)
                if j not in self.informative(floor=floor, by_regime=False)]

    def flat_everywhere(self, *, floor: float = 0.1) -> list[int]:
        return [int(j) for j in np.flatnonzero(
            (np.abs(self.pooled_trend) < floor)
            & (np.abs(self.regime_trend) < floor))]


def _within_unit_trend(units: list[data.Unit],
                       channels: np.ndarray | None = None) -> np.ndarray:
    """Mean within-unit Spearman correlation of each channel with age.

    Taken per unit and then averaged, rather than pooled across units,
    because pooling would mix between-unit differences in absolute sensor
    level into what is meant to be a statement about drift with age.
    """
    totals = np.zeros(data.N_SENSORS)
    counts = np.zeros(data.N_SENSORS)
    offset = 0
    for unit in units:
        block = (channels[offset:offset + unit.observed]
                 if channels is not None else unit.sensors)
        offset += unit.observed
        cycles = unit.cycles()
        for j in range(data.N_SENSORS):
            column = block[:, j]
            # A channel with no variation inside this unit trends at zero
            # by definition. Tested before calling out, because scipy
            # warns and returns NaN for a constant input, and a NaN here
            # would silently drop the channel from the average instead of
            # counting the zero it actually is.
            #
            # Tested by range and not by standard deviation. In FD001 the
            # P2 channel holds one value for all 192 cycles of unit 1, and
            # `std()` still returns 3.6e-15 for it rather than zero, because
            # the variance is accumulated rather than exact. `ptp` is the
            # same test scipy itself applies, and is exact.
            if np.ptp(column) == 0.0 or unit.observed < 3:
                counts[j] += 1
                continue
            found = stats.spearmanr(column, cycles).statistic
            if np.isfinite(found):
                totals[j] += found
                counts[j] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)


def screen(subset: str = "FD001", *, seed: int = 0) -> Screening:
    """Screen one subset's training fleet, pooled and within regime."""
    units = data.load(subset, "train")
    labels, _ = data.regime_labels(units, seed=seed)
    regimes = int(np.unique(labels).size)

    stacked = np.vstack([u.sensors for u in units])
    spread = stacked.std(axis=0)
    centre = np.abs(stacked.mean(axis=0))
    relative = spread / np.where(centre > 0, centre, 1.0)

    normaliser = Normaliser.fit(units, labels)
    standardised = normaliser.apply_fleet(units, labels)

    return Screening(
        subset=subset,
        pooled_trend=_within_unit_trend(units),
        regime_trend=_within_unit_trend(units, standardised),
        spread=spread, relative_spread=relative, regimes=regimes)


def masking_report(subsets=data.SUBSETS) -> list[dict]:
    """The pooled-versus-within-regime comparison, one row per subset."""
    rows = []
    for subset in subsets:
        found = screen(subset)
        pooled = found.informative(by_regime=False)
        by_regime = found.informative(by_regime=True)
        rows.append({
            "subset": subset,
            "regimes": found.regimes,
            "pooled_informative": len(pooled),
            "regime_informative": len(by_regime),
            "masked": len(found.masked()),
            "masked_channels": found.masked(),
            "pooled_max": float(np.abs(found.pooled_trend).max()),
            "regime_max": float(np.abs(found.regime_trend).max()),
            "screening": found,
        })
    return rows


# -- regime-wise standardisation -----------------------------------------

@dataclass(frozen=True)
class Normaliser:
    """Centres and scales each channel within each operating regime.

    Fitted on a training fleet and then applied unchanged to test units,
    which matters for two reasons. A test record can be short enough that
    its own per-regime statistics are meaningless, and estimating them from
    the unit being predicted would let the normalisation absorb part of the
    degradation it is supposed to expose.

    A test cycle is assigned to the nearest training regime centre, so the
    regimes themselves are also fixed by the training fleet.
    """

    centres: np.ndarray             # (regimes, 3) in setting space
    means: np.ndarray               # (regimes, 21)
    scales: np.ndarray              # (regimes, 21)
    setting_scale: np.ndarray       # (3,) used to make the centres metric

    @classmethod
    def fit(cls, units: list[data.Unit],
            labels: np.ndarray | None = None, *,
            seed: int = 0) -> "Normaliser":
        if labels is None:
            labels, _ = data.regime_labels(units, seed=seed)
        settings = np.vstack([u.settings for u in units])
        sensors = np.vstack([u.sensors for u in units])

        setting_scale = settings.std(axis=0)
        setting_scale[setting_scale == 0] = 1.0

        groups = np.unique(labels)
        centres = np.zeros((groups.size, data.N_SETTINGS))
        means = np.zeros((groups.size, data.N_SENSORS))
        scales = np.ones((groups.size, data.N_SENSORS))
        for index, group in enumerate(groups):
            mask = labels == group
            centres[index] = settings[mask].mean(axis=0)
            means[index] = sensors[mask].mean(axis=0)
            block = sensors[mask]
            spread = block.std(axis=0)
            # A channel with no variation inside a regime standardises to
            # zero rather than to a division by zero.
            #
            # Constancy is decided by the range, not by `spread > 0`. For a
            # channel that holds one value for every cycle, `std` can still
            # return something like 3.6e-15 -- in FD001, P2 does exactly
            # that -- and dividing by that is a division by zero wearing a
            # disguise. It happens to come out at zero here only because
            # the numerator is bit-exactly zero too, which is luck rather
            # than arithmetic and would stop being true under any rounding
            # of the source data.
            constant = np.ptp(block, axis=0) == 0.0
            scales[index] = np.where(constant, 1.0, spread)
        return cls(centres=centres, means=means, scales=scales,
                   setting_scale=setting_scale)

    @property
    def regimes(self) -> int:
        return int(self.centres.shape[0])

    def assign(self, settings: np.ndarray) -> np.ndarray:
        """Nearest training regime for each row of settings."""
        settings = np.atleast_2d(np.asarray(settings, dtype=float))
        scaled = settings / self.setting_scale
        centres = self.centres / self.setting_scale
        distance = ((scaled[:, None, :] - centres[None, :, :]) ** 2
                    ).sum(axis=2)
        return np.argmin(distance, axis=1)

    def apply(self, unit: data.Unit) -> np.ndarray:
        """One unit's sensors, standardised within its own regimes."""
        labels = self.assign(unit.settings)
        return (unit.sensors - self.means[labels]) / self.scales[labels]

    def apply_fleet(self, units: list[data.Unit],
                    labels: np.ndarray | None = None) -> np.ndarray:
        """A whole fleet, stacked in unit order."""
        if labels is None:
            return np.vstack([self.apply(u) for u in units])
        sensors = np.vstack([u.sensors for u in units])
        return (sensors - self.means[labels]) / self.scales[labels]


if __name__ == "__main__":
    print("channel screening: pooled against within-regime\n")
    for row in masking_report():
        found = row["screening"]
        print(f"{row['subset']}  {row['regimes']} regime(s)")
        print(f"  informative pooled {row['pooled_informative']:2d}/21 "
              f"(max |rho| {row['pooled_max']:.3f})   "
              f"within regime {row['regime_informative']:2d}/21 "
              f"(max |rho| {row['regime_max']:.3f})")
        if row["masked"]:
            print(f"  {row['masked']} channel(s) only show their trend once "
                  f"the regime is held fixed:")
            for j in row["masked_channels"]:
                print(f"     {data.SENSOR_NAMES[j][:36]:36s} "
                      f"{found.pooled_trend[j]:+.3f} -> "
                      f"{found.regime_trend[j]:+.3f}")
        print()
