"""Planted truths: fleets whose answer is known by construction.

Every headline claim in this project is checked against simulated data
whose parameters were chosen in advance. That is the only way to tell a
working estimator from a plausible one, because a broken survival model
does not raise — it returns a curve.

The important content here is `censoring_recovery`, which exists because
of a result on the real data that contradicted the obvious expectation and
needed explaining rather than reporting.

**Three censoring schemes, not one.** How a unit comes to be censored
decides which estimator is even consistent, and the three schemes below
are genuinely different problems wearing the same clothes.

`independent`
    The censoring time is drawn without reference to the unit's lifetime.
    This is the assumption the right-censored likelihood is built on, and
    under it the censored MLE is consistent while both naive treatments are
    biased short. The textbook case.

`proportional`
    Each unit is censored at a random fraction of *its own* lifetime. The
    censoring time now carries information about the lifetime, which is
    precisely what "non-informative censoring" rules out. The censored MLE
    is no longer consistent and comes out biased long.

    This is not a hypothetical. It is what C-MAPSS does: across the four
    subsets the correlation between a test unit's censoring cycle and its
    true failure cycle runs from +0.56 to +0.88, and the truncation
    fraction averages about 0.63 with a wide spread. The simulation
    reproduces the real data's bias pattern closely, which is what
    establishes the mechanism rather than merely naming it.

`end_of_test`
    Every unit still running at a common cut-off is censored. This is the
    scheme an operator actually faces — the censored engines are the ones
    still on the wing — and it is strongly length-biased: the censored set
    is exactly the long-lived tail. Here dropping the censored units is
    catastrophic rather than harmless.

The third scheme is in the project for one reason. On C-MAPSS, discarding
the censored units turns out to be nearly unbiased, and it would be very
easy to read that as "censoring handling is fussy and dropping them is
fine". It is nothing of the kind. It happens because C-MAPSS censors a
random half of a fleet whose lifetime distribution matches the half it
keeps, so the discarded units are a random subsample rather than a biased
one. Change the censoring scheme to the one real fleets have and the same
shortcut fails hard. `censoring_recovery` runs all three so that the
caveat is a measured number instead of a sentence of hedging.
"""

from __future__ import annotations

import numpy as np

import agelaw

SCHEMES = ("independent", "proportional", "end_of_test")

# Chosen in advance, and roughly where the real FD001 fleet sits, so the
# simulated bias pattern is comparable to the measured one.
TRUE_BETA = 4.5
TRUE_ETA = 225.0


def fleet(beta: float = TRUE_BETA, eta: float = TRUE_ETA, n: int = 40000,
          *, seed: int = 0) -> np.ndarray:
    """True lifetimes from a Weibull with known parameters."""
    rng = np.random.default_rng(seed)
    return eta * rng.weibull(beta, n)


def censor(truth: np.ndarray, scheme: str, *, rate: float = 0.5,
           seed: int = 0, floor: float = 0.10
           ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a censoring scheme, returning `(observed, failed)`.

    `rate` is the share of units selected for censoring, which for
    `independent` is an upper bound on the realised rate because a
    censoring time drawn beyond the unit's lifetime censors nothing.
    `floor` is the earliest fraction of life at which `proportional` will
    cut a record.
    """
    rng = np.random.default_rng(seed + 1)
    truth = np.asarray(truth, dtype=float)
    n = truth.size

    if scheme == "independent":
        chosen = rng.random(n) < rate
        # Drawn from the lifetime range but unrelated to the unit's own
        # lifetime, which is what independence means here.
        times = np.quantile(truth, rng.random(n))
        cut = chosen & (times < truth)
        return np.where(cut, times, truth), ~cut

    if scheme == "proportional":
        chosen = rng.random(n) < rate
        fraction = rng.uniform(floor, 1.0, n)
        return np.where(chosen, fraction * truth, truth), ~chosen

    if scheme == "end_of_test":
        # A single cut-off placed so that `rate` of the fleet is still
        # running when the test stops.
        limit = float(np.quantile(truth, 1.0 - rate))
        cut = truth > limit
        return np.where(cut, limit, truth), ~cut

    raise KeyError(f"{scheme} is not one of {SCHEMES}")


def treatments(observed: np.ndarray, failed: np.ndarray,
               truth: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The four treatments of `lifetimes.treatments`, for simulated data."""
    observed = np.asarray(observed, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    truth = np.asarray(truth, dtype=float)
    return {
        "censored MLE": (observed, failed),
        "drop censored": (observed[failed], failed[failed]),
        "as failures": (observed, np.ones_like(failed, dtype=bool)),
        "oracle": (truth, np.ones_like(failed, dtype=bool)),
    }


def censoring_recovery(*, beta: float = TRUE_BETA, eta: float = TRUE_ETA,
                       n: int = 40000, rate: float = 0.5,
                       seed: int = 0) -> list[dict]:
    """Can each treatment recover a known lifetime law, under each scheme?

    One row per (scheme, treatment), carrying the recovered parameters and
    their signed error against the planted truth. The whole censoring
    argument of the report is this table.
    """
    truth = fleet(beta, eta, n, seed=seed)
    rows = []
    for scheme in SCHEMES:
        observed, failed = censor(truth, scheme, rate=rate, seed=seed)
        realised = float(1.0 - failed.mean())
        for label, (time, event) in treatments(observed, failed,
                                               truth).items():
            fitted = agelaw.fit(time, event, name=f"{scheme}/{label}")
            rows.append({
                "scheme": scheme,
                "treatment": label,
                "censoring_rate": realised,
                "true_beta": beta,
                "true_eta": eta,
                "beta": fitted.beta,
                "eta": fitted.eta,
                "beta_error": fitted.beta / beta - 1.0,
                "eta_error": fitted.eta / eta - 1.0,
                "n": fitted.n,
                "failures": fitted.failures,
            })
    return rows


def measured_censoring_dependence(lives) -> dict:
    """How far the real fleet departs from independent censoring.

    The correlation between a censored unit's cut-off cycle and its true
    failure cycle, which under independent censoring would be zero, plus
    the truncation fractions that say which simulated scheme the real data
    resembles. Takes `lifetimes.Life` records.
    """
    from scipy import stats

    censored = [one for one in lives if not one.failed]
    if len(censored) < 3:
        return {"units": len(censored), "correlation": float("nan"),
                "p_value": float("nan")}
    observed = np.array([one.observed for one in censored], dtype=float)
    truth = np.array([one.truth for one in censored], dtype=float)
    fraction = observed / truth
    result = stats.pearsonr(observed, truth)
    return {
        "units": len(censored),
        "correlation": float(result.statistic),
        "p_value": float(result.pvalue),
        "fraction_mean": float(fraction.mean()),
        "fraction_sd": float(fraction.std()),
        "fraction_min": float(fraction.min()),
        "fraction_max": float(fraction.max()),
    }


if __name__ == "__main__":
    print(f"planted truth: beta {TRUE_BETA}, eta {TRUE_ETA}\n")
    current = None
    for row in censoring_recovery():
        if row["scheme"] != current:
            current = row["scheme"]
            print(f"{current}  (censoring {row['censoring_rate']:.0%})")
        print(f"  {row['treatment']:14s} beta {row['beta']:6.3f} "
              f"({row['beta_error']:+7.2%})   eta {row['eta']:7.2f} "
              f"({row['eta_error']:+7.2%})")
