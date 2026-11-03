"""Lifetime records, and the four ways of treating the censored ones.

One engine becomes one lifetime observation: how far it ran, and whether
the record ended in failure. That is a severe compression — twenty-one
sensor channels per cycle thrown away in favour of two numbers — and it is
deliberate. It is exactly the information an age-based estimator is allowed
to have, and `agelaw` is built on nothing else, so that the comparison
against a sensor-driven estimator is a comparison between *kinds* of
information rather than between two tunings of the same model.

**Why the censored units are not missing data.** A C-MAPSS test unit's
record stops before failure. That is not an absent observation; it is the
observation "this engine was still running at cycle 134". A fleet of those
carries real information about the lifetime distribution, and the
right-censored likelihood is how it gets used.

The dataset allows something unusual here. In most reliability work the
cost of mishandling censoring can only be argued for, because the censored
units' true lifetimes are by definition unknown. Here they are known: the
archive supplies them. So `treatments` builds four fits of the same
quantity,

    censored MLE      the correct thing: survivors contribute survival
    drop censored     survivors discarded as though they were missing
    as failures       survivors counted as having failed when the record
                      stopped
    oracle            survivors' true failure cycles used as if observed

and the fourth is the one that cannot normally exist. It is not a method —
nobody can compute it in service — it is the answer the others are scored
against.

**What that comparison actually showed, which was not what was expected.**
The expectation going in was the textbook one: the censored MLE lands on
the oracle, and both naive treatments miss short. On C-MAPSS that is wrong
in two of its three parts. Measured on the pooled FD001 fleet, against the
oracle:

    censored MLE     eta  +5.3%    beta  +6.0%
    drop censored    eta  +0.1%    beta  -3.0%
    as failures      eta -16.0%    beta -35.6%

and the same pattern holds on all four subsets. Discarding the censored
units — the treatment that is supposed to be a blunder — is the most
accurate of the three, and the correct likelihood is biased *long*.

Both halves of that have the same cause, and it is a property of how the
dataset was built:

* **The censoring is not independent of the lifetime.** Each test unit's
  record is cut at a random fraction of its own life — the correlation
  between cut-off cycle and true failure cycle runs +0.56 to +0.88 across
  the four subsets, and the truncation fraction averages about 0.63. A
  censoring time that knows the lifetime is exactly what the
  right-censored likelihood assumes away, so the MLE is not consistent
  here, and it errs long. `simulate.censoring_recovery` reproduces this
  from a planted truth, so it is a diagnosed mechanism and not a guess.
* **The censored units are a random half, not the long-lived tail.** The
  train and test fleets have indistinguishable lifetime distributions
  (two-sample KS, p = 0.07 to 0.99). So dropping the test units discards a
  random subsample: it halves the data and widens every interval, but it
  biases nothing. The familiar bias from dropping censored observations
  needs the censored set to be length-biased, and here it is not.

**Which is not a licence to drop censored data.** That shortcut works on
C-MAPSS because of an artefact of its construction. Under the censoring an
operator actually faces — a common cut-off, where the censored engines are
the ones still flying and therefore the long-lived tail — the same shortcut
recovers a shape 43% too high and a scale 22% too low, while the censored
MLE stays within half a per cent. `simulate` runs that scheme alongside
the other two for exactly this reason. Across all three, the censored MLE
is the only treatment that is never badly wrong; each naive treatment is
catastrophic under some scheme it cannot detect from the data it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import data

TREATMENTS = ("censored MLE", "drop censored", "as failures", "oracle")


@dataclass(frozen=True)
class Life:
    """One engine, compressed to a lifetime."""

    subset: str
    split: str
    unit: int
    observed: float          # cycles in the record
    failed: bool             # did the record end in failure
    truth: float             # true failure cycle, for evaluation only

    @property
    def remaining(self) -> float:
        """True remaining useful life at the end of the record."""
        return self.truth - self.observed


def build(units: list[data.Unit]) -> list[Life]:
    """Lifetime records for a fleet, in unit order."""
    return [Life(subset=u.subset, split=u.split, unit=u.unit,
                 observed=float(u.observed), failed=u.failed,
                 truth=float(u.true_life))
            for u in units]


def pooled(subset: str = "FD001") -> list[Life]:
    """Both fleets of one subset, which is where censoring appears.

    The train fleet alone contains no censored units at all, so an analysis
    confined to it cannot say anything about censoring. Pooling the two is
    what the operator's position actually looks like: a history of engines
    that ran to failure, plus a set still on the wing.
    """
    return (build(data.load(subset, "train"))
            + build(data.load(subset, "test")))


def arrays(lives: list[Life]) -> tuple[np.ndarray, np.ndarray]:
    """The `(time, failed)` pair every estimator here takes."""
    return (np.array([one.observed for one in lives], dtype=float),
            np.array([one.failed for one in lives], dtype=bool))


def truth(lives: list[Life]) -> np.ndarray:
    """True failure cycles. For evaluation, never for fitting."""
    return np.array([one.truth for one in lives], dtype=float)


def census(lives: list[Life]) -> dict:
    """What the fleet looks like before anything is fitted to it."""
    time, failed = arrays(lives)
    real = truth(lives)
    censored = ~failed
    return {
        "units": len(lives),
        "failures": int(failed.sum()),
        "censored": int(censored.sum()),
        "censoring_rate": float(censored.mean()) if lives else float("nan"),
        "observed_total": float(time.sum()),
        "truth_total": float(real.sum()),
        # How much of the fleet's real service life the records actually
        # cover. The shortfall is the information censoring withholds.
        "coverage": float(time.sum() / real.sum()) if lives else float("nan"),
        "hidden_cycles": float(real.sum() - time.sum()),
        "median_truth": float(np.median(real)) if lives else float("nan"),
        "median_observed": float(np.median(time)) if lives else float("nan"),
        "median_remaining_if_censored": (
            float(np.median(real[censored] - time[censored]))
            if censored.any() else float("nan")),
    }


def treatments(lives: list[Life]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The same fleet as each of the four treatments would see it.

    Returns `(time, failed)` pairs ready for a fitter, so that the
    difference between the four fits is entirely a difference in how the
    censored units were handled and not in anything else.
    """
    time, failed = arrays(lives)
    real = truth(lives)

    return {
        # Survivors contribute "lasted at least this long".
        "censored MLE": (time, failed),
        # Survivors discarded. Note the fleet shrinks, which is the
        # second thing wrong with it.
        "drop censored": (time[failed], failed[failed]),
        # Survivors counted as deaths at the moment the record stopped.
        "as failures": (time, np.ones_like(failed, dtype=bool)),
        # Survivors' real failure cycles, as if the test had been allowed
        # to run. Unobtainable in service; the target the others are
        # judged against.
        "oracle": (real, np.ones_like(failed, dtype=bool)),
    }
