"""Fetch the NASA C-MAPSS turbofan degradation data and cache it.

**Where the data comes from.** The dataset is NASA's Commercial Modular
Aero-Propulsion System Simulation run-to-failure study (Saxena, Goebel,
Simon and Eklund, PHM08). Its canonical home was the Prognostics Center of
Excellence repository at `ti.arc.nasa.gov/c/6/`. That URL no longer serves
the file: it redirects to a landing page with no download on it. The copy
used here is the PHM Data Challenge S3 mirror,

    https://phm-datasets.s3.amazonaws.com/NASA/
    6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

which is a zip containing one zip (`CMAPSSData.zip`) containing the text
files. Both layers' SHA-256 digests are pinned in `DIGESTS` below and
checked on every download, so a silently substituted mirror fails loudly
rather than quietly changing the numbers.

**What is in it.** Four subsets, each a fleet of engines of one type,
simulated from a healthy start through a fault that grows until the engine
is declared failed. They differ in two ways that matter:

    FD001   100 train /  100 test units   1 operating condition,  1 fault
    FD002   260 train /  259 test units   6 operating conditions, 1 fault
    FD003   100 train /  100 test units   1 operating condition,  2 faults
    FD004   248 train /  249 test units   6 operating conditions, 2 faults

Each row is one operating cycle of one engine: unit number, cycle number,
three operational settings, twenty-one sensor channels.

**The censoring structure, which is the reason this dataset is worth the
trouble.** A training unit is run until it fails, so its lifetime is
observed exactly. A test unit's record *stops before failure* — it is
right-censored — and the archive separately supplies the true remaining
useful life at that cut-off, in `RUL_FDxxx.txt`.

That is an unusual and very useful combination. The test units are
genuinely censored observations, so a model has to handle them as such;
but their true failure times are also known, so the cost of handling them
*badly* can be measured against ground truth rather than argued about.
`Unit.true_life` exposes that truth and `Unit.failed` exposes what an
honest estimator is allowed to know. Nothing in the estimators reads
`true_life`; it is for evaluation only.

**Operating condition is not a nuisance here.** In FD002 and FD004 the
three settings move the engine between six regimes, and a sensor reading
means something different in each. `regime_labels` recovers the regimes by
clustering the settings rather than assuming the published count, which
makes the count itself a result that can be checked.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

MIRROR = ("https://phm-datasets.s3.amazonaws.com/NASA/"
          "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
CANONICAL = "https://ti.arc.nasa.gov/c/6/"      # redirects, kept for the record

INNER_NAME = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"

DIGESTS = {
    "outer": "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2",
    "inner": "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f",
}

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "CMAPSSData.zip"
CACHE = ROOT / "data" / "parsed.npz"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# Composition as the archive's own readme.txt states it, copied verbatim and
# not corrected. `verify_composition` checks the parse against it, and on
# FD004 the two disagree: the readme says 248 train and 249 test, while the
# files hold 249 train and 248 test. The readme is the party at fault, and
# RUL_FD004.txt settles it — it carries 248 values, one per test unit, so the
# test fleet is 248 and the readme has the two numbers the wrong way round.
# Left in place deliberately: a parser checked against a document is only as
# good as the document, and this is what that looks like when it shows up.
README_SAYS = {
    "FD001": {"train": 100, "test": 100, "conditions": 1, "faults": 1},
    "FD002": {"train": 260, "test": 259, "conditions": 6, "faults": 1},
    "FD003": {"train": 100, "test": 100, "conditions": 1, "faults": 2},
    "FD004": {"train": 248, "test": 249, "conditions": 6, "faults": 2},
}

# Where the readme is wrong, what the files actually contain, established by
# the RUL file length rather than asserted.
CORRECTIONS = {("FD004", "train"): 249, ("FD004", "test"): 248}

# Sensor channels, in column order, from Saxena et al. (2008) table 1. The
# archive ships the columns unlabelled; these names are what the channels
# are, and several of them turn out to be constant within a subset, which is
# a result rather than a defect.
SENSOR_NAMES = (
    "T2 fan inlet temperature",
    "T24 LPC outlet temperature",
    "T30 HPC outlet temperature",
    "T50 LPT outlet temperature",
    "P2 fan inlet pressure",
    "P15 bypass duct pressure",
    "P30 HPC outlet pressure",
    "Nf physical fan speed",
    "Nc physical core speed",
    "epr engine pressure ratio",
    "Ps30 HPC outlet static pressure",
    "phi fuel flow over Ps30",
    "NRf corrected fan speed",
    "NRc corrected core speed",
    "BPR bypass ratio",
    "farB burner fuel-air ratio",
    "htBleed bleed enthalpy",
    "Nf_dmd demanded fan speed",
    "PCNfR_dmd demanded corrected fan speed",
    "W31 HPT coolant bleed",
    "W32 LPT coolant bleed",
)
SETTING_NAMES = ("altitude setting", "Mach setting", "throttle setting")

N_SENSORS = len(SENSOR_NAMES)
N_SETTINGS = len(SETTING_NAMES)


# -- one engine -----------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One engine's recorded history.

    `settings` is (cycles, 3) and `sensors` is (cycles, 21), both in cycle
    order starting at cycle 1.

    The three lifetime quantities are deliberately kept apart:

    `observed`
        Cycles actually in the record. For a training unit this is the life;
        for a test unit the record stops early and this is only a lower
        bound.
    `failed`
        Whether the record ends in failure. True for train, False for test.
        This pair — `observed`, `failed` — is everything an honest estimator
        may use.
    `true_life`
        The real failure cycle. Known exactly for training units and
        supplied by the archive for test units. For evaluation only; an
        estimator that reads it is cheating, and the tests check that none
        of them do.
    """

    subset: str
    split: str
    unit: int
    settings: np.ndarray
    sensors: np.ndarray
    rul_at_end: int

    @property
    def observed(self) -> int:
        """Cycles in the record."""
        return int(self.sensors.shape[0])

    @property
    def failed(self) -> bool:
        """Did the record end in failure, or was it cut short?"""
        return self.rul_at_end == 0

    @property
    def true_life(self) -> int:
        """Cycle at which this engine actually failed."""
        return self.observed + int(self.rul_at_end)

    def rul(self) -> np.ndarray:
        """True remaining useful life at each recorded cycle."""
        return np.arange(self.true_life - 1, self.rul_at_end - 1, -1,
                         dtype=float)

    def cycles(self) -> np.ndarray:
        return np.arange(1, self.observed + 1, dtype=float)


# -- fetching -------------------------------------------------------------

def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download() -> bytes:
    """The inner CMAPSSData.zip bytes, with both digests verified."""
    request = Request(MIRROR, headers={"User-Agent": "remaining-useful-life"})
    with urlopen(request, timeout=300) as response:
        outer = response.read()

    seen = _digest(outer)
    if seen != DIGESTS["outer"]:
        raise RuntimeError(
            f"mirror returned unexpected content\n"
            f"  expected sha256 {DIGESTS['outer']}\n"
            f"  received sha256 {seen}\n"
            f"  from {MIRROR}")

    inner = zipfile.ZipFile(io.BytesIO(outer)).read(INNER_NAME)
    seen = _digest(inner)
    if seen != DIGESTS["inner"]:
        raise RuntimeError(f"inner archive digest {seen} != {DIGESTS['inner']}")
    return inner


def archive(*, refresh: bool = False) -> zipfile.ZipFile:
    """The dataset archive, downloaded once and kept in `data/`."""
    if refresh or not ARCHIVE.exists():
        payload = _download()
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(payload)
        print(f"  fetched {len(payload) / 1e6:.1f} MB -> {ARCHIVE.name}")
    else:
        payload = ARCHIVE.read_bytes()
        if _digest(payload) != DIGESTS["inner"]:
            raise RuntimeError(
                f"{ARCHIVE} does not match the pinned digest; delete it and "
                f"re-run with --refresh")
    return zipfile.ZipFile(io.BytesIO(payload))


# -- parsing --------------------------------------------------------------

def _parse_table(payload: bytes) -> np.ndarray:
    """A 26-column whitespace table, as floats.

    The files carry trailing spaces on every line, which `np.loadtxt` copes
    with, and no header. Column count is asserted because a short row would
    otherwise shift every sensor by one and still look like data.
    """
    table = np.loadtxt(io.BytesIO(payload))
    expected = 2 + N_SETTINGS + N_SENSORS
    if table.ndim != 2 or table.shape[1] != expected:
        raise ValueError(f"expected {expected} columns, got {table.shape}")
    return table


def _build(cache: dict) -> dict[tuple[str, str], list[Unit]]:
    """Slice the flat tables into per-engine records."""
    out: dict[tuple[str, str], list[Unit]] = {}
    for subset in SUBSETS:
        truth = cache[f"{subset}_rul"].astype(int)
        for split in ("train", "test"):
            table = cache[f"{subset}_{split}"]
            ids = table[:, 0].astype(int)
            if split == "test" and np.unique(ids).size != truth.size:
                # The alignment that everything downstream depends on. The
                # RUL file is positional - value i belongs to test unit i+1 -
                # so a length disagreement would silently attach the wrong
                # truth to every unit rather than fail.
                raise ValueError(
                    f"{subset}: RUL_{subset}.txt has {truth.size} values but "
                    f"test_{subset}.txt has {np.unique(ids).size} units; the "
                    f"two cannot be aligned")
            units: list[Unit] = []
            for index, unit_id in enumerate(np.unique(ids)):
                rows = table[ids == unit_id]
                order = np.argsort(rows[:, 1])          # by cycle
                rows = rows[order]
                if not np.array_equal(rows[:, 1],
                                      np.arange(1, rows.shape[0] + 1)):
                    raise ValueError(
                        f"{subset}/{split} unit {unit_id} has a gap in its "
                        f"cycle numbering")
                units.append(Unit(
                    subset=subset, split=split, unit=int(unit_id),
                    settings=rows[:, 2:2 + N_SETTINGS].copy(),
                    sensors=rows[:, 2 + N_SETTINGS:].copy(),
                    rul_at_end=0 if split == "train" else int(truth[index])))
            out[(subset, split)] = units
    return out


def _cache_arrays(*, refresh: bool = False) -> dict:
    """Flat tables for all four subsets, parsed once and kept as an npz."""
    if CACHE.exists() and not refresh:
        with np.load(CACHE) as stored:
            return {key: stored[key] for key in stored.files}

    source = archive(refresh=refresh)
    cache: dict[str, np.ndarray] = {}
    for subset in SUBSETS:
        for split in ("train", "test"):
            table = _parse_table(source.read(f"{split}_{subset}.txt"))
            cache[f"{subset}_{split}"] = table
            print(f"  {subset} {split:5s}  {table.shape[0]:6d} cycles  "
                  f"{len(np.unique(table[:, 0])):3d} units")
        cache[f"{subset}_rul"] = np.loadtxt(
            io.BytesIO(source.read(f"RUL_{subset}.txt"))).ravel()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **cache)
    return cache


_MEMO: dict[tuple[str, str], list[Unit]] | None = None


def load(subset: str = "FD001", split: str = "train", *,
         refresh: bool = False) -> list[Unit]:
    """Engines of one subset and split, in unit order."""
    global _MEMO
    if _MEMO is None or refresh:
        _MEMO = _build(_cache_arrays(refresh=refresh))
    if subset not in SUBSETS:
        raise KeyError(f"{subset} is not one of {SUBSETS}")
    if split not in ("train", "test"):
        raise KeyError(f"{split} is not 'train' or 'test'")
    return _MEMO[(subset, split)]


def load_all(*, refresh: bool = False) -> dict[tuple[str, str], list[Unit]]:
    """Every subset and split."""
    load(refresh=refresh)
    assert _MEMO is not None
    return _MEMO


# -- what the parse actually produced -------------------------------------

def census(units: list[Unit]) -> dict:
    """Lifetime summary of a fleet, censoring kept separate from failure."""
    observed = np.array([u.observed for u in units], dtype=float)
    failed = np.array([u.failed for u in units], dtype=bool)
    truth = np.array([u.true_life for u in units], dtype=float)
    return {
        "units": len(units),
        "failures": int(failed.sum()),
        "censored": int((~failed).sum()),
        "censoring_rate": float(1.0 - failed.mean()) if units else float("nan"),
        "cycles": int(observed.sum()),
        "observed_min": float(observed.min()) if units else float("nan"),
        "observed_median": float(np.median(observed)) if units else float("nan"),
        "observed_max": float(observed.max()) if units else float("nan"),
        "true_life_median": float(np.median(truth)) if units else float("nan"),
        "true_life_max": float(truth.max()) if units else float("nan"),
    }


def verify_composition() -> list[dict]:
    """Does the parse match the archive's own readme?

    Cheap, and it catches the failure mode that matters most in a parser
    like this one: a subset read into the wrong slot, or a unit lost at a
    page boundary, both of which produce numbers rather than errors.

    It also catches an error in the readme. FD004 disagrees, and the
    disagreement is the document's: `RUL_FD004.txt` holds 248 values, which
    can only be one per test unit, so the 249 train / 248 test split in the
    files is right and the readme has the pair transposed. Reported rather
    than patched over, with `expected` carrying the corrected count.
    """
    rows = []
    fleets = load_all()
    for subset in SUBSETS:
        said = README_SAYS[subset]
        row = {"subset": subset}
        for split in ("train", "test"):
            found = len(fleets[(subset, split)])
            expected = CORRECTIONS.get((subset, split), said[split])
            row[f"{split}_readme"] = said[split]
            row[f"{split}_found"] = found
            row[f"{split}_expected"] = expected
            row[f"{split}_ok"] = found == expected
        row["agrees_with_readme"] = (row["train_found"] == said["train"]
                                     and row["test_found"] == said["test"])
        row["ok"] = row["train_ok"] and row["test_ok"]
        rows.append(row)
    return rows


def channel_variation(units: list[Unit]) -> np.ndarray:
    """Standard deviation of each sensor channel across a whole fleet.

    Pooled over every cycle of every unit. Several channels come back at
    exactly zero in some subsets: they are constant, not merely quiet, and
    which ones differs by subset. `sensors.py` makes that the subject of an
    analysis rather than dropping them here.
    """
    stacked = np.vstack([u.sensors for u in units])
    return stacked.std(axis=0)


SCATTER_TOLERANCE = 1e-3
MAX_REGIMES = 8


def regime_scatter(units: list[Unit], *, seed: int = 0,
                   limit: int = MAX_REGIMES) -> dict[int, float]:
    """Within-cluster scatter of the operational settings against k.

    Each value is the k-means inertia as a fraction of the inertia at k=1,
    so 1.0 is "no structure found yet" and 0.0 is "every point sits exactly
    on a centre".
    """
    stacked = np.vstack([u.settings for u in units])
    scale = stacked.std(axis=0)
    scale[scale == 0] = 1.0
    points = stacked / scale
    inertias = {k: _kmeans(points, k, seed)[2] for k in range(1, limit + 1)}
    total = inertias[1] if inertias[1] > 0 else 1.0
    return {k: value / total for k, value in inertias.items()}


def count_regimes(units: list[Unit], *, seed: int = 0,
                  tolerance: float = SCATTER_TOLERANCE) -> int:
    """How many discrete operating regimes does this fleet actually run?

    Not an elbow rule, because it does not need to be. The subsets that run
    several conditions hold them at a few discrete nominal flight
    conditions, so the within-cluster scatter does not merely bend at the
    right k — it falls off a cliff. On FD002 the sequence of relative
    scatter runs

        k=1  1.0e+00   k=2  2.6e-01   k=3  6.1e-02   k=4  2.0e-02
        k=5  4.1e-03   k=6  8.5e-07   k=7  7.5e-07   k=8  5.7e-07

    Three things in that row, together, identify six without a judgement
    call: the drop from k=5 to k=6 is nearly four orders of magnitude, the
    k=6 value is six orders below the k=1 value, and k=7 and k=8 buy almost
    nothing, so there is no seventh group left to find.

    What survives at k=6 is not cluster structure but dither. The six
    centres sit at (altitude, Mach, throttle) of roughly (0, 0.0, 100),
    (10, 0.25, 100), (20, 0.70, 100), (25, 0.62, 60), (35, 0.84, 100) and
    (42, 0.84, 100); within a regime the throttle is *exactly* constant and
    altitude and Mach carry a standard deviation of about 0.003 and 0.0006,
    which is simulated instrument noise on a held setpoint. So the residual
    is the noise floor, and calling it zero -- which four-decimal output
    invites -- would be wrong by six orders of magnitude rather than
    harmless.

    On FD001 and FD003 the same sequence decays smoothly -- 1.00, 0.50,
    0.32, 0.23, 0.19, 0.16 -- close to the 1/k decay you get from
    clustering noise with no cluster structure in it, and it never
    approaches the tolerance. There is one operating point and the settings
    jitter around it. Returning 1 is then the honest answer rather than a
    threshold artefact.

    So the rule is: the smallest k whose scatter falls below `tolerance`,
    and 1 if none does. The tolerance sits in a gap four orders of
    magnitude wide, so nothing here is sensitive to its exact value.
    """
    scatter = regime_scatter(units, seed=seed)
    for k in sorted(scatter):
        if scatter[k] < tolerance:
            return k
    return 1


def regime_labels(units: list[Unit], *, regimes: int | None = None,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Assign every recorded cycle to an operating regime.

    The regime count is recovered by `count_regimes` rather than taken from
    the readme, so the published "SIX" and "ONE" become results that agree
    rather than assumptions. Returns the per-cycle label vector for the
    fleet stacked in unit order, and the regime centres in original units.
    """
    stacked = np.vstack([u.settings for u in units])
    scale = stacked.std(axis=0)
    scale[scale == 0] = 1.0
    points = stacked / scale

    if regimes is None:
        regimes = count_regimes(units, seed=seed)

    labels, centres, _ = _kmeans(points, regimes, seed)
    return labels, centres * scale


def _kmeans(points: np.ndarray, k: int, seed: int,
            iterations: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's algorithm with a k-means++ start.

    Written out rather than imported because scipy's `kmeans2` picks its
    own restart policy, and the regime count is a reported result: it has to
    be reproducible from a stated seed.
    """
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(points.shape[0])]]
    for _ in range(1, k):
        gap = np.min(
            ((points[:, None, :] - np.array(centres)[None, :, :]) ** 2
             ).sum(axis=2), axis=1)
        total = gap.sum()
        pick = (rng.integers(points.shape[0]) if total <= 0
                else rng.choice(points.shape[0], p=gap / total))
        centres.append(points[pick])
    centre = np.array(centres)

    labels = np.zeros(points.shape[0], dtype=int)
    for _ in range(iterations):
        distance = ((points[:, None, :] - centre[None, :, :]) ** 2).sum(axis=2)
        fresh = np.argmin(distance, axis=1)
        if np.array_equal(fresh, labels):
            break
        labels = fresh
        for index in range(k):
            member = points[labels == index]
            if member.size:
                centre[index] = member.mean(axis=0)
    distance = ((points[:, None, :] - centre[None, :, :]) ** 2).sum(axis=2)
    inertia = float(np.min(distance, axis=1).sum())
    return labels, centre, inertia


if __name__ == "__main__":
    print(f"source {MIRROR}")
    fleets = load_all(refresh=True)
    print("\ncomposition: found / readme")
    for row in verify_composition():
        note = "" if row["agrees_with_readme"] else "   <- readme is wrong here"
        print(f"  {row['subset']}  train {row['train_found']:3d}/"
              f"{row['train_readme']:3d}  test {row['test_found']:3d}/"
              f"{row['test_readme']:3d}{note}")
    print("\nlifetimes")
    for key in sorted(fleets):
        summary = census(fleets[key])
        print(f"  {key[0]} {key[1]:5s}  n={summary['units']:3d}  "
              f"failures={summary['failures']:3d}  "
              f"censored={summary['censored']:3d}  "
              f"observed {summary['observed_min']:.0f}-"
              f"{summary['observed_max']:.0f} "
              f"(median {summary['observed_median']:.0f})  "
              f"true median {summary['true_life_median']:.0f}")
