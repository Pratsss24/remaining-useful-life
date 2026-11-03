"""Tests for the data layer.

A parser for a fixed-format text archive fails in one characteristic way:
it produces a table of plausible numbers that are wrong by a shift. None of
that raises, and none of it is visible in a summary statistic. So these
tests check structure and invariants rather than values:

* **Alignment.** `RUL_FDxxx.txt` is positional, so value *i* belongs to
  test unit *i+1*. If that correspondence slips, every reported error is
  wrong and nothing complains. Checked by length, by the independent
  arithmetic `true_life = observed + rul_at_end`, and against the one
  ground truth available without the RUL file at all: a training unit's
  remaining life at its last cycle is exactly zero.
* **Agreement with the archive's own documentation**, where the
  documentation is right, and a pinned expectation of the disagreement
  where it is not. The readme has FD004's train and test counts the wrong
  way round; that is asserted as a known fact about the archive so it
  cannot silently change under us.
* **Recovery of a documented quantity.** The operating-regime count is
  derived by clustering and must land on the readme's "ONE" and "SIX"
  without being told them.
* **Separation of truth from evidence.** `observed` and `failed` are what
  an estimator may use; `true_life` is not. The test fleet must be
  entirely censored and the train fleet entirely failed, because that
  distinction is what the rest of the project rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data          # noqa: E402


@pytest.fixture(scope="module")
def fleets():
    return data.load_all()


# -- structure ------------------------------------------------------------

def test_every_subset_and_split_is_present(fleets):
    assert set(fleets) == {(s, p) for s in data.SUBSETS
                           for p in ("train", "test")}


def test_arrays_have_the_documented_widths(fleets):
    for units in fleets.values():
        for unit in units:
            assert unit.settings.shape == (unit.observed, data.N_SETTINGS)
            assert unit.sensors.shape == (unit.observed, data.N_SENSORS)


def test_cycles_start_at_one_and_have_no_gaps(fleets):
    """Enforced in the parser; asserted here because a gap would make
    `observed` and the last cycle number disagree, and downstream code
    uses them interchangeably."""
    for units in fleets.values():
        for unit in units:
            assert unit.cycles()[0] == 1.0
            assert unit.cycles()[-1] == unit.observed


def test_unit_numbers_are_contiguous_from_one(fleets):
    for units in fleets.values():
        assert [u.unit for u in units] == list(range(1, len(units) + 1))


# -- the censoring structure ----------------------------------------------

def test_training_units_all_failed_and_test_units_are_all_censored(fleets):
    """The entire point of the dataset. If this ever flips, the censoring
    argument in the report is about nothing."""
    for subset in data.SUBSETS:
        assert all(u.failed for u in fleets[(subset, "train")])
        assert not any(u.failed for u in fleets[(subset, "test")])


def test_a_failed_unit_has_no_life_left_at_its_last_cycle(fleets):
    """The one piece of ground truth that owes nothing to the RUL files:
    a run-to-failure record ends at failure, so remaining life there is
    exactly zero."""
    for subset in data.SUBSETS:
        for unit in fleets[(subset, "train")]:
            assert unit.rul_at_end == 0
            assert unit.rul()[-1] == 0.0
            assert unit.true_life == unit.observed


def test_true_life_is_observed_plus_remaining(fleets):
    for units in fleets.values():
        for unit in units:
            assert unit.true_life == unit.observed + unit.rul_at_end


def test_remaining_life_counts_down_by_one_each_cycle(fleets):
    for units in fleets.values():
        for unit in units:
            remaining = unit.rul()
            assert remaining.size == unit.observed
            assert remaining[-1] == unit.rul_at_end
            assert np.array_equal(np.diff(remaining),
                                  -np.ones(unit.observed - 1))


def test_censored_units_outlive_their_records(fleets):
    """A right-censored observation is a lower bound on the lifetime, so
    the truth must sit strictly beyond it."""
    for subset in data.SUBSETS:
        for unit in fleets[(subset, "test")]:
            assert unit.rul_at_end > 0
            assert unit.true_life > unit.observed


def test_the_rul_file_has_one_value_per_test_unit(fleets):
    with np.load(data.CACHE) as stored:
        for subset in data.SUBSETS:
            assert (stored[f"{subset}_rul"].size
                    == len(fleets[(subset, "test")]))


# -- against the archive's documentation ----------------------------------

def test_the_parse_matches_the_corrected_composition():
    for row in data.verify_composition():
        assert row["ok"], row


def test_the_readme_is_wrong_about_fd004_and_right_about_the_rest():
    """Pinned so the discrepancy reported in the README stays a fact rather
    than a memory. If the mirror ever ships a corrected readme, this fails
    and the write-up gets revisited."""
    rows = {row["subset"]: row for row in data.verify_composition()}
    assert rows["FD004"]["agrees_with_readme"] is False
    assert rows["FD004"]["train_found"] == 249
    assert rows["FD004"]["test_found"] == 248
    assert rows["FD004"]["train_readme"] == 248
    assert rows["FD004"]["test_readme"] == 249
    for subset in ("FD001", "FD002", "FD003"):
        assert rows[subset]["agrees_with_readme"] is True


def test_the_regime_count_is_recovered_not_assumed(fleets):
    """The readme's "ONE" and "SIX" come back out of the settings."""
    for subset in data.SUBSETS:
        found = data.count_regimes(fleets[(subset, "train")])
        assert found == data.README_SAYS[subset]["conditions"], subset


def test_the_multi_regime_subsets_have_discrete_settings(fleets):
    """Why the regime count needs no elbow rule: the k=5 to k=6 drop is
    nearly four orders of magnitude, and k=7 then buys almost nothing.

    Deliberately not asserted as exactly zero -- it is about 8e-7, the
    dither on a held setpoint -- because four-decimal output makes it look
    exact and it is not."""
    for subset in ("FD002", "FD004"):
        scatter = data.regime_scatter(fleets[(subset, "train")])
        assert scatter[5] > 1e-3
        assert scatter[6] < 1e-5
        assert scatter[5] / scatter[6] > 1e3
        # Nothing left to split: a seventh centre barely moves it.
        assert scatter[7] / scatter[6] > 0.5


def test_within_a_regime_the_throttle_is_held_exactly(fleets):
    """The dither that survives clustering is on altitude and Mach only,
    which is what makes it read as instrument noise on a setpoint rather
    than as a regime that was missed."""
    for subset in ("FD002", "FD004"):
        units = fleets[(subset, "train")]
        labels, _ = data.regime_labels(units)
        settings = np.vstack([u.settings for u in units])
        for index in np.unique(labels):
            spread = settings[labels == index].std(axis=0)
            assert spread[2] == 0.0
            assert spread[0] < 0.01
            assert spread[1] < 0.01


def test_the_single_regime_subsets_never_find_structure(fleets):
    """And why returning 1 there is not a threshold artefact: the scatter
    decays smoothly and never approaches zero."""
    for subset in ("FD001", "FD003"):
        scatter = data.regime_scatter(fleets[(subset, "train")])
        assert min(scatter.values()) > 0.05


# -- the sensor channels --------------------------------------------------

def test_some_channels_are_constant_in_the_single_regime_subsets(fleets):
    """Flagged, not dropped. `sensors.py` makes which-ones-where the
    result; this only pins that the situation exists."""
    for subset in ("FD001", "FD003"):
        spread = data.channel_variation(fleets[(subset, "train")])
        assert (spread == 0).sum() >= 2


def test_no_channel_is_constant_once_the_regimes_move(fleets):
    """In FD002 and FD004 the operating point moves, so every channel
    varies -- including the ones that carry no degradation signal at all.
    Constant-channel screening and useful-channel screening are therefore
    different questions."""
    for subset in ("FD002", "FD004"):
        spread = data.channel_variation(fleets[(subset, "train")])
        assert (spread == 0).sum() == 0


# -- the cache ------------------------------------------------------------

def test_loading_twice_gives_the_same_objects(fleets):
    """Memoised, so the 10 MB parse happens once per process."""
    assert data.load("FD001", "train") is fleets[("FD001", "train")]


def test_the_cached_archive_matches_its_pinned_digest():
    import hashlib
    if not data.ARCHIVE.exists():
        pytest.skip("archive not fetched")
    seen = hashlib.sha256(data.ARCHIVE.read_bytes()).hexdigest()
    assert seen == data.DIGESTS["inner"]
