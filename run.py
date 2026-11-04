"""Remaining useful life on the NASA C-MAPSS turbofan fleets.

    python run.py                 # everything (~3 min)
    python run.py --refresh       # re-fetch and re-parse the archive
    python run.py --subset FD002  # lead on a different fleet
    python run.py --select        # re-run the cross-validated selection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import agelaw          # noqa: E402
import charts          # noqa: E402
import data            # noqa: E402
import degradation     # noqa: E402
import evaluate        # noqa: E402
import lifetimes       # noqa: E402
import report          # noqa: E402
import sensors         # noqa: E402
import simulate        # noqa: E402
import survival        # noqa: E402

# Settled by `degradation.select` on the training fleets alone, and pinned
# so a plain run does not pay for the cross-validation. `--select` re-runs
# it and prints whether it still agrees.
CAPS = {"FD001": 100, "FD002": 100, "FD003": 100, "FD004": 140}
WINDOW = 5
NOISE_LEVELS = (0.10, 0.25, 0.60, 1.50, 3.00)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--subset", default="FD001", choices=data.SUBSETS)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "out"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    lead = args.subset

    # -- the data ---------------------------------------------------------
    if args.refresh:
        data.load_all(refresh=True)
    fleets = data.load_all()
    composition = data.verify_composition()

    print("composition: found / readme")
    for row in composition:
        note = "" if row["agrees_with_readme"] else "   <- readme is wrong"
        print(f"  {row['subset']}  train {row['train_found']:3d}/"
              f"{row['train_readme']:3d}  test {row['test_found']:3d}/"
              f"{row['test_readme']:3d}{note}")

    print("\noperating regimes, recovered from the settings")
    for subset in data.SUBSETS:
        found = data.count_regimes(fleets[(subset, "train")])
        said = data.README_SAYS[subset]["conditions"]
        print(f"  {subset}  {found} recovered, {said} documented  "
              f"{'agree' if found == said else 'DISAGREE'}")

    lives = lifetimes.pooled(lead)
    census = lifetimes.census(lives)
    print(f"\n{lead} pooled fleet: {census['units']} engines, "
          f"{census['failures']} run to failure, "
          f"{census['censored']} censored ({census['censoring_rate']:.0%})")
    print(f"  records cover {census['coverage']:.1%} of the fleet's real "
          f"service life; {census['hidden_cycles']:.0f} cycles are withheld "
          f"by censoring")

    # -- the sensors ------------------------------------------------------
    screening = sensors.masking_report()
    print("\nchannel screening, pooled against within regime")
    for row in screening:
        print(f"  {row['subset']}  {row['regimes']} regime(s)  "
              f"informative {row['pooled_informative']:2d}/21 pooled, "
              f"{row['regime_informative']:2d}/21 within regime  "
              f"({row['masked']} masked)")

    # -- censoring, against a planted truth -------------------------------
    censoring_rows = simulate.censoring_recovery()
    dependence = simulate.measured_censoring_dependence(lives)
    print(f"\nreal censoring: cut-off correlates with true lifetime at "
          f"{dependence['correlation']:+.3f} "
          f"(p {dependence['p_value']:.1e}), truncation fraction "
          f"{dependence['fraction_mean']:.2f} "
          f"+/- {dependence['fraction_sd']:.2f}")
    print("recovering a planted lifetime law")
    current = None
    for row in censoring_rows:
        if row["scheme"] != current:
            current = row["scheme"]
            print(f"  {current}  (censoring {row['censoring_rate']:.0%})")
        print(f"    {row['treatment']:14s} shape {row['beta_error']:+7.2%}   "
              f"scale {row['eta_error']:+7.2%}")

    treatment_rows = evaluate.censoring_effect(lead)
    print(f"\n{lead}: what each treatment does to the predictions")
    for row in treatment_rows:
        print(f"  {row['treatment']:14s} beta {row['beta']:5.2f}  "
              f"eta {row['eta']:6.1f}  MAE {row['mae']:6.2f}  "
              f"bias {row['bias']:+6.2f}  cover 90% {row['coverage']:5.1%}")

    # -- the estimators ---------------------------------------------------
    if args.select:
        print("\ncross-validated selection, on the training fleets only")
        for subset in data.SUBSETS:
            chosen = degradation.select(subset)
            agree = ("agree" if chosen["criteria_agree"]
                     else "one grid step apart")
            print(f"  {subset}  cap {chosen['best_cap']} "
                  f"(consistency {chosen['cap_by_consistency']}, "
                  f"fit {chosen['cap_by_fit']}: {agree})  "
                  f"window {chosen['best_window']}  "
                  f"pinned {CAPS[subset]}/{WINDOW}")

    found = evaluate.headline(lead, cap=CAPS[lead], window=WINDOW)
    law, model = found["law"], found["model"]
    selection = degradation.select(lead) if args.select else None
    if selection is None:
        # The figure needs the criteria table; it is cheap next to the rest.
        selection = degradation.select(lead)

    low, high = law.beta_confidence()
    print(f"\n{lead} estimator A, age only")
    print(f"  Weibull beta {law.beta:.4f} ({low:.3f}-{high:.3f})  "
          f"eta {law.eta:.2f} cycles  -> {law.regime}")
    print(f"  B10 life {law.b10_life:.1f} cycles, "
          f"median life {law.median_life:.1f}")

    index = model.index
    print(f"\n{lead} estimator B, degradation signal")
    print(f"  cap {model.cap} cycles, window {model.window}, "
          f"{len(index.channels)} channels, index R2 {index.r_squared:.4f}")
    print(f"  index at a healthy start {index.start_level:+.3f}, "
          f"at failure {index.failure_level:+.3f} "
          f"-> spans 0 to 1: {index.spans_unit_interval}")

    for name, rows in (("A, age only", found["age_law_bands"]),
                       ("B, degradation signal",
                        found["degradation_bands"])):
        print(f"\n  {name}: error by true remaining life")
        print(f"    {'band':>8} {'n':>4} {'MAE':>6} {'bias':>7} "
              f"{'late':>5}  cover 50/80/90/95")
        for row in rows:
            print(f"    {row['band']:>8} {row['n']:4d} {row['mae']:6.1f} "
                  f"{row['bias']:+7.1f} {row['late_share']:5.0%}  "
                  f"{row['coverage_50']:5.0%} {row['coverage_80']:5.0%} "
                  f"{row['coverage_90']:5.0%} {row['coverage_95']:5.0%}")

    print("\n  head to head, mean absolute error in cycles")
    for row in found["comparison"]:
        print(f"    {row['band']:>8} {row['n']:4d}  A {row['age law mae']:6.1f}"
              f"   B {row['degradation mae']:6.1f}   -> {row['winner']}")
    gap = found["disagreement"]
    print(f"  they disagree by {gap['mean_absolute_gap']:.1f} cycles on "
          f"average, correlation {gap['correlation']:+.3f}")

    # -- all four subsets --------------------------------------------------
    subset_rows = []
    for subset in data.SUBSETS:
        each = (found if subset == lead
                else evaluate.headline(subset, cap=CAPS[subset],
                                       window=WINDOW))
        bands_a = {row["band"]: row for row in each["age_law_bands"]}
        bands_b = {row["band"]: row for row in each["degradation_bands"]}
        subset_rows.append({
            "subset": subset,
            "near_a": bands_a["0-25"]["mae"],
            "near_b": bands_b["0-25"]["mae"],
            "pooled_a": bands_a["all"]["mae"],
            "pooled_b": bands_b["all"]["mae"],
            "near_cover_a": bands_a["0-25"]["coverage_90"],
            "pooled_cover_a": bands_a["all"]["coverage_90"],
            "near_cover_b": bands_b["0-25"]["coverage_90"],
        })

    print("\nall four fleets, mean absolute error in cycles")
    for row in subset_rows:
        print(f"  {row['subset']}  near failure A {row['near_a']:5.1f} "
              f"B {row['near_b']:5.1f} ({row['near_a'] / row['near_b']:.1f}x)"
              f"   pooled A {row['pooled_a']:5.1f} B {row['pooled_b']:5.1f}"
              f"   A cover 90% {row['near_cover_a']:4.0%} near / "
              f"{row['pooled_cover_a']:4.0%} pooled")

    # -- planted truth for estimator B ------------------------------------
    recovery_rows = []
    for noise in NOISE_LEVELS:
        fleet = simulate.degradation_fleet(180, noise=noise, seed=1)
        train, test = fleet[:90], simulate.truncate_fleet(fleet[90:], seed=5)
        labels = np.zeros(sum(u.observed for u in train), dtype=int)
        normaliser = sensors.Normaliser.fit(train, labels)
        channels = tuple(range(11))
        prepared = degradation.prepare(train, normaliser, channels)

        gaps = {}
        for cap in degradation.CAP_GRID:
            candidate = degradation.fit_index(
                prepared, subset="SIM", cap=cap, channels=channels,
                normaliser=normaliser)
            gaps[cap] = (abs(candidate.start_level - 1.0)
                         + abs(candidate.failure_level))
        recovered = min(gaps, key=lambda cap: gaps[cap])

        simulated = degradation.fit_index(
            prepared, subset="SIM", cap=100, channels=channels,
            normaliser=normaliser)
        calibration = degradation.fit_calibration(simulated, prepared,
                                                  window=WINDOW)
        health = np.array([
            degradation.trailing_mean(simulated.apply(u), WINDOW)[-1]
            for u in test])
        truth = np.array([u.rul_at_end for u in test], dtype=float)
        error = calibration.quantile(health, 0.5) - truth
        inside = truth < 100
        covered = ((truth >= calibration.quantile(health, 0.05))
                   & (truth <= calibration.quantile(health, 0.95)))
        recovery_rows.append({
            "noise": noise, "cap": recovered,
            "r_squared": simulated.r_squared,
            "mae_inside": float(np.abs(error[inside]).mean()),
            "coverage": float(covered.mean())})

    print("\nrecovering a planted health law (true horizon 100 cycles)")
    for row in recovery_rows:
        print(f"  noise {row['noise']:4.2f}  cap found {row['cap']:3d}  "
              f"index R2 {row['r_squared']:.3f}  "
              f"MAE inside horizon {row['mae_inside']:5.1f}  "
              f"cover 90% {row['coverage']:5.1%}")

    # -- output ------------------------------------------------------------
    time, failed = lifetimes.arrays(lives)
    curve = survival.kaplan_meier(time, failed, name=lead)

    figures = {
        "error_by_band": charts.error_by_band(found, out_dir),
        "coverage": charts.coverage(found, out_dir),
        "health_index": charts.health_index(model, lead, out_dir),
        "cap_selection": charts.cap_selection(selection, out_dir),
        "sensor_masking": charts.sensor_masking(screening, out_dir),
        "censoring_recovery": charts.censoring_recovery(censoring_rows,
                                                        out_dir),
        "survival_fit": charts.survival_fit(curve, law, out_dir),
        "disagreement": charts.disagreement(found, out_dir),
        "extrapolation_failure": charts.extrapolation_failure(
            model, lead, out_dir),
        "across_subsets": charts.across_subsets(subset_rows, out_dir),
    }

    report_path = report.build(
        subset=lead, found=found, census=census, composition=composition,
        screening=screening, censoring_rows=censoring_rows,
        treatment_rows=treatment_rows, dependence=dependence,
        subset_rows=subset_rows, recovery_rows=recovery_rows,
        selection=selection, figures=figures,
        out_path=out_dir / "report.pdf")

    print(f"\nreport  {report_path}")
    for path in figures.values():
        print(f"figure  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
