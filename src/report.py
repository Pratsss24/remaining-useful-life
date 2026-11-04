"""Build the report as a printable PDF.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper


def _bands(rows) -> tuple[list[str], list[list[str]]]:
    out = [[row["band"], str(row["n"]), f"{row['mae']:.1f}",
            f"{row['bias']:+.1f}", f"{row['late_share']:.0%}",
            f"{row['coverage_50']:.0%}", f"{row['coverage_90']:.0%}"]
           for row in rows]
    return (["true RUL", "n", "MAE", "bias", "optimistic",
             "cover 50%", "cover 90%"], out)


def _head_to_head(rows) -> tuple[list[str], list[list[str]]]:
    out = [[row["band"], str(row["n"]), f"{row['age law mae']:.1f}",
            f"{row['degradation mae']:.1f}", row["winner"]]
           for row in rows]
    return (["true RUL", "n", "A, age only", "B, degradation", "closer"],
            out)


def _censoring(rows) -> tuple[list[str], list[list[str]]]:
    """One row per treatment, one column pair per scheme.

    Pivoted rather than listed, so the comparison the section is about --
    the same treatment under different censoring mechanisms -- is a row
    the eye can follow, and so twelve rows do not spill onto a second
    page for the sake of six of them.
    """
    schemes = ["independent", "proportional", "end_of_test"]
    treatments = ["censored MLE", "drop censored", "as failures", "oracle"]
    lookup = {(row["scheme"], row["treatment"]): row for row in rows}

    out = []
    for treatment in treatments:
        line = [treatment]
        for scheme in schemes:
            row = lookup[(scheme, treatment)]
            line.append(f"{row['beta_error']:+.1%}")
            line.append(f"{row['eta_error']:+.1%}")
        out.append(line)
    return (["treatment", "indep. shape", "indep. scale",
             "propn. shape", "propn. scale", "cut-off shape",
             "cut-off scale"], out)


def _treatments(rows) -> tuple[list[str], list[list[str]]]:
    out = [[row["treatment"], f"{row['beta']:.2f}", f"{row['eta']:.1f}",
            f"{row['mae']:.1f}", f"{row['bias']:+.1f}",
            f"{row['coverage']:.0%}"] for row in rows]
    return (["treatment", "beta", "eta", "MAE", "bias", "cover 90%"], out)


def _screening(rows) -> tuple[list[str], list[list[str]]]:
    out = [[row["subset"], str(row["regimes"]),
            f"{row['pooled_informative']} of 21",
            f"{row['regime_informative']} of 21",
            str(row["masked"]), f"{row['pooled_max']:.2f}",
            f"{row['regime_max']:.2f}"] for row in rows]
    return (["subset", "regimes", "informative pooled",
             "informative in regime", "masked", "max pooled |rho|",
             "max regime |rho|"], out)


def _subsets(rows) -> tuple[list[str], list[list[str]]]:
    out = [[row["subset"], f"{row['near_a']:.1f}", f"{row['near_b']:.1f}",
            f"{row['pooled_a']:.1f}", f"{row['pooled_b']:.1f}",
            f"{row['near_cover_a']:.0%}", f"{row['pooled_cover_a']:.0%}"]
           for row in rows]
    return (["subset", "A near failure", "B near failure", "A pooled",
             "B pooled", "A cover 90% near", "A cover 90% pooled"], out)


def _composition(rows) -> tuple[list[str], list[list[str]]]:
    out = [[row["subset"], f"{row['train_found']}", f"{row['test_found']}",
            f"{row['train_readme']}", f"{row['test_readme']}",
            "agrees" if row["agrees_with_readme"] else "readme is wrong"]
           for row in rows]
    return (["subset", "train found", "test found", "train in readme",
             "test in readme", ""], out)


def _recovery(rows) -> tuple[list[str], list[list[str]]]:
    out = [[f"{row['noise']:.2f}", f"{row['cap']}", f"{row['r_squared']:.3f}",
            f"{row['mae_inside']:.1f}", f"{row['coverage']:.0%}"]
           for row in rows]
    return (["sensor noise", "cap found", "index R2",
             "MAE inside horizon", "cover 90%"], out)


def build(*, subset: str, found: dict, census: dict, composition: list,
          screening: list, censoring_rows: list, treatment_rows: list,
          dependence: dict, subset_rows: list, recovery_rows: list,
          selection: dict, figures: dict, out_path: Path) -> Path:
    """Compose the PDF."""
    law = found["law"]
    model = found["model"]
    bands_a = {row["band"]: row for row in found["age_law_bands"]}
    bands_b = {row["band"]: row for row in found["degradation_bands"]}
    pooled_a, pooled_b = bands_a["all"], bands_b["all"]
    near_a, near_b = bands_a["0-25"], bands_b["0-25"]
    worst = min(subset_rows, key=lambda row: row["near_cover_a"])

    with Paper(out_path,
               "Remaining useful life, and how far ahead you asked",
               "Two independent estimators on the NASA C-MAPSS turbofan "
               "run-to-failure fleets, with coverage reported as measured"
               ) as paper:

        paper.cover(
            headline=(
                f"On the engines closest to failure the sensor-driven "
                f"estimator is {near_a['mae'] / near_b['mae']:.1f} times more "
                f"accurate than the age-only one, whose nominal 50% interval "
                f"covers {near_a['coverage_50']:.0%} of them."),
            small=(
                f"{subset}: {pooled_a['n']} censored engines predicted from a "
                f"fleet of {census['failures']} run to failure. Pooled mean "
                f"absolute error {pooled_a['mae']:.0f} cycles for the age law "
                f"against {pooled_b['mae']:.0f} for the degradation signal. "
                f"On {worst['subset']}, the hardest fleet, the age law's "
                f"nominal 90% interval covers {worst['near_cover_a']:.0%} of "
                f"the engines nearest failure."),
            tiles=[
                ("B near failure", f"{near_b['mae']:.0f} cycles"),
                ("A near failure", f"{near_a['mae']:.0f} cycles"),
                ("A cover 50%, pooled", f"{pooled_a['coverage_50']:.0%}"),
                ("A cover 50%, near failure",
                 f"{near_a['coverage_50']:.0%}"),
            ],
            blurb=(
                "Remaining-life accuracy depends almost entirely on how far "
                "ahead the question is asked, so a single error number for a "
                "fleet is close to meaningless. Every figure here is reported "
                "against the true remaining life, and every prediction "
                "interval is reported at the coverage it was measured to "
                "achieve rather than the coverage it claims."),
            figure_path=figures["error_by_band"])

        paper.section(
            "The question, and why one number cannot answer it",
            body=(
                "An engine is on the wing and the question is how much "
                "longer it can stay there. The C-MAPSS archive supplies four "
                "simulated fleets: one set of engines run to failure, and a "
                "second set whose records stop before failure, with the true "
                "remaining life supplied separately. So predictions can be "
                "scored against truth, and the censored half is genuinely "
                "censored rather than merely incomplete.\n\n"
                "The difficulty of the question is not constant. Saying that "
                "an engine twelve cycles from failure has about twelve "
                "cycles left is a different task from saying the same of one "
                "two hundred cycles out, and the two errors point in "
                "opposite directions. Pooled over this fleet the age-only "
                f"estimator's bias is {pooled_a['bias']:+.0f} cycles, which "
                f"is the average of {near_a['bias']:+.0f} on the engines "
                "closest to failure and a negative bias on those furthest "
                "from it. It describes neither group."),
            note=f"{subset}, {pooled_a['n']} censored engines",
            table=_bands(found["age_law_bands"]),
            after=(
                "Error and measured coverage for the age-only estimator, "
                "banded by true remaining life. The last row is the pooled "
                "figure, printed to show what it conceals rather than "
                "instead of the bands."))

        paper.section(
            "Where the data came from",
            body=(
                "The dataset is NASA's Commercial Modular Aero-Propulsion "
                "System Simulation run-to-failure study. Its canonical home "
                "at the Prognostics Center of Excellence no longer serves "
                "the file: the URL redirects to a landing page with no "
                "download on it. The copy used here is the PHM Data "
                "Challenge S3 mirror, and the SHA-256 digests of both zip "
                "layers are pinned in the code, so a substituted mirror "
                "fails loudly instead of quietly changing the numbers.\n\n"
                "Checking the parse against the archive's own readme rather "
                "than trusting it turned up an error in the documentation. "
                "The readme gives FD004 as 248 training and 249 test units; "
                "the files hold 249 and 248. RUL_FD004.txt settles it with "
                "248 values, one per test unit, so the readme has the pair "
                "transposed."),
            note="composition: what the files hold against what the readme "
                 "claims",
            table=_composition(composition),
            after=(
                "The operating-regime count is also recovered rather than "
                "assumed, and agrees with the readme's ONE and SIX. Within "
                "a regime the throttle is held exactly constant and the "
                "altitude and Mach settings carry about 0.003 and 0.0006 of "
                "dither, so the regimes are discrete flight conditions and "
                "not fuzzy clusters."))

        paper.section(
            "Estimator A: the fleet lifetime law",
            body=(
                "The first estimator is told an engine's age and nothing "
                "else. A two-parameter Weibull is fitted to the fleet's "
                "lifetimes by right-censored maximum likelihood, and "
                "remaining life follows from conditioning that law on "
                "survival to the engine's current age. The inversion is "
                "closed-form, so the point prediction and the interval come "
                "from one expression with no simulation in it.\n\n"
                f"The fitted shape is {law.beta:.2f}, comfortably above one "
                f"and with a confidence interval excluding it, so the fleet "
                f"is in wear-out: hazard rises with age, which is what a "
                f"growing fault should look like. The scale is "
                f"{law.eta:.0f} cycles.\n\n"
                "This estimator is deliberately the weaker of the two and "
                "deliberately the one that cannot be accused of leaking "
                "anything. Two engines of the same age always receive the "
                "same answer, so its error has a floor set by the spread of "
                "the lifetime distribution itself. It is the floor the "
                "second estimator has to beat."),
            note=f"{subset}, fitted on the {census['failures']} "
                 f"run-to-failure engines",
            figure_path=figures["survival_fit"])

        paper.section(
            "Censoring: the right treatment depends on the mechanism",
            body=(
                "Half of each pooled fleet is right-censored. Those records "
                "are not missing data; each one says \"this engine was still "
                "running at cycle 134\", and the censored likelihood is how "
                "that gets used. Because C-MAPSS supplies the censored "
                "units' true failure cycles, the cost of mishandling them "
                "can be measured against an oracle rather than argued "
                "about.\n\n"
                "The measurement did not agree with the textbook. Discarding "
                "the censored units, supposedly the blunder, came closest to "
                "the oracle, and the correct likelihood came out biased long "
                "by five to fifteen per cent on the scale. Simulation from a "
                "planted truth explains both. C-MAPSS cuts each test record "
                "at a random fraction of its own life, so the cut-off "
                f"correlates with the true lifetime at "
                f"{dependence['correlation']:+.2f} here — which is precisely "
                "what non-informative censoring rules out, and it biases the "
                "likelihood long. And because the censored units are a "
                "random half of a fleet whose lifetime distribution matches "
                "the half retained, discarding them costs precision and "
                "biases nothing."),
            note="recovering a known lifetime law, 40,000 simulated units",
            table=_censoring(censoring_rows),
            warn=(
                "This is not a licence to drop censored data. Under the "
                "censoring an operator actually faces — a common cut-off, "
                "where the censored engines are the long-lived tail — the "
                "same shortcut recovers a shape 43% too high and a scale 22% "
                "too low, while the censored likelihood stays within half a "
                "per cent. Across all three schemes the censored MLE is the "
                "only treatment never badly wrong."),
            figure_path=figures["censoring_recovery"])

        paper.section(
            "What the treatments do to the predictions",
            body=(
                "Carried through from the fitted parameters to the "
                "predictions anyone would act on, the ranking inverts. "
                "Counting the survivors as failures gives much the worst fit "
                "of the lifetime law, and the best remaining-life "
                "predictions of the three available treatments — better even "
                "than the oracle.\n\n"
                "Two errors partially cancelling. The age-only estimator "
                "over-predicts remaining life when fed the correct law, and "
                "a law biased short pushes the predictions back down. The "
                "lesson is not that mishandling censoring is harmless. It is "
                "that downstream prediction error cannot decide whether the "
                "censoring was handled correctly; that question is settled "
                "on the parameters, against a planted truth."),
            note=f"{subset}, lifetime law fitted on the pooled fleet under "
                 f"each treatment",
            table=_treatments(treatment_rows),
            after=(
                "The oracle row uses the censored units' true failure "
                "cycles. It is the unobtainable target, not a method. Every "
                "row predicts the same engines from a law fitted to a fleet "
                "containing them, so the differences between rows are "
                "attributable to the treatment alone."))

        paper.section(
            "Which sensors carry signal, and where",
            body=(
                "Screening the twenty-one channels by how strongly each "
                "moves with age gives two different answers depending on "
                "whether the operating regime is held fixed, and the "
                "difference is a result rather than a detail. On the "
                "single-regime subsets the two screenings agree exactly. On "
                "the six-regime subsets, pooled screening finds one usable "
                "channel out of twenty-one — no correlation in FD002 exceeds "
                "0.28 — and the fleet looks as though it has almost no "
                "degradation signal in it. Holding the regime fixed recovers "
                "fourteen.\n\n"
                "Moving the engine between six flight conditions swings "
                "every sensor by far more than degradation does, so pooling "
                "buries a real monotone drift under a much larger "
                "between-regime variance. Ps30 reads +0.18 pooled in FD002 "
                "and +0.79 once the regime is fixed."),
            note="mean within-unit rank correlation between channel and age",
            table=_screening(screening),
            after=(
                "A second finding came from a channel that refused to stay "
                "excluded. The engine pressure ratio is exactly flat in "
                "FD001 and FD002 and trends at +0.31 and +0.21 in FD003 and "
                "FD004 — the two subsets that degrade the fan as well as the "
                "compressor. So channel usefulness depends on which fault is "
                "growing, not only on how the engine is flown, and a sensor "
                "set chosen on FD001 would discard the channel that marks "
                "the second failure mode."),
            figure_path=figures["sensor_masking"])

        paper.section(
            "Estimator B: the degradation signal",
            body=(
                "The second estimator is never told how long an engine "
                "lives — no lifetime law, no survival curve, not even the "
                "age. The channels are reduced to one health index by least "
                "squares against a target that sits at 1 while the engine is "
                "healthy and falls to 0 at failure, the index is smoothed "
                "causally, and it is converted to cycles through the "
                "empirical conditional distribution of true remaining life "
                "given health. The interval is that measured spread, not a "
                "variance formula.\n\n"
                "The target needs a cap, because degradation has an onset: "
                "an engine does not begin dying at cycle 1, and a target "
                "declining from the first cycle asks the regression to find "
                "a trend that is not there. The cap is estimated rather than "
                "taken from the literature's usual 125, and two criteria "
                "that could disagree do not — the cap at which the index "
                "lands on 1 healthy and 0 dead without being asked to, and "
                f"the cap maximising fit quality, agree on {model.cap} "
                f"cycles here.\n\n"
                "Read physically, that cap is the horizon over which these "
                "instruments say anything at all. Beyond roughly a hundred "
                "cycles from failure they do not distinguish an engine with "
                "150 cycles left from one with 250, which sets a ceiling on "
                "what any sensor-driven estimator can achieve."),
            note=f"{subset}, cap {model.cap} cycles, "
                 f"{len(model.index.channels)} channels, "
                 f"index R2 {model.index.r_squared:.3f}",
            figure_path=[figures["health_index"], figures["cap_selection"]])

        paper.section(
            "The route that does not work",
            body=(
                "The obvious thing to do with a declining health index is to "
                "fit its slope and project it to the failure threshold. The "
                "fitted slope comes out too shallow by a factor of 3.6 at "
                "the median and by more than five on a quarter of the fleet, "
                "and not because of noise: the index carries a per-cycle "
                "noise of about 0.042 against an eleven-cycle mean, which is "
                "small. To reach the threshold on time the index would have "
                "to fall at about -0.009 per cycle, which is roughly what "
                "the run-to-failure engines average over their final hundred "
                "cycles — but the slope fitted on a test unit's trailing "
                "window has a median of -0.003, with half the fleet between "
                "-0.007 and -0.001.\n\n"
                "The cause is attenuation. The index is a regression "
                "estimate of health, so it is shrunk toward the fleet mean, "
                "and a shrunken signal has a shallower slope than the "
                "quantity it estimates. Extrapolating a shrunken slope "
                "extrapolates the shrinkage. Degradation also accelerates, "
                "which compounds it: an engine censored while still at index "
                "0.8 is in the slow early part of its decline, and the fall "
                "still to come is steeper than anything in its own history. "
                "Calibrating the inversion fixes both at once, because the "
                "calibration is fitted on the same shrunken index it is "
                "applied to."),
            note="FD001 test fleet, slope fitted on a 40-cycle trailing "
                 "window",
            figure_path=figures["extrapolation_failure"],
            after=(
                "A second bias was found and fixed on the way. The "
                "calibration was being fitted on a cycle-weighted population "
                "and applied to a unit-weighted one: an engine that lives 350 "
                "cycles contributes more than twice as many rows as one that "
                "lives 150, almost all from its long healthy stretch. At "
                "health 0.8 to 0.9 the median remaining life among training "
                "cycles is 130 against 95 among test units at the same "
                "health. Weighting each engine equally removes most of it; "
                "the remainder is cap saturation and is reported rather than "
                "corrected."))

        paper.section(
            "The two estimators against each other",
            body=(
                "Neither estimator dominates, which is the point of having "
                f"two. Close to failure the degradation signal is "
                f"{near_a['mae'] / near_b['mae']:.1f} times more accurate — "
                f"{near_b['mae']:.0f} cycles against {near_a['mae']:.0f} — "
                "and that is the band with consequences attached. Its "
                "advantage is still about two to one out to fifty cycles, "
                "and then it collapses.\n\n"
                "Beyond fifty cycles the two sit within a factor of two of "
                "each other and which one wins is not predictable from the "
                "horizon alone: the age law takes one or two bands on every "
                "one of the four fleets, but never the same ones. That is "
                "what the detectability horizon looks like from the outside. "
                "Past it the sensors have nothing left to add, the "
                "degradation estimator falls back on the spread of its top "
                "calibration bin, and the fleet's lifetime distribution is "
                "as good a guide as anything.\n\n"
                "They disagree by "
                f"{found['disagreement']['mean_absolute_gap']:.0f} cycles on "
                f"average and their predictions correlate at only "
                f"{found['disagreement']['correlation']:+.2f}, so they are "
                "not two tunings of one model. Where they disagree, which is "
                "right is predictable from the horizon rather than from "
                "which looks better."),
            note=f"{subset}, mean absolute error in cycles",
            table=_head_to_head(found["comparison"]),
            figure_path=figures["disagreement"])

        paper.section(
            "Coverage, as measured",
            body=(
                "A prediction interval is a claim with a number attached, "
                "and whether it holds is an empirical question. Pooled over "
                "the test fleet both estimators look broadly calibrated: the "
                "age law's nominal 90% interval covers "
                f"{pooled_a['coverage_90']:.0%} here and the degradation "
                f"signal's covers {pooled_b['coverage_90']:.0%}.\n\n"
                "Conditioning on the horizon takes that picture apart, and "
                "it does so in two different ways on different fleets. On "
                f"{subset}'s engines within 25 cycles of failure the age "
                f"law's nominal 50% interval covers "
                f"{near_a['coverage_50']:.0%} and its nominal 80% covers "
                f"{near_a['coverage_80']:.0%}, while its nominal 90% covers "
                f"{near_a['coverage_90']:.0%} — the prediction is so far off "
                "that only the widest interval catches the truth at all, and "
                "it catches everything. On "
                f"{worst['subset']} the same nominal 90% interval covers "
                f"{worst['near_cover_a']:.0%}.\n\n"
                "Neither is calibration. The intervals are the wrong shape "
                "rather than uniformly too wide or too narrow, and pooling "
                "averages an interval that misses almost everything near "
                "failure against one that contains almost everything "
                "elsewhere, which comes out looking respectable."),
            note=(f"{subset}, degradation signal; nominal levels fixed "
                  "before any coverage was computed"),
            table=_bands(found["degradation_bands"]),
            warn=(
                "These are measured coverages, not intended ones. Where a "
                "nominal 90% interval covers 53% of a band, 53% is what "
                "appears. The degradation estimator's intervals hold up "
                "better because they are built from the conditional spread "
                "actually observed in the training fleet rather than from an "
                "assumed error law — which is a reason to prefer that "
                "construction, not evidence that the model is right."),
            figure_path=figures["coverage"])

        paper.section(
            "Does it hold across the four fleets",
            body=(
                "The subsets differ in operating conditions and fault "
                "modes, and the comparison holds on all four. The "
                "degradation signal is between four and nine times more "
                "accurate than the age law on the engines closest to "
                "failure, and wins pooled on every subset. The age law's "
                "intervals come apart near failure on all four, though not "
                "at the same nominal level: at 90% the shortfall shows on "
                "three of them and worsens with the fleet's difficulty, "
                f"down to {worst['near_cover_a']:.0%} on {worst['subset']} — "
                "six operating regimes and two fault modes, where the "
                "health index is also weakest and the fitted cap longest. "
                f"On {subset} the nominal 90% interval instead covers "
                f"{near_a['coverage_90']:.0%} near failure and the failure "
                "appears at the narrower levels, where nominal 50% covers "
                f"{near_a['coverage_50']:.0%} against "
                f"{pooled_a['coverage_50']:.0%} pooled."),
            note="mean absolute error in cycles; coverage at nominal 90%",
            table=_subsets(subset_rows),
            figure_path=figures["across_subsets"])

        paper.section(
            "Recovering a planted truth, and where it stops",
            body=(
                "Simulated fleets with a known health law, a known "
                "detectability horizon and known useless channels check that "
                "the estimator finds what is actually there. With a clean "
                "signal it recovers the planted horizon exactly and predicts "
                "remaining life inside it to within three cycles. As sensor "
                "noise rises the error grows, the recovered horizon drifts "
                "by a grid step, and the intervals widen to keep their "
                "coverage — which is the behaviour wanted, since an "
                "estimator losing its signal should become less certain "
                "rather than merely less accurate."),
            note="90 run-to-failure and 90 censored simulated engines per row",
            table=_recovery(recovery_rows),
            after=(
                "The last row is the useful one: at noise 3.0 the index "
                "explains little and the error is 27 cycles, so the method "
                "has stopped working — and says so through its intervals "
                "rather than through a confident wrong number."))

        paper.section(
            "What this does and does not establish",
            body=(
                "Established. Remaining-life error depends so strongly on "
                "the horizon that a pooled figure misdescribes both ends of "
                "a fleet, and pooled coverage can look calibrated while the "
                "intervals are badly overconfident exactly where decisions "
                "are made. A sensor-driven estimator is several times more "
                "accurate than an age-only one close to failure, and within "
                "a factor of two of it beyond fifty cycles, where which one "
                "wins is not predictable from the horizon. Which sensors "
                "carry signal depends on "
                "the operating regime and on the fault mode, so a channel "
                "selection is not transferable between fleets of the same "
                "engine.\n\n"
                "Not established. These are simulated engines. C-MAPSS "
                "models degradation with a smooth growing fault and "
                "Gaussian sensor noise, and real fleets have maintenance "
                "actions, replaced parts, sensor drift and faults that "
                "arrive without warning, none of which is here. The "
                "detectability horizon of about a hundred cycles is a "
                "property of this simulator's instrumentation and should "
                "not be carried over to real hardware.\n\n"
                "The censoring conclusion also deserves care. Dropping the "
                "censored units performs well here because of how the "
                "archive was built, and badly under the censoring real "
                "fleets have. The transferable result is the method for "
                "telling the difference — compare treatments against a "
                "planted truth under the censoring mechanism you actually "
                "face — not the ranking that method produced on this "
                "dataset."),
            note="limitations")

    return out_path
