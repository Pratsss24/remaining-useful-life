"""Figures. Committed to the repository so the write-up renders anywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

import data                            # noqa: E402
import degradation as degradation_mod  # noqa: E402
import sensors                         # noqa: E402

PALETTE = {"ink": "#16181d", "muted": "#5b6472", "grid": "#e3e6ea",
           "good": "#17803d", "bad": "#b4232c", "accent": "#2563a8",
           "warm": "#d9a441", "plum": "#6b4c9a"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.facecolor": "white",
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _bands(rows):
    return [row for row in rows if row["band"] != "all"]


# -- the headline: error against how far ahead you ask --------------------

def error_by_band(found: dict, out_dir: Path) -> Path:
    """Mean absolute error against true remaining life, both estimators.

    The figure the project exists to produce. The pooled averages are drawn
    as flat dashed lines so that what they conceal is visible rather than
    argued.
    """
    first = _bands(found["age_law_bands"])
    second = _bands(found["degradation_bands"])
    labels = [row["band"] for row in first]
    spots = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    axes[0].bar(spots - width / 2, [row["mae"] for row in first], width,
                color=PALETTE["accent"], label="A, age only")
    axes[0].bar(spots + width / 2, [row["mae"] for row in second], width,
                color=PALETTE["bad"], label="B, degradation signal")
    pooled_a = next(r for r in found["age_law_bands"]
                    if r["band"] == "all")["mae"]
    pooled_b = next(r for r in found["degradation_bands"]
                    if r["band"] == "all")["mae"]
    axes[0].axhline(pooled_a, color=PALETTE["accent"], linestyle="--",
                    linewidth=1.1)
    axes[0].axhline(pooled_b, color=PALETTE["bad"], linestyle="--",
                    linewidth=1.1)
    axes[0].annotate(f"A pooled {pooled_a:.0f}", (len(labels) - 0.4,
                                                  pooled_a),
                     fontsize=7.6, color=PALETTE["accent"], ha="right",
                     va="bottom")
    axes[0].annotate(f"B pooled {pooled_b:.0f}", (len(labels) - 0.4,
                                                  pooled_b),
                     fontsize=7.6, color=PALETTE["bad"], ha="right",
                     va="top")
    axes[0].set_xticks(spots)
    axes[0].set_xticklabels(labels, fontsize=7.8)
    axes[0].set_xlabel("true remaining life at prediction, cycles")
    axes[0].set_ylabel("mean absolute error, cycles")
    axes[0].set_title("error depends on how far ahead you ask")
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])

    axes[1].axhline(0.0, color=PALETTE["muted"], linewidth=0.9)
    axes[1].bar(spots - width / 2, [row["bias"] for row in first], width,
                color=PALETTE["accent"], label="A, age only")
    axes[1].bar(spots + width / 2, [row["bias"] for row in second], width,
                color=PALETTE["bad"], label="B, degradation signal")
    axes[1].set_xticks(spots)
    axes[1].set_xticklabels(labels, fontsize=7.8)
    axes[1].set_xlabel("true remaining life at prediction, cycles")
    axes[1].set_ylabel("bias, cycles (positive = optimistic)")
    axes[1].set_title("and the bias changes sign across the fleet")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.suptitle(f"{found['subset']}  remaining-life error by horizon",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, "error_by_band")


# -- coverage, measured --------------------------------------------------

def coverage(found: dict, out_dir: Path) -> Path:
    """Measured against nominal coverage, pooled and near failure.

    The left panel is the pooled calibration, where both estimators look
    broadly reasonable. The right panel is the same thing for the engines
    within 25 cycles of failure, where the age law collapses.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    for ax in axes:
        ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":",
                linewidth=1.0, label="perfect calibration")
        ax.set_xlim(0.15, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("nominal coverage")
        _grid(ax)

    for key, colour, label in (("age_law_calibration", PALETTE["accent"],
                                "A, age only"),
                               ("degradation_calibration", PALETTE["bad"],
                                "B, degradation signal")):
        rows = found[key]
        axes[0].plot([row["nominal"] for row in rows],
                     [row["measured"] for row in rows], "o-",
                     color=colour, linewidth=1.5, markersize=4, label=label)

    axes[0].set_ylabel("measured coverage")
    axes[0].set_title("pooled over the whole test fleet")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    nominal = (0.5, 0.8, 0.9, 0.95)
    for key, colour, label in (("age_law", PALETTE["accent"], "A, age only"),
                               ("degradation", PALETTE["bad"],
                                "B, degradation signal")):
        forecast = found[key]
        near = forecast.truth < 25
        measured = [float(forecast.covered(level)[near].mean())
                    for level in nominal]
        axes[1].plot(nominal, measured, "o-", color=colour, linewidth=1.5,
                     markersize=4, label=label)
        for level, value in zip(nominal, measured):
            if colour == PALETTE["accent"]:
                axes[1].annotate(f"{value:.0%}", (level, value), fontsize=7.4,
                                 color=colour, ha="left", va="top",
                                 xytext=(3, -2), textcoords="offset points")

    axes[1].set_title("engines within 25 cycles of failure")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(f"{found['subset']}  prediction intervals: what they claim "
                 f"and what they cover", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, "coverage")


# -- the health index ----------------------------------------------------

def health_index(model, subset: str, out_dir: Path, *,
                 shown: int = 12) -> Path:
    """Index trajectories against remaining life, and the cap that bounds
    what the sensors can see."""
    units = data.load(subset, "train")[:shown]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    for unit in units:
        smoothed = model.index.smoothed(unit, model.window)
        axes[0].plot(unit.rul(), smoothed, color=PALETTE["accent"],
                     alpha=0.45, linewidth=1.0)
    axes[0].axvline(model.cap, color=PALETTE["bad"], linestyle="--",
                    linewidth=1.3,
                    label=f"fitted cap, {model.cap} cycles")
    axes[0].axhline(0.0, color=PALETTE["muted"], linewidth=0.8)
    axes[0].axhline(1.0, color=PALETTE["muted"], linewidth=0.8,
                    linestyle=":")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("true remaining life, cycles (failure at the right)")
    axes[0].set_ylabel("health index")
    axes[0].set_title(f"{shown} run-to-failure engines")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    _grid(axes[0])

    centres = model.calibration.centres
    axes[1].fill_between(centres,
                         model.calibration.quantile(centres, 0.05),
                         model.calibration.quantile(centres, 0.95),
                         color=PALETTE["bad"], alpha=0.16, linewidth=0,
                         label="90% interval, as measured")
    axes[1].plot(centres, model.calibration.quantile(centres, 0.5),
                 color=PALETTE["bad"], linewidth=1.7,
                 label="median remaining life")
    axes[1].set_xlabel("health index")
    axes[1].set_ylabel("remaining life, cycles")
    axes[1].set_title("the calibration, from the training fleet")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.suptitle(f"{subset}  the health index and what it converts to",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, "health_index")


def cap_selection(chosen: dict, out_dir: Path) -> Path:
    """Two criteria for the detectability horizon, agreeing."""
    rows = chosen["cap_table"]
    caps = [row["cap"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(caps, [row["gap"] for row in rows], "o-",
            color=PALETTE["accent"], linewidth=1.6, markersize=4.5,
            label="self-consistency gap  |start-1| + |failure|")
    ax.set_xlabel("cap, cycles")
    ax.set_ylabel("consistency gap (lower is better)")
    _grid(ax)

    twin = ax.twinx()
    twin.plot(caps, [row["r_squared"] for row in rows], "s--",
              color=PALETTE["bad"], linewidth=1.4, markersize=4,
              label="index fit, R-squared")
    twin.set_ylabel("R-squared (higher is better)",
                    color=PALETTE["bad"])
    twin.tick_params(axis="y", colors=PALETTE["bad"])
    twin.spines["top"].set_visible(False)

    ax.axvline(chosen["cap_by_consistency"], color=PALETTE["good"],
               linestyle=":", linewidth=1.4)
    ax.annotate(f"chosen: {chosen['cap_by_consistency']}",
                (chosen["cap_by_consistency"], ax.get_ylim()[1]),
                fontsize=8, color=PALETTE["good"], ha="center", va="top")

    handles = ax.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    # Below the axes: one curve is U-shaped and the other its mirror, so
    # between them they reach into every corner.
    fig.legend(handles[0] + extra[0], handles[1] + extra[1],
               frameon=False, fontsize=8, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    ax.set_title("the detectability horizon, by two criteria that could "
                 "disagree")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out_dir, "cap_selection")


# -- sensors -------------------------------------------------------------

def sensor_masking(rows: list[dict], out_dir: Path) -> Path:
    """Pooled against within-regime screening, per channel.

    Drawn for a multi-regime subset beside a single-regime one, because the
    finding is the difference between them.
    """
    pick = [row for row in rows if row["subset"] in ("FD001", "FD002")]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)

    for ax, row in zip(axes, pick):
        screening = row["screening"]
        spots = np.arange(data.N_SENSORS)
        ax.barh(spots - 0.2, np.abs(screening.pooled_trend), 0.4,
                color=PALETTE["muted"], label="pooled")
        ax.barh(spots + 0.2, np.abs(screening.regime_trend), 0.4,
                color=PALETTE["accent"], label="within operating regime")
        ax.axvline(sensors.TREND_FLOOR, color=PALETTE["bad"],
                   linestyle="--", linewidth=1.2,
                   label=f"screening floor {sensors.TREND_FLOOR}")
        ax.set_yticks(spots)
        ax.set_yticklabels([name.split()[0] for name in data.SENSOR_NAMES],
                           fontsize=7.2)
        ax.set_xlabel("|mean within-unit rank correlation with age|")
        ax.set_title(f"{row['subset']}  {row['regimes']} regime(s):  "
                     f"{row['pooled_informative']} pooled, "
                     f"{row['regime_informative']} within regime")
        ax.invert_yaxis()
        _grid(ax)

    # Below the axes: the bars run the full width of both panels, so any
    # in-axes corner sits on top of data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("screening the channels: holding the regime fixed recovers "
                 "a signal pooling destroys", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save(fig, out_dir, "sensor_masking")


# -- censoring -----------------------------------------------------------

def censoring_recovery(rows: list[dict], out_dir: Path) -> Path:
    """Three censoring schemes, four treatments, against a planted truth."""
    schemes = sorted({row["scheme"] for row in rows},
                     key=lambda name: ["independent", "proportional",
                                       "end_of_test"].index(name))
    treatments = ["censored MLE", "drop censored", "as failures", "oracle"]
    colours = [PALETTE["accent"], PALETTE["warm"], PALETTE["bad"],
               PALETTE["good"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharex=True)
    spots = np.arange(len(schemes))
    width = 0.2

    for which, (ax, key, label) in enumerate((
            (axes[0], "beta_error", "shape error"),
            (axes[1], "eta_error", "scale error"))):
        for index, (treatment, colour) in enumerate(zip(treatments, colours)):
            values = [next(row[key] for row in rows
                           if row["scheme"] == scheme
                           and row["treatment"] == treatment)
                      for scheme in schemes]
            ax.bar(spots + (index - 1.5) * width, values, width,
                   color=colour, label=treatment if which == 0 else None)
        ax.axhline(0.0, color=PALETTE["muted"], linewidth=0.9)
        ax.set_xticks(spots)
        ax.set_xticklabels([name.replace("_", " ") for name in schemes],
                           fontsize=8.4)
        ax.set_ylabel(f"{label}, fraction of the truth")
        ax.set_title(label)
        _grid(ax)

    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("recovering a known lifetime law: the right treatment "
                 "depends on how the censoring happened",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, "censoring_recovery")


def survival_fit(curve, law, out_dir: Path) -> Path:
    """The non-parametric survivor function with the fitted law over it."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))

    times = np.concatenate([[0.0], curve.times])
    values = np.concatenate([[1.0], curve.values])
    axes[0].step(times, values, where="post", color=PALETTE["ink"],
                 linewidth=1.6, label="Kaplan-Meier (95% band)")
    lower = np.concatenate([[1.0], curve.lower])
    upper = np.concatenate([[1.0], curve.upper])
    axes[0].fill_between(times, lower, upper, step="post",
                         color=PALETTE["ink"], alpha=0.14, linewidth=0)
    grid = np.linspace(1.0, float(curve.times.max()) * 1.05, 400)
    axes[0].plot(grid, law.survival(grid), color=PALETTE["bad"],
                 linestyle="--", linewidth=1.5,
                 label=f"Weibull  beta {law.beta:.2f}, eta {law.eta:.0f}")
    axes[0].set_xlabel("cycles")
    axes[0].set_ylabel("surviving fraction")
    axes[0].set_title("the fleet lifetime law")
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])

    ages = np.linspace(1.0, law.quantile(0.97), 300)
    axes[1].fill_between(ages, law.rul_quantile(ages, 0.05),
                         law.rul_quantile(ages, 0.95),
                         color=PALETTE["accent"], alpha=0.16, linewidth=0,
                         label="90% interval")
    axes[1].plot(ages, law.rul_quantile(ages, 0.5),
                 color=PALETTE["accent"], linewidth=1.7,
                 label="median remaining life")
    axes[1].plot(ages, law.mean_residual_life(ages),
                 color=PALETTE["plum"], linewidth=1.2, linestyle="--",
                 label="mean remaining life")
    axes[1].set_xlabel("age, cycles")
    axes[1].set_ylabel("remaining life, cycles")
    axes[1].set_title("estimator A: what the law says, given age alone")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "survival_fit")


# -- disagreement and the dead end ---------------------------------------

def disagreement(found: dict, out_dir: Path) -> Path:
    """The two estimators against each other and against the truth."""
    first, second = found["age_law"], found["degradation"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    limit = float(max(first.truth.max(), first.median.max(),
                      second.median.max())) * 1.05
    for ax, forecast, colour, label in (
            (axes[0], first, PALETTE["accent"], "A, age only"),
            (axes[0], second, PALETTE["bad"], "B, degradation signal")):
        ax.scatter(forecast.truth, forecast.median, s=14, alpha=0.6,
                   color=colour, label=label, linewidths=0)
    axes[0].plot([0, limit], [0, limit], color=PALETTE["muted"],
                 linestyle=":", linewidth=1.0, label="perfect")
    axes[0].set_xlabel("true remaining life, cycles")
    axes[0].set_ylabel("predicted remaining life, cycles")
    axes[0].set_title("predicted against true")
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])

    gap = second.median - first.median
    closer_b = np.abs(second.error) < np.abs(first.error)
    axes[1].scatter(first.truth[closer_b], gap[closer_b], s=16, alpha=0.7,
                    color=PALETTE["bad"], linewidths=0,
                    label="B closer to the truth")
    axes[1].scatter(first.truth[~closer_b], gap[~closer_b], s=16, alpha=0.7,
                    color=PALETTE["accent"], linewidths=0,
                    label="A closer to the truth")
    axes[1].axhline(0.0, color=PALETTE["muted"], linewidth=0.9)
    axes[1].set_xlabel("true remaining life, cycles")
    axes[1].set_ylabel("B minus A, cycles")
    axes[1].set_title(f"where they disagree, and who is right "
                      f"(r = {found['disagreement']['correlation']:+.2f})")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.suptitle(f"{found['subset']}  two independent estimators, "
                 f"mean gap {found['disagreement']['mean_absolute_gap']:.0f} "
                 f"cycles", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir, "disagreement")


def extrapolation_failure(model, subset: str, out_dir: Path) -> Path:
    """Why the index's slope is not extrapolated: measured attenuation."""
    units = data.load(subset, "test")
    slopes, needed, errors = [], [], []
    for unit in units:
        found = degradation_mod.crossing_extrapolation(model, unit)
        if found.get("remaining") is None:
            continue
        slopes.append(found["slope"])
        needed.append(found["needed_slope"])
        errors.append(found["remaining"] - found["truth"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

    axes[0].scatter(np.abs(slopes), np.abs(needed), s=18, alpha=0.65,
                    color=PALETTE["bad"], linewidths=0)
    edge = float(max(np.abs(slopes + needed))) * 1.05
    axes[0].plot([0, edge], [0, edge], color=PALETTE["muted"],
                 linestyle=":", linewidth=1.0,
                 label="where extrapolation would work")
    axes[0].set_xlabel("|fitted slope| of the index, per cycle")
    axes[0].set_ylabel("|slope needed| to reach the threshold on time")
    axes[0].set_title("the fitted slope is too shallow, every time")
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])

    axes[1].hist(errors, bins=26, color=PALETTE["bad"], alpha=0.8)
    axes[1].axvline(0.0, color=PALETTE["ink"], linewidth=1.1)
    axes[1].set_xlabel("projected minus true remaining life, cycles")
    axes[1].set_ylabel("engines")
    axes[1].set_title(f"so the projection overshoots "
                      f"(median {np.median(errors):+.0f} cycles)")
    _grid(axes[1])

    fig.suptitle("the route that does not work: extrapolating a shrunken "
                 "slope extrapolates the shrinkage",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, "extrapolation_failure")


def across_subsets(rows: list[dict], out_dir: Path) -> Path:
    """Both estimators on all four subsets, near failure and pooled."""
    labels = [row["subset"] for row in rows]
    spots = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    for ax, key, title in ((axes[0], "near", "within 25 cycles of failure"),
                           (axes[1], "pooled", "pooled over the fleet")):
        ax.bar(spots - width / 2, [row[f"{key}_a"] for row in rows], width,
               color=PALETTE["accent"], label="A, age only")
        ax.bar(spots + width / 2, [row[f"{key}_b"] for row in rows], width,
               color=PALETTE["bad"], label="B, degradation signal")
        ax.set_xticks(spots)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        _grid(ax)
    axes[0].set_ylabel("mean absolute error, cycles")
    axes[0].legend(frameon=False, fontsize=8)

    fig.suptitle("the same comparison on all four subsets",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, "across_subsets")
