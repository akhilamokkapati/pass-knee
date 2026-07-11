"""
run_capture.py
PASS knee module — capture-to-graph + accuracy report, for any source.

Pulls a session through the SAME source interface (source.get_data -> Capture),
runs the biomechanics engine over it, low-pass filters the knee angle offline
(zero-phase), and produces:

  * a plot that OVERLAYS raw and filtered knee angle (so the filter's effect is
    visible and any filtfilt edge artifacts on sharp transitions are catchable),
    with ground truth drawn too when the source has it, and activity bands shaded
    when the source provides labels;
  * printed summary stats — a citable number, not just a picture.

Two sources are wired here; the analysis/plot functions are source-agnostic and
take the flexion axis, neutral and sample rate as arguments:

  synthetic : independent ground-truth angle -> the error panel is vs ground
              truth, with the +/-2.5 deg band.
  hugadb    : real human IMU, Madgwick-fused, NO ground truth (knee_angle_deg is
              NaN) -> the lower panel shows raw-minus-filtered (what the filter
              removed), and we report per-activity knee flexion (a real ROM
              number) instead of an accuracy error.

HONEST SCOPE
------------
On synthetic data the error is engine-vs-forward-model (numerical). On HuGaDB
there is no reference angle, so the plot shows filtered vs raw and physiological
ROM, not an accuracy figure. The +/-2.5 deg clinical target is only truly tested
against a gold-standard reference (goniometer / motion capture).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from biomechanics.relative_orientation import knee_relative, remove_offset
from biomechanics.joint_angles import knee_flexion_angle, DEFAULT_FLEXION_AXIS
from filters import lowpass_offline, DEFAULT_CUTOFF_HZ
from calibrate import calibrate_from_quaternions
from metrics import summarize
from repetitions import detect_reps
from sources.synthetic import SyntheticSource
from sources.hugadb import HuGaDBSource, HUGADB_FLEXION_AXIS

CLINICAL_TARGET_DEG = 2.5
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class AccuracyStats:
    max_abs_error_deg: float
    rms_error_deg: float
    mean_error_deg: float
    n_samples: int


# --- engine over a capture ------------------------------------------------

def recovered_angle(cap, axis=DEFAULT_FLEXION_AXIS, q_neutral=IDENTITY_QUAT) -> np.ndarray:
    """Engine-recovered knee flexion angle over a whole Capture (batched).

    axis and q_neutral default to the synthetic convention (no calibration,
    +x flexion); HuGaDB passes its own -Y axis and a standing-window neutral."""
    rel = remove_offset(knee_relative(cap.quat_thigh, cap.quat_shank), q_neutral)
    return knee_flexion_angle(rel, axis=axis)


def accuracy_stats(ground_truth: np.ndarray, recovered: np.ndarray) -> AccuracyStats:
    """Max-abs and RMS error of recovered vs ground truth, both in degrees."""
    err = np.asarray(recovered, float) - np.asarray(ground_truth, float)
    return AccuracyStats(
        max_abs_error_deg=float(np.max(np.abs(err))),
        rms_error_deg=float(np.sqrt(np.mean(err ** 2))),
        mean_error_deg=float(np.mean(err)),
        n_samples=int(err.size),
    )


def format_stats(stats: AccuracyStats) -> str:
    within = "within" if stats.max_abs_error_deg <= CLINICAL_TARGET_DEG else "OUTSIDE"
    return (
        f"n={stats.n_samples}  max error {stats.max_abs_error_deg:.4g} deg  "
        f"RMS error {stats.rms_error_deg:.4g} deg  bias {stats.mean_error_deg:+.4g} deg  "
        f"[{within} +/-{CLINICAL_TARGET_DEG} deg target]"
    )


def straight_leg_neutral(cap, fs_hz: float) -> np.ndarray:
    """Neutral quaternion for the source: a 'standing' window if labelled, else
    the first ~1 s (assumed quiet). Identity is the fallback for unlabelled,
    already-calibrated sources (synthetic)."""
    if cap.activity is not None and np.any(cap.activity == "standing"):
        idx = np.where(cap.activity == "standing")[0]
    elif cap.activity is not None:
        idx = np.arange(min(cap.seq.size, int(round(fs_hz))))
    else:
        return IDENTITY_QUAT
    return calibrate_from_quaternions(cap.quat_thigh[idx], cap.quat_shank[idx]).q_neutral


def activity_flexion_lines(activity: np.ndarray, angle: np.ndarray) -> list[str]:
    """Per-activity median and peak flexion — the citable ROM numbers."""
    lines = []
    for lab in dict.fromkeys(activity):          # unique, order-preserving
        m = activity == lab
        lines.append(f"{lab:12s} median {np.median(angle[m]):5.1f}  peak {np.max(angle[m]):5.1f} deg")
    return lines


def _reps_line(r) -> str:
    """Rep count WITH its confidence indicators — never a bare number."""
    s = f"reps: {r.count}"
    edges = [e for e, flag in (("start", r.partial_at_start), ("end", r.partial_at_end)) if flag]
    if edges:
        which = "both edges" if len(edges) == 2 else f"{edges[0]} edge"
        s += f" ({which} partial, recording trimmed)"
    s += f";  adaptive prominence {r.effective_prominence_deg:.1f} deg"
    if not np.isnan(r.amplitude_cv):
        s += f";  amplitude CV {r.amplitude_cv:.2f}"
    if not np.isnan(r.period_cv):
        s += f", period CV {r.period_cv:.2f}"
    return s


def metric_lines(angle: np.ndarray, fs_hz: float) -> list[str]:
    """The full metric set for the figure: direct reductions + reps + confidence.
    Computed on the FILTERED angle (clean signal)."""
    m = summarize(angle, fs_hz)
    return [
        (f"ROM {m.range_of_motion_deg:.1f} deg    max flexion {m.max_flexion_deg:.1f} deg    "
         f"max extension {m.max_extension_deg:.1f} deg    peak vel {m.peak_angular_velocity_dps:.0f} deg/s"),
        _reps_line(detect_reps(angle, fs_hz)),
    ]


# --- plotting -------------------------------------------------------------

def _draw_activity_bands(ax, t, activity):
    """Shade contiguous activity segments; return legend handles."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = list(dict.fromkeys(activity))
    cmap = plt.get_cmap("Pastel2")
    colors = {lab: cmap(i % 8) for i, lab in enumerate(labels)}
    changes = np.where(activity[1:] != activity[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(activity)]))
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=colors[activity[s]], alpha=0.5, lw=0)
    return [Patch(color=colors[l], alpha=0.5, label=l) for l in labels]


def plot_capture(cap, raw, filtered, *, ground_truth=None, cutoff_hz,
                 title, caption, stats_lines, save_path=None, show=False):
    """Two panels: raw vs filtered knee angle (overlaid) on top; below, error vs
    ground truth (with +/-2.5 band) if available, else raw-minus-filtered."""
    import matplotlib.pyplot as plt

    t = np.asarray(cap.t_ms, float) / 1000.0
    fig, (ax_ang, ax_low) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    band_handles = []
    if cap.activity is not None:
        band_handles = _draw_activity_bands(ax_ang, t, cap.activity)

    ax_ang.plot(t, raw, color="0.6", lw=0.8, label="raw knee angle")
    ax_ang.plot(t, filtered, color="C0", lw=1.8, label=f"filtered ({cutoff_hz:g} Hz low-pass)")
    if ground_truth is not None:
        ax_ang.plot(t, ground_truth, "--", color="C3", lw=1.3, label="ground truth")
    ax_ang.set_ylabel("Knee flexion (deg)")
    ax_ang.set_title(title)
    ax_ang.grid(alpha=0.3)
    line_leg = ax_ang.legend(loc="upper right", fontsize=8)
    if band_handles:
        ax_ang.add_artist(line_leg)
        ax_ang.legend(handles=band_handles, loc="upper left", fontsize=8, title="activity")

    if ground_truth is not None:
        err = filtered - ground_truth
        ax_low.axhspan(-CLINICAL_TARGET_DEG, CLINICAL_TARGET_DEG, color="green",
                       alpha=0.12, label=f"+/-{CLINICAL_TARGET_DEG} deg target")
        ax_low.plot(t, err, color="firebrick", lw=1)
        ax_low.set_ylabel("Filtered\nerror (deg)")
        ax_low.legend(loc="upper right", fontsize=8)
    else:
        ax_low.plot(t, raw - filtered, color="darkorange", lw=0.8)
        ax_low.axhline(0, color="0.5", lw=0.6)
        ax_low.set_ylabel("Raw - filtered\n(deg)")
    ax_low.set_xlabel("Time (s)")
    ax_low.grid(alpha=0.3)

    fig.text(0.01, 0.005, "\n".join(stats_lines), fontsize=8, family="monospace",
             va="bottom")
    fig.text(0.01, 0.97, caption, fontsize=8, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.02 + 0.02 * len(stats_lines), 1, 0.955))

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


# --- driver ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PASS knee capture-to-graph + accuracy report")
    ap.add_argument("--source", choices=["synthetic", "hugadb"], default="synthetic")
    ap.add_argument("--file", type=str, help="HuGaDB CSV path (for --source hugadb)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to capture")
    ap.add_argument("--noise", type=float, default=0.0, help="synthetic sensor noise (deg)")
    ap.add_argument("--seed", type=int, default=0, help="synthetic RNG seed")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF_HZ, help="low-pass cutoff (Hz)")
    ap.add_argument("--save", type=str, default=None, help="output PNG path")
    ap.add_argument("--no-show", action="store_true", help="save only, no window")
    args = ap.parse_args()

    if args.source == "hugadb":
        if not args.file:
            ap.error("--source hugadb requires --file")
        src = HuGaDBSource(args.file)
        cap = src.get_data(args.duration)
        fs, axis = src.fs_hz, HUGADB_FLEXION_AXIS
        neutral = straight_leg_neutral(cap, fs)
        title = f"PASS knee — HuGaDB real IMU (Madgwick-fused)  [{src.filepath.name}]"
        caption = ("real human IMU; no ground-truth angle (knee_angle_deg=NaN) — "
                   "raw vs offline-filtered shown; error is not sensor-vs-limb accuracy")
        gt = None
    else:
        src = SyntheticSource(noise_deg=args.noise, seed=args.seed)
        cap = src.get_data(args.duration or 5.0)
        fs, axis, neutral = src.rate_hz, DEFAULT_FLEXION_AXIS, IDENTITY_QUAT
        title = "PASS knee — synthetic (engine recovery vs ground truth)"
        caption = ("validation = engine vs synthetic forward model (numerical), "
                   "NOT sensor-vs-limb accuracy")
        gt = cap.knee_angle_deg

    raw = recovered_angle(cap, axis=axis, q_neutral=neutral)
    filtered = lowpass_offline(raw, cutoff_hz=args.cutoff, fs_hz=fs)

    # summary stats — metrics lead (the feasibility numbers), then validation,
    # then filter effect, then per-activity context.
    interior = slice(int(0.1 * raw.size), int(0.9 * raw.size) or None)
    stats_lines = metric_lines(filtered, fs)
    if gt is not None:
        stats_lines.append("filtered vs ground truth:  " + format_stats(accuracy_stats(gt, filtered)))
    stats_lines.append(
        f"filter: {args.cutoff:g} Hz low-pass, fs={fs:g} Hz;  "
        f"noise removed RMS(raw-filtered)={np.sqrt(np.mean((raw[interior]-filtered[interior])**2)):.3g} deg")
    if cap.activity is not None:
        stats_lines += activity_flexion_lines(cap.activity, filtered)

    for line in stats_lines:
        print(line)

    plot_capture(cap, raw, filtered, ground_truth=gt, cutoff_hz=args.cutoff,
                 title=title, caption=caption, stats_lines=stats_lines,
                 save_path=args.save, show=not args.no_show)
    if args.save:
        print(f"saved plot -> {args.save}")


if __name__ == "__main__":
    main()
