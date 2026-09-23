"""Figures of merit and the standard plots.

Everything that turns a lattice into a number or a figure lives here, and both
the walkthrough notebook and the comparison tables call it. A value shown in a
plot and the same value in a summary table therefore come from one function.

Quantities are measured against a reference lattice -- the ideally tapered
machine -- rather than against zero, because coarse tapering distorts the
closed orbit and the target is the orbit the ideal machine would have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TwissView:
    """Read-only view over a twiss, from a live table or from a checkpoint.

    A live `TwissTable` already indexes both per-element columns and scalars
    by name; a checkpoint keeps them in two files. Wrapping the pair here lets
    the same metric code serve both.
    """

    def __init__(self, vectors, scalars=None):
        self._vectors = vectors
        self._scalars = scalars or {}

    def __getitem__(self, key):
        if self._scalars and key in self._scalars:
            return self._scalars[key]
        return self._vectors[key]

    def __contains__(self, key):
        if key in self._scalars:
            return True
        try:
            self._vectors[key]
            return True
        except Exception:
            return False

    @classmethod
    def from_checkpoint(cls, directory):
        """Build a view from a checkpoint's twiss.parquet / twiss.json pair."""
        from . import io_ckpt

        vectors, payload = io_ckpt.read_twiss(directory)
        return cls(vectors, payload["values"])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def column(twiss, name) -> np.ndarray:
    """A per-element column as a float array."""
    return np.asarray(twiss[name], dtype=float)


def scalar(twiss, *candidate_names) -> float:
    """First of the candidate names the twiss carries, as a float.

    Several quantities have been renamed across xsuite versions, so the
    candidates are tried in order and a missing one gives NaN rather than
    raising.
    """
    for name in candidate_names:
        # Attribute access first: a live twiss table treats an unknown key as
        # an expression to evaluate, which raises rather than reporting a
        # miss. A checkpoint view has no attributes and answers by key.
        value = getattr(twiss, name, None)
        if value is None:
            try:
                value = twiss[name]
            except Exception:
                value = None
        if value is None:
            continue
        array = np.asarray(value)
        if array.size:
            return float(array.reshape(-1)[0])
    return float("nan")


def indices_of(twiss, names) -> np.ndarray:
    """Row positions of named elements, for a live twiss or a checkpoint."""
    names = list(np.asarray(names, dtype=str))
    try:
        return twiss.rows.indices[names]
    except AttributeError:
        position = {name: i for i, name in enumerate(np.asarray(twiss["name"], dtype=str))}
        return np.array([position[name] for name in names], dtype=int)


def rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.nanmean(values**2)))


def max_abs(values) -> float:
    return float(np.nanmax(np.abs(np.asarray(values, dtype=float))))


def percent_change(value, reference) -> float:
    if not np.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (value / reference - 1.0)


def percentile_abs(values, percentile) -> float:
    return float(np.percentile(np.abs(np.asarray(values, dtype=float)), percentile))


# --------------------------------------------------------------------------
# Optics summary
# --------------------------------------------------------------------------


def optics_summary(twiss, reference) -> dict:
    """Every cheap figure of merit for one lattice, against a reference.

    Note: `eq_sigma_delta` and `eq_sigma_zeta` are not present in every xsuite
    version. Where they are missing the momentum-spread and bunch-length rows
    come out NaN rather than being derived, so the table always reports what
    the twiss actually provided.
    """
    betx_beating = (column(twiss, "betx") - column(reference, "betx")) / column(reference, "betx")
    bety_beating = (column(twiss, "bety") - column(reference, "bety")) / column(reference, "bety")
    delta_x = column(twiss, "x") - column(reference, "x")
    delta_y = column(twiss, "y") - column(reference, "y")
    delta_dx = column(twiss, "dx") - column(reference, "dx")
    delta_dy = column(twiss, "dy") - column(reference, "dy")
    delta = column(twiss, "delta")

    emit_x = scalar(twiss, "eq_gemitt_x", "rad_int_eq_gemitt_x")
    emit_y = scalar(twiss, "eq_gemitt_y", "rad_int_eq_gemitt_y")
    emit_x_ref = scalar(reference, "eq_gemitt_x", "rad_int_eq_gemitt_x")
    emit_y_ref = scalar(reference, "eq_gemitt_y", "rad_int_eq_gemitt_y")

    return {
        "Horizontal tune Qx": scalar(twiss, "qx"),
        "Vertical tune Qy": scalar(twiss, "qy"),
        "Horizontal tune shift": scalar(twiss, "qx") - scalar(reference, "qx"),
        "Vertical tune shift": scalar(twiss, "qy") - scalar(reference, "qy"),
        "Horizontal chromaticity Qx_prime": scalar(twiss, "dqx"),
        "Vertical chromaticity Qy_prime": scalar(twiss, "dqy"),
        "Minimum tune separation |C-|": scalar(twiss, "c_minus"),
        "Synchrotron tune Qs": scalar(twiss, "qs"),
        "RMS beta-x beating [%]": 100 * rms(betx_beating),
        "Maximum beta-x beating [%]": 100 * max_abs(betx_beating),
        "RMS beta-y beating [%]": 100 * rms(bety_beating),
        "Maximum beta-y beating [%]": 100 * max_abs(bety_beating),
        "RMS horizontal-orbit residual [um]": 1e6 * rms(delta_x),
        "Maximum horizontal-orbit residual [um]": 1e6 * max_abs(delta_x),
        "RMS vertical-orbit residual [um]": 1e6 * rms(delta_y),
        "Maximum vertical-orbit residual [um]": 1e6 * max_abs(delta_y),
        "RMS horizontal-dispersion residual [mm]": 1e3 * rms(delta_dx),
        "Maximum horizontal-dispersion residual [mm]": 1e3 * max_abs(delta_dx),
        "RMS vertical-dispersion residual [mm]": 1e3 * rms(delta_dy),
        "Maximum vertical-dispersion residual [mm]": 1e3 * max_abs(delta_dy),
        # The sawtooth: what coarse tapering acts on in the first place.
        "Energy sawtooth peak-to-peak [%]": 100 * float(np.ptp(delta)),
        "Energy sawtooth RMS [%]": 100 * rms(delta),
        "Energy deviation from ideal RMS [%]": 100 * rms(delta - column(reference, "delta")),
        "Equilibrium horizontal emittance [nm]": 1e9 * emit_x,
        "Horizontal emittance change [%]": percent_change(emit_x, emit_x_ref),
        "Equilibrium vertical emittance [pm]": 1e12 * emit_y,
        "Vertical emittance change [%]": percent_change(emit_y, emit_y_ref),
        "Equilibrium longitudinal emittance [um]": 1e6
        * scalar(twiss, "eq_gemitt_zeta", "rad_int_eq_gemitt_zeta"),
        "Equilibrium momentum spread [%]": 100
        * scalar(twiss, "eq_sigma_delta", "rad_int_sigma_delta"),
        "Equilibrium bunch length [mm]": 1e3
        * scalar(twiss, "eq_sigma_zeta", "eq_sigma_z", "rad_int_sigma_zeta"),
        "Energy loss per turn [GeV]": 1e-9
        * scalar(twiss, "energy_loss", "eneloss_turn", "rad_int_energy_loss"),
        "Momentum compaction factor": scalar(twiss, "momentum_compaction_factor"),
        "Slip factor": scalar(twiss, "slip_factor"),
    }


def optics_comparison(cases: dict, reference) -> pd.DataFrame:
    """Summary of several lattices side by side, metrics as rows."""
    return pd.DataFrame({label: optics_summary(tw, reference) for label, tw in cases.items()})


def rf_summary(rf_phase: pd.DataFrame, energy_loss_eV: float) -> dict:
    """RF headroom for a machine losing `energy_loss_eV` per turn.

    The voltage is fixed and only the phase is adjusted, so a cavity supplies
    at most q*V0 at crest and sin(phase_s) = (energy to replace) / (q*V0) must
    stay below one. Coarser tapering costs more energy per turn and pushes it
    up, which makes the remaining headroom the first thing to fail as the
    grouping is made coarser -- before any optics quantity does.
    """
    total_voltage = float(np.sum(rf_phase["voltage_MV"])) * 1e6
    ratio = energy_loss_eV / total_voltage if total_voltage else float("nan")
    taper_deg = np.asarray(rf_phase["phase_taper_deg"], dtype=float)
    return {
        "rf_n_cavities": len(rf_phase),
        "rf_total_voltage_MV": total_voltage * 1e-6,
        "rf_sin_phase_s": ratio,
        "rf_headroom_to_crest [%]": 100.0 * (1.0 - ratio),
        "rf_synchronous_phase [deg]": float(np.rad2deg(np.arcsin(ratio)))
        if abs(ratio) <= 1
        else float("nan"),
        "rf_phase_taper_rms [deg]": rms(taper_deg),
        "rf_phase_taper_max [deg]": max_abs(taper_deg),
    }


def corrector_summary(corrections: pd.DataFrame) -> dict:
    """Strength distribution per corrector family, in mT m.

    Max, RMS and the 95th percentile are carried from the start: the magnet
    team needs the distribution, and adding a percentile later would mean
    re-reading every checkpoint.
    """
    summary = {}
    for family, group in corrections.groupby("family"):
        field = np.asarray(group["bl_mtm"], dtype=float)
        used = int(np.count_nonzero(np.abs(np.asarray(group["kick_rad"], dtype=float)) > 0))
        summary.update(
            {
                f"{family}_n_installed": len(group),
                f"{family}_n_used": used,
                f"{family}_bl_rms_mtm": rms(field),
                f"{family}_bl_p95_mtm": percentile_abs(field, 95),
                f"{family}_bl_max_mtm": max_abs(field),
            }
        )
    return summary


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_energy_and_orbit(cases: dict, *, axes=None, figsize=(11, 7)):
    """Energy deviation and horizontal orbit along the ring, several cases.

    `cases` maps a label to a twiss; each is drawn in both panels.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for label, twiss in cases.items():
        s_km = column(twiss, "s") / 1e3
        axes[0].plot(s_km, column(twiss, "delta"), lw=1.3, label=label)
        axes[1].plot(s_km, column(twiss, "x") * 1e3, lw=1.3, label=label)

    axes[0].set_ylabel(r"$\Delta p / p_0$")
    axes[1].set_xlabel(r"$s$ [km]")
    axes[1].set_ylabel(r"$x$ [mm]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    return axes


def plot_optics_comparison(twiss, reference, label, monitor_names, *, figsize=(14, 9.5)):
    """Orbit, dispersion and beta beating against the reference.

    Left column is every twiss location, right column the monitors only, since
    the monitors are what a correction can actually see.
    """
    import matplotlib.pyplot as plt

    monitor_names = list(np.asarray(monitor_names, dtype=str))

    s_all_km = column(reference, "s") / 1e3
    orbit_all = 1e6 * (column(twiss, "x") - column(reference, "x"))
    dispersion_all = 1e3 * (column(twiss, "dx") - column(reference, "dx"))
    beating_all = 100.0 * (column(twiss, "betx") / column(reference, "betx") - 1.0)

    reference_index = indices_of(reference, monitor_names)
    case_index = indices_of(twiss, monitor_names)
    s_monitor_km = column(reference, "s")[reference_index] / 1e3
    orbit_monitor = 1e6 * (column(twiss, "x")[case_index] - column(reference, "x")[reference_index])
    dispersion_monitor = 1e3 * (
        column(twiss, "dx")[case_index] - column(reference, "dx")[reference_index]
    )
    beating_monitor = 100.0 * (
        column(twiss, "betx")[case_index] / column(reference, "betx")[reference_index] - 1.0
    )

    def label_of(values, unit):
        return f"RMS={rms(values):.3g} {unit}, max |.|={max_abs(values):.3g} {unit}"

    figure, axes = plt.subplots(3, 2, figsize=figsize, sharex="col")
    panels = [
        (0, 0, s_all_km, orbit_all, "tab:green", "-", "um"),
        (0, 1, s_monitor_km, orbit_monitor, "tab:green", ".", "um"),
        (1, 0, s_all_km, dispersion_all, "tab:blue", "-", "mm"),
        (1, 1, s_monitor_km, dispersion_monitor, "tab:blue", ".", "mm"),
        (2, 0, s_all_km, beating_all, "tab:red", "-", "%"),
        (2, 1, s_monitor_km, beating_monitor, "tab:red", ".", "%"),
    ]
    for row, col, s, values, colour, style, unit in panels:
        axis = axes[row, col]
        if style == ".":
            axis.plot(s, values, ".", color=colour, ms=3, label=label_of(values, unit))
        else:
            axis.plot(s, values, color=colour, lw=0.9, label=label_of(values, unit))

    axes[0, 0].set_title("All elements")
    axes[0, 1].set_title("Monitors only")
    axes[0, 0].set_ylabel(r"$\Delta x$ [$\mu$m]")
    axes[1, 0].set_ylabel(r"$\Delta D_x$ [mm]")
    axes[2, 0].set_ylabel(r"$\Delta\beta_x/\beta_x$ [%]")
    axes[2, 0].set_xlabel(r"$s$ [km]")
    axes[2, 1].set_xlabel(r"$s$ [km]")

    for axis in axes.flat:
        axis.axhline(0.0, color="black", lw=0.7)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    figure.suptitle(label)
    figure.tight_layout()
    return axes


def plot_correction_history(history, *, tolerance_m=1e-6, ax=None, figsize=(7.5, 4.2)):
    """Monitor residual against iteration, with the requested tolerance."""
    import matplotlib.pyplot as plt

    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)

    ax.semilogy(history["iteration"], 1e6 * history["x_rms_m"], "o-", label="horizontal")
    if "y_rms_m" in history and np.any(np.asarray(history["y_rms_m"]) > 0):
        ax.semilogy(history["iteration"], 1e6 * history["y_rms_m"], "s-", label="vertical")
    ax.axhline(1e6 * tolerance_m, color="black", ls="--", lw=0.9, label="tolerance")
    ax.set_xlabel("Correction iteration")
    ax.set_ylabel(r"Monitor residual RMS [$\mu$m]")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    return ax


def plot_corrector_strengths(corrections, *, ax=None, figsize=(11, 4.5)):
    """Integrated field of each corrector along the ring, one series per family."""
    import matplotlib.pyplot as plt

    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)

    for family, group in corrections.groupby("family"):
        field = np.asarray(group["bl_mtm"], dtype=float)
        ax.plot(
            np.asarray(group["s_m"], dtype=float) / 1e3,
            field,
            ".",
            ms=3,
            label=f"{family}: RMS={rms(field):.3g}, max={max_abs(field):.3g} mT m",
        )
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xlabel(r"$s$ [km]")
    ax.set_ylabel(r"Integrated field $B\ell$ [mT m]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return ax


def plot_tapering_error(applied, *, ax=None, figsize=(11, 4.2)):
    """How far each circuit's single setting is from the ideal of its magnets."""
    import matplotlib.pyplot as plt

    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)

    for family, group in applied.groupby("family"):
        ax.plot(
            np.arange(len(group)),
            np.asarray(group["maximum_error"], dtype=float),
            ".",
            ms=5,
            label=f"{family}: max={group['maximum_error'].max():.3g}",
        )
    ax.set_xlabel("Circuit")
    ax.set_ylabel(r"max $|\delta_{ideal} - \delta_{applied}|$")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    return ax


# --------------------------------------------------------------------------
# Comparison plots
# --------------------------------------------------------------------------


def plot_ladder(steps, metric, *, ax=None, log=True, figsize=(9, 5.5), **kw):
    """One metric along the cumulative tapering steps, one line per branch.

    `steps` is a grouping's `steps.parquet`, or several of them concatenated,
    in which case one line is drawn per granularity and branch.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)

    order = steps.sort_values("step")
    labels = list(dict.fromkeys(order["label"]))
    position = {label: i for i, label in enumerate(labels)}

    keys = [c for c in ("n_circuits", "branch") if c in steps.columns]
    for key, group in order.groupby(keys, sort=True):
        group = group.sort_values("step")
        name = " ".join(
            f"{k}={v}" for k, v in zip(keys, key if isinstance(key, tuple) else (key,))
        )
        ax.plot(
            [position[label] for label in group["label"]],
            group[metric],
            "o-",
            ms=4,
            label=name,
            **kw,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel(metric)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    return ax


def plot_vs_granularity(summary, metric, *, ax=None, log=True, figsize=(9, 5.5), **kw):
    """One metric against the number of circuits per arc, one line per scenario.

    This is the granularity curve the study is about: how each correction
    option degrades as the dipoles are powered in fewer, larger groups.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)

    for vid, group in summary.groupby("vid", sort=True):
        group = group.sort_values("n_circuits")
        name = group["name"].iloc[0].split("-", 1)[1]
        ax.plot(group["n_circuits"], group[metric], "o-", ms=4, label=f"{vid} {name}", **kw)

    ax.set_xlabel("Dipole circuits per arc")
    ax.set_ylabel(metric)
    if log:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    return ax
