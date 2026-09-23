"""The build stages: one function per step of a scenario.

Each function takes a line, modifies it in place, and returns what happened.
They are independent: each does one thing, and any of them can be applied on
its own so that its effect can be measured by itself.

    apply_ideal_tapering       every magnet tapered to its own local energy
    apply_no_tapering          no tapering anywhere, RF phase re-solved
    apply_insertion_tapering   a tapering map written onto the insertions only
    apply_group_tapering       chosen circuit families powered one setting each
    correct_orbit_global       steering correctors or dipole trims

`apply_group_tapering` re-derives the ideal tapering at every iteration of its
fixed point, so its result does not depend on what was applied to the line
before it. Tapering the insertions first gives the RF a viable starting point,
but it does not change where the iteration lands.

`apply_tapering_model` composes the tapering steps into one call, because a
scenario root has to be a single checkpoint. It adds nothing of its own.

Throughout, the orbit correctors are never tapered. In the arcs the dipoles
and the sextupoles always follow their circuits; only the quadrupoles switch
between circuit tapering and ideal magnet-by-magnet tapering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xtrack as xt
from xtrack.twiss import ClosedOrbitSearchError

from tapering import compensate_energy_loss, get_rf_phase

from . import io_ckpt, orbit_corr
from .regions import (
    circuit_groups,
    insertion_elements,
    select_trim_dipoles,
    taperable_elements,
    validate_trim_selection,
)

DIPOLE_CIRCUIT_FAMILIES = ("DIP",)

# Circuits whose magnets follow the quadrupole tapering mode.
QUADRUPOLE_CIRCUIT_FAMILIES = ("QF", "QD")

# Sextupoles are always powered by their circuits: that is what the machine is
# expected to have, so it is not an axis of the study.
SEXTUPOLE_CIRCUIT_FAMILIES = ("SF1", "SF2", "SD1", "SD2")

# Families that are always powered from their circuits.
ALWAYS_GROUPED_FAMILIES = (*DIPOLE_CIRCUIT_FAMILIES, *SEXTUPOLE_CIRCUIT_FAMILIES)

# Which families each quadrupole mode of the grid groups, so a scenario name
# can be turned into a family list without anyone editing a list by hand.
# Keyed by the values of `grid.QUADS`. There is nothing to do for ideally
# tapered quadrupoles beyond leaving them out: every magnet outside the
# grouped families keeps its own ideal tapering.
GROUPED_FAMILIES_BY_QUADS = {
    "coarse": (*ALWAYS_GROUPED_FAMILIES, *QUADRUPOLE_CIRCUIT_FAMILIES),
    "ideal": ALWAYS_GROUPED_FAMILIES,
}


@dataclass(frozen=True)
class TaperingConfig:
    """Settings of the tapering fixed point."""

    max_iter: int = 10
    tolerance: float = 1e-9
    cavity_midpoint: str = "qd18f.0"
    rf_warn_ratio: float = 0.9
    verbose: bool = False

    def as_dict(self) -> dict:
        return {
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "cavity_midpoint": self.cavity_midpoint,
            "rf_warn_ratio": self.rf_warn_ratio,
        }


@dataclass(frozen=True)
class OrbitCorrectionConfig:
    """Settings of the global orbit correction."""

    tolerance_m: float = 1e-6
    max_iterations: int = 10
    gain: float = 1.0
    monitor_pattern: str = "bpm.*"
    # Steering correctors only.
    rcond: float = 1e-6
    n_singular_values: int | None = None
    # Dipole trims only.
    trim_position: int = 1
    regularization: float = 1e-6
    solver_max_iterations: int = 400
    verbose: bool = False

    def as_dict(self) -> dict:
        return {
            "tolerance_m": self.tolerance_m,
            "max_iterations": self.max_iterations,
            "gain": self.gain,
            "monitor_pattern": self.monitor_pattern,
            "rcond": self.rcond,
            "n_singular_values": self.n_singular_values,
            "trim_position": self.trim_position,
            "regularization": self.regularization,
            "solver_max_iterations": self.solver_max_iterations,
        }


@dataclass
class StageResult:
    """What a stage did, for the checkpoint's provenance and metrics.

    Two separate questions, because they have different answers:

    `converged`         the iteration settled and left a usable machine
    `reached_tolerance` it also got below the tolerance it was asked for

    A stage can settle at a residual its configured tolerance cannot reach.
    That is a fixed point, not a failure, so the two are reported apart
    instead of being collapsed into one flag.
    """

    stage: str
    converged: bool
    iterations: int
    reached_tolerance: bool = True
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    corrections: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Small and JSON-serialisable: this is what goes into meta.json.
    info: dict = field(default_factory=dict)
    # Per-magnet ideal tapering, kept out of `info` because it has one entry
    # per taperable element and does not belong in the provenance file.
    taper_reference: dict | None = None


# --------------------------------------------------------------------------
# Reading and writing the tapering
# --------------------------------------------------------------------------


def read_tapering(line, elements=None) -> dict:
    """Current tapering value of each taperable element."""
    if elements is None:
        elements = taperable_elements(line)
    return {str(name): float(line[str(name)].delta_taper) for name in elements}


def clear_tapering(line, elements=None) -> None:
    """Remove the tapering, leaving every magnet at its nominal strength."""
    if elements is None:
        elements = taperable_elements(line)
    for name in elements:
        line[str(name)].delta_taper = 0.0


def restore_tapering(line, taper_by_name, elements=None) -> None:
    """Write stored tapering values back onto the line."""
    if elements is None:
        elements = taper_by_name.keys()
    for name in elements:
        line[str(name)].delta_taper = float(taper_by_name[str(name)])


def read_rf_phase(line) -> pd.DataFrame:
    """Phase of each powered cavity.

    `phase_taper` is what the tapering added; the total phase the cavity sits
    at is the lattice's own base phase plus that, and lattices differ in
    whether they carry the base in `lag` or in `phase`, so both are summed.
    """
    rows = []
    for name in line.element_names:
        element = line[name]
        if not hasattr(element, "voltage"):
            continue

        voltage = float(element.voltage)
        if voltage == 0:
            continue

        phase_taper = float(getattr(element, "phase_taper", 0.0))
        base_phase = np.deg2rad(float(getattr(element, "lag", 0.0))) + float(
            getattr(element, "phase", 0.0)
        )
        rows.append(
            {
                "name": name,
                "voltage_MV": voltage * 1e-6,
                "phase_taper_rad": phase_taper,
                "phase_taper_deg": np.rad2deg(phase_taper),
                "phase_total_rad": base_phase + phase_taper,
                "phase_total_deg": np.rad2deg(base_phase + phase_taper),
            }
        )
    return pd.DataFrame(rows)


def corrector_elements(line) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical orbit correctors, among the taperable elements."""
    taperable = taperable_elements(line)
    horizontal = np.asarray([n for n in taperable if n.startswith("hcor")], dtype=str)
    vertical = np.asarray([n for n in taperable if n.startswith("vcor")], dtype=str)
    return horizontal, vertical


# --------------------------------------------------------------------------
# Reference lattices
# --------------------------------------------------------------------------


def apply_ideal_tapering(line) -> StageResult:
    """Taper every magnet to its own local energy: the ideal machine.

    This is xsuite's own compensation, and it is the reference every other
    scenario is compared against.
    """
    line.compensate_radiation_energy_loss(delta0="zero_mean")
    return StageResult(
        stage="taper",
        converged=True,
        iterations=0,
        info={"model": "individual"},
        taper_reference=read_tapering(line),
    )


def apply_no_tapering(line, *, cfg: TaperingConfig = TaperingConfig()) -> StageResult:
    """Remove all tapering and re-solve the RF phase for the untapered machine."""
    clear_tapering(line)
    result = get_rf_phase(
        line,
        max_iter=cfg.max_iter,
        rf_warn_ratio=cfg.rf_warn_ratio,
        verbose=cfg.verbose,
    )
    return StageResult(
        stage="taper",
        converged=result["particle_on_co"] is not None,
        iterations=len(result["eloss_history_eV"]),
        info={
            "model": "none",
            "eloss_eV": result["eloss_eV"],
            "energy_error_eV": result["energy_error_eV"],
        },
    )


# --------------------------------------------------------------------------
# The tapering model
# --------------------------------------------------------------------------


def apply_insertion_tapering(
    line,
    *,
    regions: dict,
    ideal_taper: dict,
    cfg: TaperingConfig = TaperingConfig(),
) -> StageResult:
    """Taper the insertions magnet by magnet and re-solve the RF phase.

    Takes the regions and a tapering map, and writes that map onto the
    insertion magnets only. Nothing else on the line is touched, so the effect
    of tapering the insertions alone can be measured.
    """
    insertions = [name for name in insertion_elements(regions) if name in ideal_taper]
    restore_tapering(line, ideal_taper, insertions)
    result = get_rf_phase(
        line,
        max_iter=cfg.max_iter,
        rf_warn_ratio=cfg.rf_warn_ratio,
        verbose=cfg.verbose,
    )
    return StageResult(
        stage="taper",
        converged=result["particle_on_co"] is not None,
        iterations=len(result["eloss_history_eV"]),
        info={
            "model": "insertions",
            "n_insertion_elements": len(insertions),
            "eloss_eV": result["eloss_eV"],
            "energy_error_eV": result["energy_error_eV"],
        },
    )


def apply_group_tapering(
    line,
    *,
    circuits: pd.DataFrame,
    families,
    untapered_families=(),
    cfg: TaperingConfig = TaperingConfig(),
) -> StageResult:
    """Power the given circuit families from one setting each.

    Every circuit of `families` is driven to the mean of the ideal tapering of
    its magnets, and the RF is set to replace the energy actually lost. The
    mean is recomputed from the closed orbit at every iteration, so the
    grouping and the orbit it produces are solved together rather than the
    grouping being averaged once against a fixed reference.

    Magnets fall into three cases:

        `families`            one setting per circuit
        `untapered_families`  held at zero, not tapered at all
        everything else       ideal tapering, magnet by magnet

    The third case covers the magnets that belong to no circuit -- the
    insertions, and the dispersion-suppressor quadrupoles and sextupoles
    outside the main circuit families -- so they stay ideally tapered whatever
    the arc circuits are doing. The orbit correctors are never tapered.

    This is a fixed-point iteration that re-derives the ideal tapering each
    time round, so its result does not depend on how the line was tapered
    beforehand. It can be applied to a bare line, or on top of the insertion
    tapering, and reach the same machine.
    """
    families = tuple(families)
    untapered_families = tuple(untapered_families)
    overlap = set(families) & set(untapered_families)
    if overlap:
        raise ValueError(
            f"families cannot be both grouped and untapered: {sorted(overlap)}"
        )

    mean_groups = circuit_groups(circuits, families=families) if families else {}
    zero_groups = (
        circuit_groups(circuits, families=untapered_families)
        if untapered_families
        else {}
    )
    if not mean_groups and not zero_groups:
        raise ValueError(
            f"no circuits found for families {families} or {untapered_families}"
        )

    setters_mean = [
        xt.MultiSetter(line, elements, field="delta_taper")
        for elements in mean_groups.values()
    ]
    setters_zero = [
        xt.MultiSetter(line, elements, field="delta_taper")
        for elements in zero_groups.values()
    ]
    # Correctors are excluded from the tapering entirely.
    setters_zero += [
        xt.MultiSetter(line, correctors, field="delta_taper")
        for correctors in corrector_elements(line)
        if len(correctors)
    ]

    # The fixed point reports nothing about its own convergence, so the
    # per-iteration hook is used to record it.
    eloss_history = []

    def record(_line, iteration, eloss):
        eloss_history.append({"iteration": iteration, "eloss": float(eloss)})

    compensate_energy_loss(
        line,
        delta_taper_setters_mean=setters_mean,
        delta_taper_setters_no_tapering=setters_zero,
        # Everything outside the circuits keeps the ideal tapering.
        apply_full_taper=True,
        cavity_midpoint=cfg.cavity_midpoint,
        max_iter=cfg.max_iter,
        tolerance=cfg.tolerance,
        rf_warn_ratio=cfg.rf_warn_ratio,
        on_iteration=record,
        verbose=cfg.verbose,
    )

    # The loop breaks on the tolerance before the hook runs, so fewer
    # recorded iterations than the maximum means the tolerance was met.
    iterations = len(eloss_history)
    reached_tolerance = iterations < cfg.max_iter

    # The absolute tolerance is not reachable on every machine, and failing to
    # reach it does not mean the iteration diverged. What decides whether the
    # tapering succeeded is that it left a machine with a closed orbit; how
    # far the energy loss still moved is reported as a number alongside.
    try:
        line.find_closed_orbit()
        converged = True
    except ClosedOrbitSearchError:
        converged = False

    p0c = float(np.asarray(line.particle_ref.p0c).ravel()[0])
    losses = [entry["eloss"] for entry in eloss_history]
    eloss_change_eV = abs(losses[-1] - losses[-2]) * p0c if len(losses) > 1 else float("nan")

    return StageResult(
        stage="taper",
        converged=converged,
        reached_tolerance=reached_tolerance,
        iterations=iterations,
        history=pd.DataFrame(eloss_history),
        info={
            "model": "groups",
            "families": list(families),
            "untapered_families": list(untapered_families),
            "n_circuits": len(setters_mean),
            "n_magnets": int(sum(len(g) for g in mean_groups.values())),
            "eloss_eV": losses[-1] * p0c if losses else float("nan"),
            "eloss_change_eV": eloss_change_eV,
            "tolerance": cfg.tolerance,
        },
    )


def apply_tapering_model(
    line,
    *,
    regions: dict,
    circuits: pd.DataFrame,
    families=ALWAYS_GROUPED_FAMILIES,
    cfg: TaperingConfig = TaperingConfig(),
) -> StageResult:
    """One coarse tapering model, as a scenario root needs it.

    A convenience composition of the steps above, used where the whole root
    has to be one call: measure the ideal tapering, clear it, taper the
    insertions, then power `families` from their circuits. Every magnet not in
    those families keeps the ideal magnet-by-magnet tapering.

    The steps are independent and can be applied separately; they are bundled
    here only because a scenario root is a single checkpoint.
    """
    ideal = apply_ideal_tapering(line).taper_reference
    clear_tapering(line)
    insertions = apply_insertion_tapering(
        line, regions=regions, ideal_taper=ideal, cfg=cfg
    )
    result = apply_group_tapering(line, circuits=circuits, families=families, cfg=cfg)
    result.info["n_insertion_elements"] = insertions.info["n_insertion_elements"]
    result.taper_reference = ideal
    return result


def applied_tapering(line, circuits: pd.DataFrame, ideal_taper: dict) -> pd.DataFrame:
    """Per circuit, the setting applied and how far it is from ideal.

    The maximum error is the direct measure of how coarse a grouping is,
    before any correction hides it.
    """
    rows = []
    for circuit, group in circuits.groupby("circuit", sort=False):
        elements = [name for name in group["element"] if name in ideal_taper]
        if not elements:
            continue
        applied = np.asarray([float(line[name].delta_taper) for name in elements])
        ideal = np.asarray([ideal_taper[name] for name in elements])
        rows.append(
            {
                "circuit": circuit,
                "family": group["family"].iloc[0],
                "sector": group["sector"].iloc[0],
                "number_of_elements": len(elements),
                "applied_delta_taper": float(np.mean(applied)),
                "ideal_min": float(np.min(ideal)),
                "ideal_max": float(np.max(ideal)),
                "maximum_error": float(np.max(np.abs(ideal - applied))),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Global orbit correction
# --------------------------------------------------------------------------


def _settled(values, *, floor: float = 0.0, rtol: float = 1e-2) -> bool:
    """True when the iteration stopped moving, or already sits below `floor`.

    The floor matters for a plane that is not being corrected at all: its
    residual is numerical noise, where the relative change between two
    iterations is meaningless and can be of order one.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return False
    last, previous = values[-1], values[-2]
    if abs(last) <= floor:
        return True
    return abs(last - previous) <= rtol * abs(last)


def _monitor_names(line, pattern: str) -> list:
    # Plain str, never numpy str: an element name used to index element_refs
    # ends up in the serialised expressions, and numpy reprs itself as
    # `np.str_('...')`, which the loader cannot evaluate.
    return [str(name) for name in line.get_table().rows[pattern].name]


def _steering_correctors(line) -> tuple[np.ndarray, np.ndarray]:
    """Thin correctors used for steering, by name prefix."""
    table = line.get_table()
    multipoles = table.rows[table.element_type == "Multipole"]
    return (
        [str(name) for name in multipoles.rows["hcor.*"].name],
        [str(name) for name in multipoles.rows["vcor.*"].name],
    )


def corrections_table(
    line,
    names,
    kicks_rad,
    *,
    family: str,
    plane: str,
    brho_tm: float | None = None,
    s_m=None,
) -> pd.DataFrame:
    """Applied strengths of one corrector family, ready to be written.

    Converts kick angles to integrated field, looking up each element's
    position and length on the line. Pass every corrector the correction could
    have used, including those that ended at zero: a corrector driven to
    nothing has to stay distinguishable from one that was never touched.

        table = stages.corrections_table(
            line, trim_names, kicks, family="dip_trim", plane="x")
        io_ckpt.write_corrections(directory, table)
    """
    names = [str(name) for name in names]
    kicks = np.asarray(kicks_rad, dtype=float)
    if len(names) != len(kicks):
        raise ValueError(f"{len(names)} correctors but {len(kicks)} kicks")

    if brho_tm is None:
        brho_tm = orbit_corr.magnetic_rigidity(line)
    if s_m is None:
        table = line.get_table()
        position = dict(
            zip(np.asarray(table.name, dtype=str), np.asarray(table.s, dtype=float))
        )
        s_m = [position[name] for name in names]

    lengths = np.asarray([float(getattr(line[name], "length", 0.0)) for name in names])
    bl_mtm = kicks * brho_tm * 1e3

    return pd.DataFrame(
        {
            "family": family,
            "plane": plane,
            "name": names,
            "s_m": np.asarray(s_m, dtype=float),
            "length_m": lengths,
            "kick_rad": kicks,
            "bl_mtm": bl_mtm,
            # Thin correctors have no length, so a field per metre is undefined.
            "b_mt": np.where(lengths > 0, bl_mtm / np.where(lengths > 0, lengths, 1.0), np.nan),
        }
    )


def _empty_corrections() -> pd.DataFrame:
    return pd.DataFrame(columns=list(io_ckpt.CORRECTION_COLUMNS))


def correct_orbit_with_correctors(
    line,
    *,
    reference_twiss,
    cfg: OrbitCorrectionConfig = OrbitCorrectionConfig(),
    taper_cfg: TaperingConfig = TaperingConfig(),
) -> StageResult:
    """Correct both planes onto the reference orbit with the steering correctors."""
    monitors = _monitor_names(line, cfg.monitor_pattern)
    horizontal, vertical = _steering_correctors(line)

    line.steering_correctors_x = horizontal
    line.steering_correctors_y = vertical
    line.steering_monitors_x = monitors
    line.steering_monitors_y = monitors

    correction = orbit_corr.TrajectoryCorrection(
        line=line,
        # The response matrix is built on the ideal optics, not the distorted one.
        twiss_table=reference_twiss,
        monitor_names_x=monitors,
        corrector_names_x=horizontal,
        monitor_names_y=monitors,
        corrector_names_y=vertical,
        n_singular_values=cfg.n_singular_values,
    )
    correction.x_correction.set_reference_orbit_from_twiss(reference_twiss)
    correction.y_correction.set_reference_orbit_from_twiss(reference_twiss)

    history, reached_tolerance = orbit_corr.correct_orbit_with_correctors(
        correction,
        tolerance_x=cfg.tolerance_m,
        tolerance_y=cfg.tolerance_m,
        max_iterations=cfg.max_iterations,
        n_singular_values=cfg.n_singular_values,
        rcond=cfg.rcond,
        gain=cfg.gain,
        verbose=cfg.verbose,
    )
    # Stopping above the tolerance is not a failure if the residual stopped
    # improving: the correction reached what these correctors can do.
    converged = reached_tolerance or (
        _settled(history["x_rms_m"], floor=cfg.tolerance_m)
        and _settled(history["y_rms_m"], floor=cfg.tolerance_m)
    )

    # The correctors changed the orbit, so the energy loss changed with it.
    get_rf_phase(
        line,
        max_iter=taper_cfg.max_iter,
        rf_warn_ratio=taper_cfg.rf_warn_ratio,
        verbose=cfg.verbose,
    )

    brho = orbit_corr.magnetic_rigidity(line)
    corrections = pd.concat(
        [
            corrections_table(
                line,
                single.corrector_names,
                single.get_kick_values(),
                family=family,
                plane=plane,
                brho_tm=brho,
                s_m=single.s_correctors,
            )
            for plane, family, single in (
                ("x", "hcor", correction.x_correction),
                ("y", "vcor", correction.y_correction),
            )
        ],
        ignore_index=True,
    )

    return StageResult(
        stage="global_oc_hcorr",
        converged=converged,
        reached_tolerance=reached_tolerance,
        iterations=int(history["iteration"].iloc[-1]),
        history=history,
        corrections=corrections,
        info={
            "brho_tm": brho,
            "n_monitors": len(monitors),
            "n_correctors_x": len(horizontal),
            "n_correctors_y": len(vertical),
            "final_rms_x_m": float(history["x_rms_m"].iloc[-1]),
            "final_rms_y_m": float(history["y_rms_m"].iloc[-1]),
        },
    )


def correct_orbit_with_dipole_trims(
    line,
    *,
    regions: dict,
    reference_twiss,
    cfg: OrbitCorrectionConfig = OrbitCorrectionConfig(),
    taper_cfg: TaperingConfig = TaperingConfig(),
) -> StageResult:
    """Correct the horizontal orbit with one trim per arc cell.

    `cfg.trim_position` selects which dipole of each three-dipole cell carries
    the trim. There is no vertical equivalent, so this stage leaves the
    vertical plane untouched.
    """
    monitors = _monitor_names(line, cfg.monitor_pattern)
    trims, layout = select_trim_dipoles(line, regions, position=cfg.trim_position)
    validate_trim_selection(line, regions, trims, layout, position=cfg.trim_position)

    base_knl = orbit_corr.prepare_dipole_trims(line, trims)
    orbit_corr.verify_dipole_trims(line, trims, base_knl)

    history, kicks, reached_tolerance = orbit_corr.correct_orbit_with_dipole_trims(
        line,
        reference=reference_twiss,
        monitor_names=monitors,
        trim_dipole_names=trims,
        trim_base_knl=base_knl,
        tolerance_rms_m=cfg.tolerance_m,
        max_iterations=cfg.max_iterations,
        gain=cfg.gain,
        regularization=cfg.regularization,
        solver_max_iterations=cfg.solver_max_iterations,
        verbose=cfg.verbose,
    )
    converged = reached_tolerance or _settled(history["x_rms_m"], floor=cfg.tolerance_m)

    get_rf_phase(
        line,
        max_iter=taper_cfg.max_iter,
        rf_warn_ratio=taper_cfg.rf_warn_ratio,
        verbose=cfg.verbose,
    )

    brho = orbit_corr.magnetic_rigidity(line)
    corrections = corrections_table(
        line, trims, kicks, family="dip_trim", plane="x",
        brho_tm=brho, s_m=layout["s_m"],
    )

    return StageResult(
        stage="global_oc_trim",
        converged=converged,
        reached_tolerance=reached_tolerance,
        iterations=int(history["iteration"].iloc[-1]),
        history=history,
        corrections=corrections,
        info={
            "brho_tm": brho,
            "n_monitors": len(monitors),
            "n_trims": len(trims),
            "trim_position": cfg.trim_position,
            "final_rms_x_m": float(history["x_rms_m"].iloc[-1]),
        },
    )


def correct_orbit_global(
    line,
    *,
    method: str,
    reference_twiss,
    regions: dict | None = None,
    cfg: OrbitCorrectionConfig = OrbitCorrectionConfig(),
    taper_cfg: TaperingConfig = TaperingConfig(),
) -> StageResult:
    """Dispatch to the requested global orbit correction."""
    if method == "hcorr":
        return correct_orbit_with_correctors(
            line, reference_twiss=reference_twiss, cfg=cfg, taper_cfg=taper_cfg
        )
    if method == "dip_trim":
        if regions is None:
            raise ValueError("the dipole-trim correction needs the lattice regions")
        return correct_orbit_with_dipole_trims(
            line,
            regions=regions,
            reference_twiss=reference_twiss,
            cfg=cfg,
            taper_cfg=taper_cfg,
        )
    raise ValueError(f"unknown global orbit correction method: {method!r}")
