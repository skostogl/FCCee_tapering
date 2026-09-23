"""Building scenarios into checkpoints.

`get(name)` is memoised recursion: return the checkpoint if it exists and its
stamp is still valid, otherwise build the parent, load its lattice, apply the
one missing stage, twiss, and write. Asking for a leaf materialises every
ancestor along the way, each a full checkpoint in its own right, and nothing is
ever built twice.

Each checkpoint stores a stamp over the parent's stamp, the stage and its
parameters, the circuit definition and the code version. `get` rebuilds when it
does not match, so fixing a stage does not leave downstream checkpoints
silently built with the old one.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import xobjects as xo
import xtrack as xt

from . import grid, io_ckpt, regions as rg, stages

TWISS_CONFIG = {"radiation_analysis": True}


def code_version() -> str:
    """Git SHA of the working tree, so a stamp pins the code that built it."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class Builder:
    """Builds the scenarios of one energy mode into a checkpoint tree."""

    def __init__(
        self,
        lattice: str | Path,
        *,
        mode: str = "t",
        root: str | Path = "checkpoints",
        taper_cfg: stages.TaperingConfig = stages.TaperingConfig(),
        orbit_cfg: stages.OrbitCorrectionConfig = stages.OrbitCorrectionConfig(),
        verbose: bool = True,
    ):
        self.lattice = str(lattice)
        self.mode = mode
        self.root = Path(root)
        self.taper_cfg = taper_cfg
        self.orbit_cfg = orbit_cfg
        self.verbose = verbose
        self.code = code_version()

        self._regions = None
        self._circuits: dict[str, pd.DataFrame] = {}
        self._reference_twiss = None

    # ----------------------------------------------------------------
    # Lattice and per-grouping data
    # ----------------------------------------------------------------

    def load_line(self):
        line = xt.load(self.lattice).fccee_p_ring
        line.build_tracker(_context=xo.ContextCpu())
        line.configure_radiation(model="mean")
        return line

    def regions(self):
        if self._regions is None:
            self._regions = rg.split_lattice_regions(self.load_line())
        return self._regions

    def circuits(self, grouping: str) -> pd.DataFrame | None:
        """Circuit definition of a grouping, written once and reused.

        Reference groupings have no circuits: nothing is powered in groups.
        """
        if grouping in grid.REFERENCE_GROUPINGS:
            return None
        if grouping not in self._circuits:
            n_circuits, _scope = grid.parse_grouping(grouping)
            line = self.load_line()
            circuits = rg.define_circuits(
                line, self.regions(), n_dipole_circuits_per_sector=n_circuits
            )
            io_ckpt.write_circuits(
                io_ckpt.grouping_dir(self.root, self.mode, grouping), circuits
            )
            self._circuits[grouping] = circuits
        return self._circuits[grouping]

    def reference_twiss(self):
        """Twiss of the ideally tapered machine, which every metric uses."""
        if self._reference_twiss is None:
            directory = self.get(grid.METRIC_REFERENCE)
            line = io_ckpt.read_line(directory)
            line.build_tracker(_context=xo.ContextCpu())
            self._reference_twiss = line.twiss(**TWISS_CONFIG)
        return self._reference_twiss

    # ----------------------------------------------------------------
    # Stamping
    # ----------------------------------------------------------------

    def stage_params(self, name: str) -> dict:
        """Everything that changes what a stage does, for the stamp."""
        scenario = grid.parse(name)
        stage = grid.stage_applied(name)
        params: dict = {"stage": stage, "lattice": self.lattice, "mode": self.mode}

        if stage == grid.STAGE_ROOT:
            params.update(self.taper_cfg.as_dict())
            params["quads"] = scenario.quads
            if scenario.grouping not in grid.REFERENCE_GROUPINGS:
                params["families"] = list(
                    stages.GROUPED_FAMILIES_BY_QUADS[scenario.quads]
                )
                params["circuits"] = io_ckpt.circuits_digest(
                    self.circuits(scenario.grouping)
                )
        else:
            params.update(self.orbit_cfg.as_dict())
            params.update(self.taper_cfg.as_dict())
        return params

    def stamp_of(self, name: str, parent_stamp: str | None) -> str:
        return io_ckpt.stamp(
            parent_stamp, grid.stage_applied(name), self.stage_params(name), self.code
        )

    # ----------------------------------------------------------------
    # Building
    # ----------------------------------------------------------------

    def _apply_stage(self, line, name: str) -> stages.StageResult:
        """Apply the one stage that turns this scenario's parent into it."""
        scenario = grid.parse(name)
        stage = grid.stage_applied(name)

        if stage == grid.STAGE_ROOT:
            if scenario.grouping == "CTind":
                return stages.apply_ideal_tapering(line)
            if scenario.grouping == "CToff":
                return stages.apply_no_tapering(line, cfg=self.taper_cfg)
            return stages.apply_tapering_model(
                line,
                regions=self.regions(),
                circuits=self.circuits(scenario.grouping),
                families=stages.GROUPED_FAMILIES_BY_QUADS[scenario.quads],
                cfg=self.taper_cfg,
            )

        if scenario.glob is not None:
            return stages.correct_orbit_global(
                line,
                method=scenario.glob,
                reference_twiss=self.reference_twiss(),
                regions=self.regions(),
                cfg=self.orbit_cfg,
                taper_cfg=self.taper_cfg,
            )

        raise NotImplementedError(
            f"the stage {stage!r} needed by {name} is not implemented yet"
        )

    def get(self, name: str, *, force: bool = False) -> Path:
        """Checkpoint of a scenario, building it and its ancestors if needed."""
        if not grid.is_implemented(name):
            raise NotImplementedError(
                f"{name} needs the stage "
                f"{grid.stage_applied(name)!r}, which is not implemented yet"
            )

        scenario = grid.parse(name)
        directory = io_ckpt.checkpoint_dir(self.root, self.mode, scenario.grouping, name)
        parent = grid.parent(name)

        # Built at most once, however many times it is needed below: with
        # force=True a second call would rebuild the whole ancestry again.
        parent_dir = None
        parent_stamp = None
        if parent is not None:
            parent_dir = self.get(parent, force=force)
            parent_stamp = io_ckpt.read_meta(parent_dir)["stamp"]

        stamp = self.stamp_of(name, parent_stamp)
        expect_corrections = scenario.glob is not None
        if not force and io_ckpt.is_valid(
            directory, stamp, expect_corrections=expect_corrections
        ):
            self._log(f"  {name}: up to date")
            return directory

        started = time.time()
        if parent_dir is None:
            line = self.load_line()
        else:
            line = io_ckpt.read_line(parent_dir)
            line.build_tracker(_context=xo.ContextCpu())

        result = self._apply_stage(line, name)
        twiss = line.twiss(**TWISS_CONFIG)
        wall_time = time.time() - started

        self._write(directory, name, line, twiss, result, stamp, parent, wall_time)
        self._write_metrics(directory, name, line, twiss, result)
        self._log(
            f"  {name}: built in {wall_time:.0f}s "
            f"(settled={result.converged}, tolerance={result.reached_tolerance})"
        )
        return directory

    def _write(self, directory, name, line, twiss, result, stamp, parent, wall_time):
        scenario = grid.parse(name)
        io_ckpt.write_line(
            directory,
            line,
            name=name,
            mode=self.mode,
            metadata={
                "grouping": scenario.grouping,
                "vid": scenario.vid,
                "stamp": stamp,
                "stage_applied": result.stage,
                "lattice": Path(self.lattice).name,
            },
        )
        io_ckpt.write_twiss(
            directory, twiss, ident=io_ckpt.twiss_id(stamp, TWISS_CONFIG)
        )
        if len(result.corrections):
            io_ckpt.write_corrections(directory, result.corrections)
        io_ckpt.write_meta(
            directory,
            {
                "name": name,
                "vid": scenario.vid,
                "mode": self.mode,
                "grouping": scenario.grouping,
                "quads": scenario.quads,
                "glob": scenario.glob,
                "parent": parent,
                "stage_applied": result.stage,
                "stage_params": self.stage_params(name),
                "converged": result.converged,
                "reached_tolerance": result.reached_tolerance,
                "iterations": result.iterations,
                "stage_info": result.info,
                "stamp": stamp,
                "code_version": self.code,
                "xsuite_version": f"xtrack {xt.__version__}",
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wall_time_s": round(wall_time, 1),
            },
        )

    def _write_metrics(self, directory, name, line, twiss, result) -> None:
        """Cache the cheap figures of merit next to the checkpoint.

        Written while the lattice is still in memory, which is the only time
        the RF phase is free to read. The file is a cache: deleting it costs a
        recomputation, not a rebuild.
        """
        from . import metrics, stages

        # The reference measures itself, which would otherwise recurse.
        reference = twiss if name == grid.METRIC_REFERENCE else self.reference_twiss()

        row = metrics.optics_summary(twiss, reference)
        row.update(metrics.corrector_summary(result.corrections))
        row.update(
            metrics.rf_summary(
                stages.read_rf_phase(line), float(metrics.scalar(twiss, "energy_loss"))
            )
        )
        row.update(
            {
                "converged": result.converged,
                "reached_tolerance": result.reached_tolerance,
                "iterations": result.iterations,
            }
        )
        if "eloss_change_eV" in result.info:
            row["Tapering residual [eV/iteration]"] = result.info["eloss_change_eV"]
        io_ckpt.write_metrics(directory, row)

    # Cumulative tapering steps, for reading off what each family costs.
    # Families not yet reached are held at zero, so each step shows what
    # tapering that family at all buys. Magnets outside the main circuits --
    # the insertions and the dispersion-suppressor quadrupoles and sextupoles
    # -- stay ideally tapered from the first step on, so the only thing
    # changing along the ladder is the arc circuits.
    #
    # The quadrupoles branch: they are either powered from their circuits or
    # tapered magnet by magnet, which is the grid's Q axis. Both branches end
    # on a grid root, so the ladder joins the scan without a gap.
    _DIP = stages.DIPOLE_CIRCUIT_FAMILIES
    _QUAD = stages.QUADRUPOLE_CIRCUIT_FAMILIES
    _SEXT = stages.SEXTUPOLE_CIRCUIT_FAMILIES
    _ALL_CIRCUITS = (*_DIP, *_QUAD, *_SEXT)

    DECOMPOSITION_STEPS = (
        # step, branch, label, grouped families, untapered families
        (1, "both", "Insertions only", (), _ALL_CIRCUITS),
        (2, "both", "+ dipole circuits", _DIP, (*_QUAD, *_SEXT)),
        (3, "coarse", "+ quadrupole circuits", (*_DIP, *_QUAD), _SEXT),
        (4, "coarse", "+ sextupole circuits", (*_DIP, *_QUAD, *_SEXT), ()),
        (3, "ideal", "+ quadrupoles ideal", _DIP, _SEXT),
        (4, "ideal", "+ sextupole circuits", (*_DIP, *_SEXT), ()),
    )

    STEPS_PARQUET = "steps.parquet"

    def build_steps(self, grouping: str, *, force: bool = False) -> pd.DataFrame:
        """Metrics of each cumulative tapering step of one grouping.

        Applied in sequence on one lattice, which is only for convenience:
        each step re-derives the ideal tapering, so its result does not depend
        on the steps before it. That is also why the two quadrupole branches
        can share a lattice.

        Saved as `steps.parquet` beside the grouping's scenarios. Only the
        figures of merit are kept: the intermediate lattices are diagnostics,
        not scenarios, and the ones with untapered quadrupoles cannot be
        orbit-corrected anyway.
        """
        from . import metrics

        directory = io_ckpt.grouping_dir(self.root, self.mode, grouping)
        path = directory / self.STEPS_PARQUET
        if path.exists() and not force:
            self._log(f"{grouping}: steps up to date")
            return pd.read_parquet(path)

        circuits = self.circuits(grouping)
        reference = self.reference_twiss()
        line = self.load_line()
        stages.apply_ideal_tapering(line)

        rows = []
        for step, branch, label, families, untapered in self.DECOMPOSITION_STEPS:
            started = time.time()
            result = stages.apply_group_tapering(
                line,
                circuits=circuits,
                families=families,
                untapered_families=untapered,
                cfg=self.taper_cfg,
            )
            twiss = line.twiss(**TWISS_CONFIG)

            row = {
                "mode": self.mode,
                "grouping": grouping,
                "n_circuits": grid.parse_grouping(grouping)[0],
                "step": step,
                "branch": branch,
                "label": label,
                "families": ",".join(families),
                "untapered_families": ",".join(untapered),
                "converged": result.converged,
                "reached_tolerance": result.reached_tolerance,
            }
            row.update(metrics.optics_summary(twiss, reference))
            row.update(
                metrics.rf_summary(
                    stages.read_rf_phase(line),
                    float(metrics.scalar(twiss, "energy_loss")),
                )
            )
            rows.append(row)
            self._log(
                f"  {grouping} step {step} [{branch}] {label}: "
                f"{time.time() - started:.0f}s"
            )

        frame = pd.DataFrame(rows)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    def build_all(self, grouping: str, *, force: bool = False) -> list[Path]:
        """Every implemented scenario of a grouping, in dependency order.

        Safe to interrupt and re-run: anything already valid is skipped.
        """
        self._log(f"{grouping}:")
        built = []
        for name in grid.implemented_scenarios(grouping):
            try:
                built.append(self.get(name, force=force))
            except Exception as error:  # one bad scenario must not lose the rest
                self._log(f"  {name}: FAILED {type(error).__name__}: {error}")
        return built

    def build_references(self, *, force: bool = False) -> list[Path]:
        """The two reference machines every scenario is measured against."""
        self._log("references:")
        return [self.get(name, force=force) for name in grid.REFERENCE_SCENARIOS]

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)
