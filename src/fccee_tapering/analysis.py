"""Summary tables across scenarios and granularities.

The build is chained; the analysis is not. Every row here is produced by one
function applied identically to every scenario, reading a checkpoint cold, so
a chained state cannot be told from a from-scratch one and chaining cannot
bias the comparison.

Iteration is over the registry, never over a directory listing: a missing
checkpoint raises instead of quietly producing a shorter table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import grid, io_ckpt, metrics

# Columns identifying a row, kept in front of the figures of merit.
INDEX_COLUMNS = (
    "mode",
    "grouping",
    "n_circuits",
    "name",
    "vid",
    "quads",
    "glob",
    "parent",
    "stage_applied",
)

CONVERGENCE_COLUMNS = ("converged", "reached_tolerance", "iterations", "chain_converged")


def _reference_view(root, mode):
    scenario = grid.parse(grid.METRIC_REFERENCE)
    directory = io_ckpt.checkpoint_dir(root, mode, scenario.grouping, scenario.name)
    if not io_ckpt.is_complete(directory):
        raise FileNotFoundError(
            f"the reference {grid.METRIC_REFERENCE} has no checkpoint at {directory}. "
            "Every metric is measured against it, so build it first."
        )
    return metrics.TwissView.from_checkpoint(directory)


def scenario_metrics(directory, reference, *, include_rf: bool = False) -> dict:
    """Every figure of merit for one checkpoint, read cold.

    `include_rf` opens the lattice as well, which is slower; the RF headroom
    is the one quantity that is not derivable from the twiss alone.
    """
    directory = Path(directory)
    view = metrics.TwissView.from_checkpoint(directory)
    meta = io_ckpt.read_meta(directory)

    row = metrics.optics_summary(view, reference)
    row.update(metrics.corrector_summary(io_ckpt.read_corrections(directory)))
    row.update(
        {
            "converged": meta.get("converged"),
            "reached_tolerance": meta.get("reached_tolerance"),
            "iterations": meta.get("iterations"),
            "wall_time_s": meta.get("wall_time_s"),
        }
    )
    # The tapering residual: how far the fixed point still moved per iteration.
    stage_info = meta.get("stage_info") or {}
    if "eloss_change_eV" in stage_info:
        row["Tapering residual [eV/iteration]"] = stage_info["eloss_change_eV"]

    if include_rf:
        from . import stages

        line = io_ckpt.read_line(directory)
        row.update(
            metrics.rf_summary(
                stages.read_rf_phase(line), float(metrics.scalar(view, "energy_loss"))
            )
        )
    return row


def summary(
    grouping: str,
    *,
    root: str | Path = "checkpoints",
    mode: str = "t",
    include_rf: bool = False,
) -> pd.DataFrame:
    """One row per implemented scenario of a grouping.

    Raises if any of them is missing: an incomplete table must not look like a
    complete one.
    """
    reference = _reference_view(root, mode)
    names = grid.implemented_scenarios(grouping)

    rows = []
    for name in names:
        scenario = grid.parse(name)
        directory = io_ckpt.checkpoint_dir(root, mode, grouping, name)
        if not io_ckpt.is_complete(
            directory, expect_corrections=scenario.glob is not None
        ):
            raise FileNotFoundError(f"{name} has no complete checkpoint at {directory}")

        n_circuits = grid.parse_grouping(grouping)
        row = {
            "mode": mode,
            "grouping": grouping,
            "n_circuits": n_circuits[0] if n_circuits else np.nan,
            "name": name,
            "vid": scenario.vid,
            "quads": scenario.quads,
            "glob": scenario.glob,
            "parent": grid.parent(name),
            "stage_applied": grid.stage_applied(name),
        }
        row.update(scenario_metrics(directory, reference, include_rf=include_rf))
        rows.append(row)

    frame = pd.DataFrame(rows)
    return _add_chain_convergence(frame)


def _add_chain_convergence(frame: pd.DataFrame) -> pd.DataFrame:
    """Whether every step that produced a row settled, not just its last one.

    A child's own flag says nothing about its ancestors, and the build is
    chained, so a scenario whose parent never settled would otherwise read as
    clean.
    """
    settled = dict(zip(frame["name"], frame["converged"]))
    frame["chain_converged"] = [
        all(settled.get(step, True) for step in grid.chain(name))
        for name in frame["name"]
    ]
    ordered = [c for c in INDEX_COLUMNS if c in frame.columns]
    ordered += [c for c in CONVERGENCE_COLUMNS if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered]
    return frame[ordered]


def scan(
    groupings,
    *,
    root: str | Path = "checkpoints",
    mode: str = "t",
    include_rf: bool = False,
) -> pd.DataFrame:
    """Every scenario of every granularity, in one tidy frame.

    Granularity is a column, not just a directory, so comparing across
    granularities is a groupby rather than a second script.
    """
    frames = [
        summary(grouping, root=root, mode=mode, include_rf=include_rf)
        for grouping in groupings
    ]
    return pd.concat(frames, ignore_index=True)


def deltas(
    grouping: str,
    *,
    root: str | Path = "checkpoints",
    mode: str = "t",
    include_rf: bool = False,
) -> pd.DataFrame:
    """Each metric replaced by its change from the scenario's parent.

    Labelled by the stage that caused the change, so this is the "what did
    adding the correction buy me" table. Roots keep NaN: they have no parent.
    """
    frame = summary(grouping, root=root, mode=mode, include_rf=include_rf)
    numeric = frame.select_dtypes(include="number").columns

    indexed = frame.set_index("name")
    changes = []
    for name in frame["name"]:
        parent = grid.parent(name)
        if parent is None or parent not in indexed.index:
            changes.append({column: np.nan for column in numeric})
        else:
            changes.append(
                {
                    column: indexed.loc[name, column] - indexed.loc[parent, column]
                    for column in numeric
                }
            )

    delta_frame = pd.DataFrame(changes, index=frame.index)
    labels = [c for c in INDEX_COLUMNS if c in frame.columns]
    return pd.concat([frame[labels], delta_frame], axis=1)


SUMMARY_PARQUET = "summary.parquet"
SUMMARY_CSV = "summary.csv"


def write_summary(
    groupings,
    *,
    root: str | Path = "checkpoints",
    mode: str = "t",
    include_rf: bool = True,
    refresh_cache: bool = True,
) -> pd.DataFrame:
    """Assemble the scan and save it beside the checkpoints.

    Writes `summary.parquet` (the canonical, machine-readable form) and
    `summary.csv` (the same table, readable without pandas) at the mode level,
    so the results outlive the session that produced them.

    With `refresh_cache`, each scenario's `metrics.json` is rewritten from its
    checkpoint as the table is built. That backfills checkpoints made before
    the cache existed, and picks up a changed metric definition without
    rebuilding any lattice.
    """
    frame = scan(groupings, root=root, mode=mode, include_rf=include_rf)

    if refresh_cache:
        skip = set(INDEX_COLUMNS)
        for _, row in frame.iterrows():
            directory = io_ckpt.checkpoint_dir(root, mode, row["grouping"], row["name"])
            io_ckpt.write_metrics(
                directory, {k: v for k, v in row.items() if k not in skip}
            )

    mode_dir = Path(root) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(mode_dir / SUMMARY_PARQUET, index=False)
    frame.to_csv(mode_dir / SUMMARY_CSV, index=False)
    return frame


def read_summary(*, root: str | Path = "checkpoints", mode: str = "t") -> pd.DataFrame:
    """The saved scan, without reopening any checkpoint."""
    path = Path(root) / mode / SUMMARY_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"no saved summary at {path}. Build the scenarios and call write_summary."
        )
    return pd.read_parquet(path)


def compare_table(frame: pd.DataFrame, rows=None, *, label="name") -> pd.DataFrame:
    """Transpose a summary for reading: metrics as rows, scenarios as columns.

    This is the shape the notebooks' comparison tables use, and the one to put
    in a document; `summary` stays the machine-readable form.
    """
    numeric = frame.select_dtypes(include="number")
    table = numeric.set_index(frame[label]).T
    if rows is not None:
        missing = [r for r in rows if r not in table.index]
        if missing:
            raise KeyError(f"no such metric(s): {missing}")
        table = table.loc[list(rows)]
    return table


def scan_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One metric across granularities: granularity down, scenario across.

    This is the granularity curve the study is about.
    """
    if metric not in frame.columns:
        raise KeyError(f"no such metric: {metric!r}")
    return frame.pivot_table(
        index=["n_circuits", "grouping"], columns="vid", values=metric, sort=True
    )
