"""Reading and writing scenario checkpoints.

One directory per scenario, ordered by how expensive each file is to regenerate:

    <root>/<mode>/<grouping>/
        circuits.parquet     which magnet belongs to which circuit
        <name>/
            fccee_<mode>_<name>.json.gz   the lattice          expensive
            twiss.parquet        moderate    per-element twiss columns
            twiss.json           moderate    everything in the twiss that is not per-element
            corrections.parquet  moderate    applied corrector and trim strengths
            meta.json            cheap       build provenance
            metrics.json         cheap       derived figures of merit, a regenerable cache

`line.json.gz` is the authoritative artifact: because it is kept, any twiss
quantity left out below can always be recovered by re-twissing. That is what
makes an explicit column allowlist safe rather than a gamble.

`corrections.parquet` is the exception -- it is *not* recoverable. A dipole
trim is applied by overwriting ``knl[0]``, which destroys the value it
replaced, so the applied kick exists only if it is written down here.

This module knows nothing about the physics; it moves tables to and from disk.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Lattice files follow the naming of the source lattices, fccee_<mode>_<what>,
# so a file copied out of the tree and sent to someone still says which energy
# mode and which tapering model it is.
LINE_PREFIX = "fccee"
LINE_SUFFIX = ".json.gz"
TWISS_PARQUET = "twiss.parquet"
TWISS_JSON = "twiss.json"
CORRECTIONS_PARQUET = "corrections.parquet"
META_JSON = "meta.json"
METRICS_JSON = "metrics.json"

# Shared by every scenario of one grouping, so it sits one level up.
CIRCUITS_PARQUET = "circuits.parquet"

PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "zstd"

# --------------------------------------------------------------------------
# Twiss column allowlists
# --------------------------------------------------------------------------

# Per-element columns kept in twiss.parquet. Requesting a column that the
# table does not have is an error: a typo must not become a column that
# quietly goes missing.
#
# This is a small subset of the ~86 columns a twiss carries -- just what the
# figures of merit and the standard plots read, plus mux/muy, which the trim
# comparison needs to say where a corrector sits in phase. The rest (alfx/alfy,
# px/py, dpx/dpy, zeta, ptau, ...) are left out because they dominate the file
# size and nothing reads them; any of them can be recovered by re-twissing the
# stored line.
TWISS_COLUMNS = (
    "name",
    "s",
    "length",
    "x",
    "y",
    "delta",
    "betx",
    "bety",
    "dx",
    "dy",
    "mux", 
    "muy",
    "angle",
    "radiation_flag",
)

# Everything that is not per-element goes to twiss.json. These are kept in
# full rather than curated: together they are a few hundred bytes, and they
# include the small fixed-size arrays (R-matrix, damping constants, partition
# numbers, equilibrium emittances) that the summary table needs.
TWISS_SCALARS = (
    "R_matrix",
    "beta0",
    "bets0",
    "c_minus",
    "c_minus_im_0",
    "c_minus_re_0",
    "damping_constants_s",
    "damping_constants_turns",
    "ddqx",
    "ddqy",
    "dqx",
    "dqy",
    "energy_loss",
    "eq_gemitt_x",
    "eq_gemitt_y",
    "eq_gemitt_zeta",
    "eq_nemitt_x",
    "eq_nemitt_y",
    "eq_nemitt_zeta",
    "gamma0",
    "line_length",
    "method",
    "momentum_compaction_factor",
    "only_markers",
    "p0c",
    "partition_numbers",
    "periodic",
    "qs",
    "qx",
    "qy",
    "radiation_method",
    "reference_frame",
    "rotation_matrix",
    "slip_factor",
    "slip_factor_dzeta_ddelta",
    "t_rev0",
    "values_at",
)

# Keys deliberately not stored, with the reason.
DROPPED_TWISS_KEYS = {
    "W_matrix": "per-element 6x6, ~1e6 floats; recoverable by re-twissing",
    "R_matrix_ebe": "per-element 6x6; recoverable by re-twissing",
    "n_dot_delta_kick_sq_ave": "per-element radiation integrand; the equilibrium "
    "emittances derived from it are stored instead",
    "dl_radiation": "per-element radiation integrand; recoverable by re-twissing",
    "particle_on_co": "Particles object, not serialisable as a table",
    "line_config": "build configuration, recorded in meta.json instead",
    "eigenvalues": "complex; recoverable from R_matrix",
    "steps_R_matrix": "twiss solver internals",
    # Deprecated aliases. The name each one points at is stored instead, so
    # reading these would only duplicate a value under an outdated key.
    "steps_r_matrix": "deprecated alias of steps_R_matrix",
    "kin_xprime": "deprecated alias of kin_xp",
    "kin_yprime": "deprecated alias of kin_yp",
    "T_rev0": "deprecated alias of t_rev0",
    "circumference": "deprecated alias of line_length",
    "eneloss_turn": "deprecated alias of energy_loss",
    "slip_factor_dz_ddelta": "deprecated alias of slip_factor_dzeta_ddelta",
    "angle_rad": "deprecated alias of angle",
    "completed_init": "twiss solver internals",
    "_action": "twiss solver internals",
    "_orientation": "twiss solver internals",
}

# --------------------------------------------------------------------------
# Corrections table
# --------------------------------------------------------------------------

# Applied strengths, one row per corrector or trim the stage touched, whether
# or not its kick ended up non-zero. Rows are kept for both planes so that
# "corrected to zero" stays distinguishable from "never corrected".
CORRECTION_COLUMNS = (
    "family",  # "hcor" | "vcor" | "dip_trim"
    "plane",  # "x" | "y"
    "name",
    "s_m",
    "length_m",  # 0.0 for thin correctors
    "kick_rad",
    "bl_mtm",  # integrated field, kick_rad * Brho * 1e3
    "b_mt",  # bl_mtm / length_m, NaN where the element is thin
)

# --------------------------------------------------------------------------
# Circuits
# --------------------------------------------------------------------------

# Which magnet belongs to which circuit: the model under study. The line
# records the tapering value each magnet ended up with, but not the grouping
# that produced it -- magnets sharing a circuit merely happen to share a
# value. Stored once per grouping, since every scenario of a grouping is
# powered by the same circuits.
CIRCUIT_COLUMNS = (
    "circuit",  # e.g. "DIP_3_ipa_ipb"
    "family",  # "DIP" | "QF" | "QD" | "SF1" | "SF2" | "SD1" | "SD2"
    "sector",  # e.g. "ipa_ipb"
    "element",
    "s_m",
)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def grouping_dir(root: str | Path, mode: str, grouping: str) -> Path:
    """Directory holding everything shared by one grouping's scenarios."""
    return Path(root) / mode / grouping


def checkpoint_dir(root: str | Path, mode: str, grouping: str, name: str) -> Path:
    """Directory holding one scenario's checkpoint."""
    return grouping_dir(root, mode, grouping) / name


# --------------------------------------------------------------------------
# Stamping
# --------------------------------------------------------------------------


def _digest(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, default=_jsonable)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def stamp(parent_stamp: str | None, stage: str, params: dict, code_version: str) -> str:
    """Hash identifying how a lattice was built.

    Includes the parent's stamp, so a change anywhere upstream propagates and
    every checkpoint below it rebuilds. Includes the code version, so fixing a
    stage does not leave downstream checkpoints built with the old one.
    """
    return _digest(
        {
            "parent": parent_stamp,
            "stage": stage,
            "params": params,
            "code": code_version,
        }
    )


def twiss_id(line_stamp: str, twiss_config: dict) -> str:
    """Hash identifying one twiss of one lattice.

    Written into both twiss.parquet and twiss.json so a half-updated pair is
    detected instead of silently mixing vectors from one twiss with scalars
    from another.
    """
    return _digest({"line": line_stamp, "twiss": twiss_config})


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------


def _jsonable(value):
    """Convert numpy types to plain Python for json.dump."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Twiss
# --------------------------------------------------------------------------


def split_twiss(tw, columns=TWISS_COLUMNS, scalars=TWISS_SCALARS):
    """Split a twiss table into a per-element frame and a scalar dict.

    Per-element quantities that are not plain 1-D columns are left out: the
    W-matrices and the radiation integrands are large and always recoverable
    by re-twissing.

    Raises if a requested per-element column is absent, or if a key pinned as
    a scalar turns out to be per-element. Missing scalars are reported in the
    returned dict instead, since which ones a twiss carries depends on the
    options it was called with.
    """
    n_elements = len(tw["name"])

    missing = [column for column in columns if column not in tw.keys()]
    if missing:
        raise KeyError(
            f"twiss table has no column(s) {missing}. Available per-element "
            f"columns: {sorted(k for k in tw.keys() if _is_column(tw[k], n_elements))}"
        )

    wrong_length = [
        column for column in columns if not _is_column(tw[column], n_elements)
    ]
    if wrong_length:
        raise ValueError(
            f"requested per-element column(s) {wrong_length} are not per-element "
            f"in this table (expected length {n_elements})"
        )

    vectors = pd.DataFrame({column: np.asarray(tw[column]) for column in columns})

    values: dict = {}
    absent: list[str] = []
    for key in scalars:
        if key not in tw.keys():
            absent.append(key)
            continue
        value = tw[key]
        if _is_per_element(value, n_elements):
            raise ValueError(
                f"{key!r} is pinned as a scalar but is per-element in this table "
                f"(shape {np.asarray(value).shape}, {n_elements} elements). Storing "
                "it would put a full-length array in twiss.json; move it to "
                "TWISS_COLUMNS or DROPPED_TWISS_KEYS."
            )
        values[key] = _scalar_value(value)

    # Only keys no allowlist mentions are inspected, so deprecated aliases are
    # never touched.
    covered = set(columns) | set(scalars) | set(DROPPED_TWISS_KEYS)
    unexpected = sorted(
        key
        for key in tw.keys()
        if key not in covered and not _is_per_element(tw[key], n_elements)
    )
    if unexpected:
        warnings.warn(
            f"twiss keys not covered by any allowlist and therefore not stored: "
            f"{unexpected}. Add them to TWISS_SCALARS or DROPPED_TWISS_KEYS.",
            RuntimeWarning,
            stacklevel=2,
        )

    if absent:
        values["_missing_scalars"] = absent
    return vectors, values


def _is_column(value, n_elements: int) -> bool:
    """True for a plain 1-D per-element array, the only shape parquet can take."""
    array = np.asarray(value)
    return array.ndim == 1 and array.shape[0] == n_elements


def _is_per_element(value, n_elements: int) -> bool:
    """True for anything carrying one entry per element, whatever its shape.

    Some radiation quantities are defined between elements and so are one
    entry shorter than the table; they are per-element all the same.
    """
    array = np.asarray(value)
    return array.ndim >= 1 and array.shape[0] in (n_elements, n_elements - 1)


def _scalar_value(value):
    """Normalise a non-per-element twiss entry to something JSON can hold."""
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    array = np.asarray(value)
    if array.dtype.kind not in "fiubUS":
        return str(value)
    return array.item() if array.ndim == 0 else array.tolist()


def write_twiss(directory: str | Path, tw, *, ident: str, extra: dict | None = None) -> None:
    """Write the twiss parquet/json pair.

    Both files are written under temporary names and renamed at the end, and
    both carry the same `ident`, so a half-written pair is detectable rather
    than producing vectors and scalars that quietly disagree.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    vectors, values = split_twiss(tw)
    payload = {
        "twiss_id": ident,
        "columns_requested": list(TWISS_COLUMNS),
        "xsuite_version": _xsuite_version(),
        "values": values,
    }
    if extra:
        payload.update(extra)

    parquet_tmp = directory / (TWISS_PARQUET + ".tmp")
    json_tmp = directory / (TWISS_JSON + ".tmp")
    vectors_out = vectors.copy()
    vectors_out["twiss_id"] = ident
    vectors_out.to_parquet(
        parquet_tmp, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False
    )
    _write_json(json_tmp, payload)

    os.replace(parquet_tmp, directory / TWISS_PARQUET)
    os.replace(json_tmp, directory / TWISS_JSON)


def read_twiss(directory: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read the twiss pair, checking that both halves came from one twiss."""
    directory = Path(directory)
    vectors = pd.read_parquet(directory / TWISS_PARQUET, engine=PARQUET_ENGINE)
    payload = _read_json(directory / TWISS_JSON)

    parquet_ids = set(vectors["twiss_id"].unique()) if "twiss_id" in vectors else set()
    if parquet_ids != {payload["twiss_id"]}:
        raise ValueError(
            f"{directory}: twiss.parquet carries twiss_id {parquet_ids} but "
            f"twiss.json carries {payload['twiss_id']!r}. The pair is inconsistent; "
            "rebuild the checkpoint."
        )
    return vectors.drop(columns="twiss_id"), payload


# --------------------------------------------------------------------------
# Corrections, meta, metrics
# --------------------------------------------------------------------------


def write_corrections(directory: str | Path, corrections: pd.DataFrame) -> None:
    """Write the applied corrector and trim strengths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    missing = [column for column in CORRECTION_COLUMNS if column not in corrections.columns]
    if missing:
        raise KeyError(f"corrections table is missing column(s) {missing}")

    tmp = directory / (CORRECTIONS_PARQUET + ".tmp")
    corrections[list(CORRECTION_COLUMNS)].to_parquet(
        tmp, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False
    )
    os.replace(tmp, directory / CORRECTIONS_PARQUET)


def read_corrections(directory: str | Path) -> pd.DataFrame:
    """Read the applied strengths, or an empty table when the stage applied none."""
    path = Path(directory) / CORRECTIONS_PARQUET
    if not path.exists():
        return pd.DataFrame(columns=list(CORRECTION_COLUMNS))
    return pd.read_parquet(path, engine=PARQUET_ENGINE)


def write_circuits(directory: str | Path, circuits: pd.DataFrame) -> None:
    """Write the circuit definition of one grouping."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    missing = [column for column in CIRCUIT_COLUMNS if column not in circuits.columns]
    if missing:
        raise KeyError(f"circuit table is missing column(s) {missing}")
    duplicated = circuits["element"].duplicated()
    if duplicated.any():
        raise ValueError(
            "an element appears in more than one circuit: "
            f"{sorted(circuits.loc[duplicated, 'element'].unique())[:5]}"
        )

    tmp = directory / (CIRCUITS_PARQUET + ".tmp")
    circuits[list(CIRCUIT_COLUMNS)].to_parquet(
        tmp, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False
    )
    os.replace(tmp, directory / CIRCUITS_PARQUET)


def read_circuits(directory: str | Path) -> pd.DataFrame:
    """Read the circuit definition of one grouping."""
    return pd.read_parquet(Path(directory) / CIRCUITS_PARQUET, engine=PARQUET_ENGINE)


def circuits_digest(circuits: pd.DataFrame) -> str:
    """Hash of a circuit definition, for inclusion in a checkpoint's stamp.

    Regrouping the magnets changes the machine, so it has to invalidate every
    checkpoint built from the old grouping.
    """
    ordered = circuits.sort_values("element")[["circuit", "element"]]
    return _digest(ordered.to_dict("records"))


def write_meta(directory: str | Path, meta: dict) -> None:
    """Write build provenance. Twiss results do not belong here."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / META_JSON, meta)


def read_meta(directory: str | Path) -> dict:
    return _read_json(Path(directory) / META_JSON)


def write_metrics(directory: str | Path, metrics: dict) -> None:
    """Write the derived figures of merit. This file is a regenerable cache."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / METRICS_JSON, metrics)


def read_metrics(directory: str | Path) -> dict:
    return _read_json(Path(directory) / METRICS_JSON)


# --------------------------------------------------------------------------
# Line
# --------------------------------------------------------------------------


def line_filename(name: str, mode: str) -> str:
    """File a scenario's lattice is stored under.

    ``fccee_t_CT09A-Qr-L0-G0-TC0.json.gz``: the source-lattice prefix, the
    energy mode, then the scenario. Scenario fields are hyphen-separated, so
    they never collide with the underscores that separate the parts here.
    """
    return f"{LINE_PREFIX}_{mode}_{name}{LINE_SUFFIX}"


def parse_line_filename(filename: str | Path) -> tuple[str, str]:
    """Recover ``(mode, scenario)`` from a lattice file name."""
    stem = Path(filename).name
    if not stem.startswith(f"{LINE_PREFIX}_") or not stem.endswith(LINE_SUFFIX):
        raise ValueError(
            f"not a lattice file name: {stem!r}. Expected "
            f"{LINE_PREFIX}_<mode>_<scenario>{LINE_SUFFIX}"
        )
    body = stem[len(LINE_PREFIX) + 1 : -len(LINE_SUFFIX)]
    mode, _, scenario = body.partition("_")
    if not mode or not scenario:
        raise ValueError(f"not a lattice file name: {stem!r}")
    return mode, scenario


def find_line(directory: str | Path) -> Path:
    """The one lattice file in a checkpoint directory."""
    directory = Path(directory)
    candidates = sorted(directory.glob(f"*{LINE_SUFFIX}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one {LINE_SUFFIX} file in {directory}, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def write_line(
    directory: str | Path, line, *, name: str, mode: str, metadata: dict | None = None
) -> None:
    """Serialise the lattice under its scenario name.

    Provenance is stamped into ``line.metadata``, which survives the round
    trip and sits at the top level of the JSON. A line that gets renamed or
    passed on still carries the scenario it belongs to.

    Reload it with::

        line = xt.Line.from_json("fccee_t_<name>.json.gz")
        line.build_tracker()
        tw = line.twiss(radiation_analysis=True)

    The tapering, the RF phase and the radiation settings are all part of the
    saved lattice, so the twiss reproduces the stored results directly.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    line.metadata.update({**(metadata or {}), "scenario": name, "mode": mode})

    target = directory / line_filename(name, mode)
    tmp = directory / f".{name}.tmp{LINE_SUFFIX}"
    line.to_json(tmp)
    os.replace(tmp, target)


def read_line(directory: str | Path):
    """Load the lattice of a checkpoint. Needs `build_tracker()` before use."""
    import xtrack as xt

    return xt.Line.from_json(find_line(directory))


def read_line_metadata(path: str | Path) -> dict:
    """Provenance of a lattice file, read without xsuite or a tracker."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle).get("metadata", {})


# --------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------


def is_complete(directory: str | Path, *, expect_corrections: bool = False) -> bool:
    """True when every file a finished checkpoint must have is present."""
    directory = Path(directory)
    required = [TWISS_PARQUET, TWISS_JSON, META_JSON]
    if expect_corrections:
        required.append(CORRECTIONS_PARQUET)
    if not all((directory / filename).exists() for filename in required):
        return False
    return len(list(directory.glob(f"*{LINE_SUFFIX}"))) == 1


def is_valid(directory: str | Path, expected_stamp: str, *, expect_corrections: bool = False) -> bool:
    """True when the checkpoint is complete and was built with this stamp."""
    directory = Path(directory)
    if not is_complete(directory, expect_corrections=expect_corrections):
        return False
    try:
        return read_meta(directory).get("stamp") == expected_stamp
    except (OSError, json.JSONDecodeError):
        return False


def _xsuite_version() -> str:
    try:
        import xtrack as xt

        return f"xtrack {xt.__version__}"
    except ImportError:  # tests run without the physics stack
        return "unknown"
