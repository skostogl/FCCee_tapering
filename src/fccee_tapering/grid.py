"""Scenario names for the tapering-granularity grid.

A scenario name is the complete build recipe of one lattice:

    CT<nn><A|M>-Q<z|r|p>-L<0|1>-G<0|h|d>-TC<0|1>

Every field is always present, in this fixed order, hyphen-separated. Nothing
is implied by omission: a missing field is an error, not a default.

    CT<nn><A|M>  coarse dipole grouping: nn circuits per Arc, or nn Machine-wide.
                 Zero-padded so lexical sort matches numeric sort.
    Q            quadrupole tapering: r = coarse circuits, p = ideal per magnet.
                 The sextupoles always follow the coarse circuits, so this
                 field describes the quadrupoles alone.
    L            local orbit correction: 0 off, 1 on.
    G            global orbit correction: 0 none, h = H-correctors,
                 d = trim on the first dipole of each three-dipole cell.
    TC           tune and chromaticity rematch: 0 no, 1 yes.

The field order is also the build order. Every "off" value is a genuine no-op,
so switching off the rightmost field that is not already off names the state
the build passed through on its way here. That is `parent()`, and it is why the
whole grid costs one stage application per scenario rather than one full build
each.

`vid` ("v01".."v36") is a short label for crowded plot legends. It is frozen:
an old figure labelled v17 must keep meaning the same scenario forever, so the
numbering is derived from the option values and never reassigned.

This module is pure string and table manipulation. No physics, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------
# Field vocabularies
# --------------------------------------------------------------------------

QUADS = ("coarse", "ideal")
GLOBAL_OC = (None, "hcorr", "dip_trim")

_QUADS_CODE = {"coarse": "r", "ideal": "p"}
_GLOB_CODE = {None: "0", "hcorr": "h", "dip_trim": "d"}
_QUADS_FROM_CODE = {code: value for value, code in _QUADS_CODE.items()}
_GLOB_FROM_CODE = {code: value for value, code in _GLOB_CODE.items()}

# Groupings outside the scan, used as references:
#   CTind  every dipole tapered individually (the ideal machine)
#   CToff  no tapering anywhere
REFERENCE_GROUPINGS = ("CTind", "CToff")

# The scenario every reference-relative metric is measured against.
METRIC_REFERENCE = "CTind-Qp-L0-G0-TC0"
# CToff clears the tapering everywhere, so its Q field carries no meaning; it
# is present only because every name has every field.
NO_TAPERING = "CToff-Qr-L0-G0-TC0"
REFERENCE_SCENARIOS = (METRIC_REFERENCE, NO_TAPERING)

# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

STAGE_ROOT = "taper"
STAGE_LOCAL_OC = "local_oc"
STAGE_REMATCH = "rematch"
STAGE_GLOBAL_OC = {"hcorr": "global_oc_hcorr", "dip_trim": "global_oc_trim"}

# Stages that have no implementation yet. Scenarios needing one belong to the
# grid but cannot be built; `implemented_scenarios` is the subset the build and
# the analysis iterate over. Removing a stage from this set is the only change
# needed once it is written.
UNIMPLEMENTED_STAGES = frozenset({STAGE_LOCAL_OC, STAGE_REMATCH})

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_GROUPING_RE = re.compile(r"^CT(?P<n>\d{2,3})(?P<scope>[AM])$")
_NAME_RE = re.compile(
    r"^(?P<grouping>CT(?:\d{2,3}[AM]|ind|off))"
    r"-Q(?P<quads>[rp])"
    r"-L(?P<local>[01])"
    r"-G(?P<glob>[0hd])"
    r"-TC(?P<rematch>[01])$"
)

# vid blocks, one per quadrupole-tapering value. New values are appended at the
# end so existing numbers never move.
_VID_BLOCK = {"coarse": 0, "ideal": 12}
SCENARIOS_PER_GROUPING = 24


@dataclass(frozen=True)
class Scenario:
    """One point of the grid, with its name parsed into option values."""

    name: str
    grouping: str
    vid: str
    quads: str  # "off" | "coarse" | "ideal"
    local: bool
    glob: str | None  # None | "hcorr" | "dip_trim"
    rematch: bool


# --------------------------------------------------------------------------
# Grouping tokens
# --------------------------------------------------------------------------


def format_grouping(n_circuits: int, scope: str = "A") -> str:
    """Build a grouping token, e.g. ``format_grouping(9) -> "CT09A"``."""
    if scope not in ("A", "M"):
        raise ValueError(f"scope must be 'A' (per arc) or 'M' (machine-wide), got {scope!r}")
    if not 1 <= n_circuits <= 999:
        raise ValueError(f"n_circuits must be in 1..999, got {n_circuits}")
    return f"CT{n_circuits:02d}{scope}"


def parse_grouping(grouping: str) -> tuple[int, str] | None:
    """Return ``(n_circuits, scope)``, or None for a reference grouping."""
    if grouping in REFERENCE_GROUPINGS:
        return None
    match = _GROUPING_RE.match(grouping)
    if match is None:
        raise ValueError(f"not a grouping token: {grouping!r}")
    return int(match["n"]), match["scope"]


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def format_name(grouping: str, quads: str, local: bool, glob: str | None, rematch: bool) -> str:
    """Build a scenario name from option values. Validates every field."""
    parse_grouping(grouping)  # raises if the grouping token is malformed
    if quads not in QUADS:
        raise ValueError(f"quads must be one of {QUADS}, got {quads!r}")
    if glob not in GLOBAL_OC:
        raise ValueError(f"glob must be one of {GLOBAL_OC}, got {glob!r}")
    return (
        f"{grouping}"
        f"-Q{_QUADS_CODE[quads]}"
        f"-L{int(bool(local))}"
        f"-G{_GLOB_CODE[glob]}"
        f"-TC{int(bool(rematch))}"
    )


def parse(name: str) -> Scenario:
    """Parse a scenario name. Raises on any malformed or missing field."""
    match = _NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"not a scenario name: {name!r}. Expected "
            "CT<nn><A|M>-Q<z|r|p>-L<0|1>-G<0|h|d>-TC<0|1>"
        )
    quads = _QUADS_FROM_CODE[match["quads"]]
    local = match["local"] == "1"
    glob = _GLOB_FROM_CODE[match["glob"]]
    rematch = match["rematch"] == "1"
    return Scenario(
        name=name,
        grouping=match["grouping"],
        vid=vid(quads, local, glob, rematch),
        quads=quads,
        local=local,
        glob=glob,
        rematch=rematch,
    )


def vid(quads: str, local: bool, glob: str | None, rematch: bool) -> str:
    """Frozen short label for a set of option values."""
    number = (
        _VID_BLOCK[quads]
        + (6 if local else 0)
        + 2 * GLOBAL_OC.index(glob)
        + (1 if rematch else 0)
        + 1
    )
    return f"v{number:02d}"


# --------------------------------------------------------------------------
# The build DAG
# --------------------------------------------------------------------------


def parent(name: str) -> str | None:
    """Name of the state this scenario is built from, or None if it is a root.

    Obtained by switching off the rightmost field that is not already off, so
    the parent is always itself a scenario of the grid.
    """
    scenario = parse(name)
    if scenario.rematch:
        return format_name(scenario.grouping, scenario.quads, scenario.local, scenario.glob, False)
    if scenario.glob is not None:
        return format_name(scenario.grouping, scenario.quads, scenario.local, None, False)
    if scenario.local:
        return format_name(scenario.grouping, scenario.quads, False, None, False)
    return None


def chain(name: str) -> list[str]:
    """Ancestors of a scenario, root first, ending with the scenario itself."""
    names = []
    current: str | None = name
    while current is not None:
        names.append(current)
        current = parent(current)
    return list(reversed(names))


def stage_applied(name: str) -> str:
    """The single stage that turns this scenario's parent into this scenario."""
    scenario = parse(name)
    if scenario.rematch:
        return STAGE_REMATCH
    if scenario.glob is not None:
        return STAGE_GLOBAL_OC[scenario.glob]
    if scenario.local:
        return STAGE_LOCAL_OC
    return STAGE_ROOT


def is_implemented(name: str) -> bool:
    """True when every stage needed to build this scenario exists."""
    return all(stage_applied(step) not in UNIMPLEMENTED_STAGES for step in chain(name))


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------


def all_scenarios(grouping: str) -> list[str]:
    """Every scenario of one grouping, parents strictly before children."""
    names = [
        format_name(grouping, quads, local, glob, rematch)
        for quads in QUADS
        for local in (False, True)
        for glob in GLOBAL_OC
        for rematch in (False, True)
    ]
    return sorted(names, key=lambda name: int(parse(name).vid[1:]))


def implemented_scenarios(grouping: str) -> list[str]:
    """The subset of `all_scenarios` that can actually be built today."""
    return [name for name in all_scenarios(grouping) if is_implemented(name)]


def registry(grouping: str) -> pd.DataFrame:
    """One row per scenario of a grouping, in build order.

    This is the table the build and the analysis both iterate. Iterating it
    rather than a directory listing is what makes a missing checkpoint an
    error instead of a silently shorter table.
    """
    rows = []
    for name in all_scenarios(grouping):
        scenario = parse(name)
        rows.append(
            {
                "name": scenario.name,
                "vid": scenario.vid,
                "grouping": scenario.grouping,
                "quads": scenario.quads,
                "local": scenario.local,
                "glob": scenario.glob,
                "rematch": scenario.rematch,
                "parent": parent(name),
                "stage_applied": stage_applied(name),
                "implemented": is_implemented(name),
            }
        )
    return pd.DataFrame(rows)
