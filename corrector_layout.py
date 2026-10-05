"""Give every quadrupole that has a BPM the correctors its neighbours have.

As delivered, the FCC-ee t lattice leaves 424 arc quadrupoles -- the `qf3a` and
`qd1am` families -- with a beam position monitor but no orbit corrector in either
plane, while the `qf2a` / `qd1a` magnets beside them, identical in length and
gradient, carry both. This is coming from previous model where the correctors where 
suppose to be embedded inside the sextupoles. In the current design, the correctors 
should be next to every quads.

Usage, right after the line is loaded and before any correction::

    from corrector_layout import add_missing_correctors

    added = add_missing_correctors(line)
    print(added.groupby(["plane", "family"]).size())

The correctors are zero-length `Multipole` elements named `hcor_<host>` and
`vcor_<host>`, inserted immediately after the host's BPM so that monitor and
correctors share a position, which is how the already-equipped quadrupoles are
arranged. They are created at zero strength, so the optics before any correction
is bit-for-bit what it was.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xtrack as xt

__all__ = ["missing_correctors", "add_missing_correctors", "correctors_at_monitors"]

PLANE_PREFIX = {"h": "hcor", "v": "vcor"}


def _host_of(name: str) -> str:
    """The magnet a bpm/hcor/vcor belongs to, from its name."""
    return re.sub(r"^(bpm|hcor|vcor)_", "", str(name))


def _family_of(name: str) -> str:
    """The magnet family, i.e. the name without its trailing index."""
    return re.sub(r"\.\d+$", "", str(name))


def correctors_at_monitors(line, planes=("h", "v")):
    """The correctors that sit on a magnet carrying a BPM, in order of ``s``.

    Once the layout is complete there are 2676 correctors per plane against 2636
    monitors: the 40 extra ones are the final-focus units, `qd0b/c` and
    `qf1b/c/d` on each side of the four IPs, where no BPM is installed. They are
    useless to a *global* orbit correction anyway -- beta reaches 500-1150 m
    there, so the betatron phase barely advances across the whole string and
    their response columns are duplicates of each other up to a factor sqrt(beta)
    -- and on the delivered lattice they are exactly the 40 null directions that
    made the response matrix rank 2212 instead of 2252.

    Passing the result to the correction leaves them in the lattice, powered and
    available for local IR bumps; it only keeps them out of the knob list, so the
    correction solves a square 2636 x 2636 problem rather than an over-specified
    one with 40 degenerate columns.

    Returns ``{plane: numpy array of names}``.
    """
    table = line.get_table()
    names = np.asarray(table.name, dtype=str)
    positions = dict(zip(names, np.asarray(table.s, dtype=float)))
    monitored = {_host_of(n) for n in names if n.startswith("bpm_")}

    selected = {}
    for plane in planes:
        prefix = PLANE_PREFIX[plane]
        keep = [n for n in names
                if n.startswith(f"{prefix}_") and _host_of(n) in monitored]
        keep.sort(key=lambda n: positions[n])
        selected[plane] = np.asarray(keep, dtype=str)
    return selected


def missing_correctors(line, only_families=None) -> pd.DataFrame:
    """Quadrupoles that carry a BPM but are missing a corrector.

    Returns one row per (host, plane) that has no corrector, with the host's
    position and family. Nothing is modified.

    ``only_families`` restricts the search, e.g. ``("qf3a", "qd1am")``; by
    default every quadrupole with a monitor is considered, which on the
    delivered lattice finds exactly those two families.
    """
    table = line.get_table()
    names = np.asarray(table.name, dtype=str)
    positions = np.asarray(table.s, dtype=float)
    types = np.asarray(table.element_type, dtype=str)

    quadrupole_s = {
        name: s for name, s, kind in zip(names, positions, types) if kind == "Quadrupole"
    }
    monitored = {_host_of(n) for n in names if n.startswith("bpm_")}
    equipped = {
        plane: {_host_of(n) for n in names if n.startswith(f"{prefix}_")}
        for plane, prefix in PLANE_PREFIX.items()
    }

    rows = []
    for host, s in quadrupole_s.items():
        if host not in monitored:
            continue
        family = _family_of(host)
        if only_families is not None and family not in set(only_families):
            continue
        for plane in PLANE_PREFIX:
            if host not in equipped[plane]:
                rows.append({"host": host, "family": family, "plane": plane, "s_m": s})

    frame = pd.DataFrame(rows, columns=["host", "family", "plane", "s_m"])
    return frame.sort_values(["s_m", "plane"], ignore_index=True)


def add_missing_correctors(
    line,
    only_families=None,
    planes=("h", "v"),
    verbose=True,
    check_optics=False,
):
    """Insert the missing correctors in place, and return what was added.

    Each corrector is a zero-strength, zero-length ``Multipole`` placed directly
    after the host's BPM, so a monitor and its correctors sit at one position as
    they do at the already-equipped quadrupoles. Inserting at zero strength
    leaves the optics unchanged; ``check_optics=True`` twisses before and after
    and raises if the tunes move, which costs two twisses.

    The function is idempotent: a second call finds nothing to do and returns an
    empty frame.
    """
    wanted = missing_correctors(line, only_families=only_families)
    wanted = wanted[wanted["plane"].isin(planes)].reset_index(drop=True)
    if wanted.empty:
        if verbose:
            print("corrector layout already complete, nothing inserted")
        return wanted

    if check_optics:
        before = line.twiss(eneloss_and_damping=True)

    # Insert from the end of the ring backwards: every insertion shifts the
    # indices after it, so working downstream-first keeps the ones not yet used
    # valid without recomputing the table each time.
    element_names = list(np.asarray(line.get_table().name, dtype=str))
    index_of = {name: i for i, name in enumerate(element_names)}
    added = []
    for host, group in sorted(
        wanted.groupby("host"), key=lambda kv: -index_of[f"bpm_{kv[0]}"]
    ):
        anchor = index_of[f"bpm_{host}"]
        # h first, then v, matching the order used elsewhere in the lattice
        for offset, plane in enumerate(p for p in ("h", "v") if p in set(group["plane"])):
            name = f"{PLANE_PREFIX[plane]}_{host}"
            if name in line.element_dict:
                raise ValueError(f"{name} already exists; refusing to duplicate it")
            line.insert_element(
                name=name,
                element=xt.Multipole(knl=[0.0], ksl=[0.0], length=0.0),
                index=anchor + 1 + offset,
            )
            added.append({"name": name, "host": host, "family": _family_of(host),
                          "plane": plane})

    frame = pd.DataFrame(added)
    frame = frame.merge(wanted[["host", "plane", "s_m"]], on=["host", "plane"], how="left")
    frame = frame.sort_values("s_m", ignore_index=True)

    if check_optics:
        after = line.twiss(eneloss_and_damping=True)
        for plane_name, b, a in (("qx", before.qx, after.qx), ("qy", before.qy, after.qy)):
            if abs(a - b) > 1e-9:
                raise RuntimeError(
                    f"inserting the correctors moved {plane_name} by {a - b:.3e}; "
                    "they should be zero-strength and massless"
                )

    if verbose:
        counts = frame.groupby(["plane", "family"]).size()
        print(f"inserted {len(frame)} correctors at {frame['host'].nunique()} quadrupoles")
        for (plane, family), n in counts.items():
            print(f"   {PLANE_PREFIX[plane]:5s} on {family:8s}: {n}")
        table = line.get_table()
        names = np.asarray(table.name, dtype=str)
        for plane, prefix in PLANE_PREFIX.items():
            total = sum(1 for n in names if n.startswith(f"{prefix}_"))
            print(f"   {prefix} correctors now in the ring: {total}")
        print(f"   BPMs: {sum(1 for n in names if n.startswith('bpm_'))}")

    return frame
