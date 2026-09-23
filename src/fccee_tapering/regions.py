"""Splitting the ring into regions and grouping magnets into circuits.

Three things live here, all of them pure geometry over a line:

    `split_lattice_regions`  arcs, dispersion suppressors and insertions,
                             delimited by the markers the lattice carries
    `define_circuits`        which magnet is powered by which circuit
    `select_trim_dipoles`    the dipoles used as horizontal orbit correctors

Arc dipoles come in cells of three. Circuits are cut on cell boundaries, so a
cell is never split across two power supplies; the number of circuits per
sector therefore sets the granularity in whole cells rather than in magnets.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Interaction points, in ring order. Sectors run from one to the next.
IPS = ("ipa", "ipb", "ipd", "ipf", "ipg", "iph", "ipj", "ipl")

DIPOLES_PER_CELL = 3

# Magnet families that get their own circuits, and the element name prefixes
# that identify them.
QUADRUPOLE_FAMILIES = {"QF": ("qf2a", "qf3a"), "QD": ("qd1a",)}
SEXTUPOLE_FAMILIES = {"SF1": ("sf1a",), "SF2": ("sf2a",), "SD1": ("sd1a",), "SD2": ("sd2a",)}

# Quadrupole circuits per sector, per family.
QUADRUPOLE_CIRCUITS_PER_SECTOR = 2

CIRCUIT_TABLE_COLUMNS = ("circuit", "family", "sector", "element", "s_m")


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------


def split_lattice_regions(line) -> dict:
    """Group element names by region, using the lattice's own markers.

    Returns arcs, arc+DS, DS alone and insertions. A selection may wrap around
    the end of the ring, which is why the masks are built on s rather than on
    element index.
    """
    table = line.get_table()
    names = np.asarray(table.name, dtype=str)
    s = np.asarray(table.s, dtype=float)

    valid = names != "_end_point"
    names = names[valid]
    s = s[valid]

    s_by_name = dict(zip(names, s))

    def select(start_marker, end_marker):
        start = s_by_name[start_marker]
        end = s_by_name[end_marker]

        if start <= end:
            mask = (s >= start) & (s < end)
        else:
            mask = (s >= start) | (s < end)

        return names[mask]

    arcs = {}
    arc_ds = {}
    ds = {}
    insertions = {}

    for i, left_ip in enumerate(IPS):
        right_ip = IPS[(i + 1) % len(IPS)]
        sector = f"{left_ip}_{right_ip}"

        arcs[sector] = select(
            f"end_ds_start_arc_{left_ip}",
            f"end_arc_start_ds_{right_ip}",
        )

        arc_ds[sector] = select(
            f"end_straight_start_ds_{left_ip}",
            f"end_ds_start_straight_{right_ip}",
        )

        right_ds = select(
            f"end_straight_start_ds_{left_ip}",
            f"end_ds_start_arc_{left_ip}",
        )

        left_ds = select(
            f"end_arc_start_ds_{right_ip}",
            f"end_ds_start_straight_{right_ip}",
        )

        ds[sector] = np.concatenate([right_ds, left_ds])

        insertions[right_ip] = select(
            f"end_ds_start_straight_{right_ip}",
            f"end_straight_start_ds_{right_ip}",
        )

    return {"arcs": arcs, "arc_ds": arc_ds, "ds": ds, "insertions": insertions}


def insertion_elements(regions) -> np.ndarray:
    """Every element of every insertion, as one array."""
    return np.concatenate(list(regions["insertions"].values()))


# --------------------------------------------------------------------------
# Element selection
# --------------------------------------------------------------------------


def taperable_elements(line) -> np.ndarray:
    """Elements whose strength follows the local beam energy."""
    return np.asarray(
        [name for name in line.element_names if hasattr(line[name], "delta_taper")],
        dtype=str,
    )


def _family_of(name: str) -> str:
    """Strip the instance number: qf2a.15 -> qf2a."""
    return re.sub(r"\.\d+$", "", name.lower())


def _select(line, element_names, taperable, type_by_name, *, element_type=None, prefix=None):
    """Taperable elements of a region, optionally filtered by type and family."""
    selected = []
    for name in element_names:
        if name not in taperable:
            continue
        if element_type is not None and element_type not in type_by_name.get(name, ""):
            continue
        if prefix is not None and not _family_of(name).startswith(tuple(prefix)):
            continue
        selected.append(name)
    return np.asarray(selected, dtype=str)


def _type_by_name(line) -> dict:
    table = line.get_table()
    return dict(
        zip(
            np.asarray(table.name, dtype=str),
            np.asarray(table.element_type, dtype=str),
        )
    )


def _s_by_name(line) -> dict:
    table = line.get_table()
    return dict(
        zip(np.asarray(table.name, dtype=str), np.asarray(table.s, dtype=float))
    )


# --------------------------------------------------------------------------
# Dipole cells
# --------------------------------------------------------------------------


def dipole_cells(line, regions, *, dipoles_per_cell=DIPOLES_PER_CELL) -> dict:
    """Arc+DS dipoles of each sector, grouped into consecutive cells.

    Returns one ``(n_cells, dipoles_per_cell)`` array of element names per
    sector. Raises if a sector's dipole count is not a whole number of cells,
    since every downstream grouping assumes complete cells.
    """
    taperable = set(taperable_elements(line))
    type_by_name = _type_by_name(line)

    cells = {}
    for sector, elements in regions["arc_ds"].items():
        dipoles = _select(line, elements, taperable, type_by_name, element_type="Bend")

        if len(dipoles) % dipoles_per_cell != 0:
            raise ValueError(
                f"{sector} contains {len(dipoles)} dipoles, which is not a whole "
                f"number of {dipoles_per_cell}-dipole cells."
            )
        cells[sector] = dipoles.reshape(-1, dipoles_per_cell)

    return cells


# --------------------------------------------------------------------------
# Circuits
# --------------------------------------------------------------------------


def define_circuits(
    line,
    regions,
    *,
    n_dipole_circuits_per_sector: int = 1,
    dipoles_per_cell: int = DIPOLES_PER_CELL,
) -> pd.DataFrame:
    """Assign every arc magnet to a circuit.

    Dipoles: the sector's cells are split into `n_dipole_circuits_per_sector`
    consecutive groups, then flattened back to element names. Splitting cells
    rather than individual dipoles keeps the three dipoles of a cell on one
    power supply for any number of circuits; the groups then differ by at most
    one cell when the count does not divide evenly.

    The arc dipoles cannot use their original circuits because of the twin
    aperture, so new ones are defined here.

    Quadrupoles: each family split into two consecutive circuits per sector.
    Sextupoles: one circuit per family per arc.

    Returns one row per magnet, with the circuit it belongs to.
    """
    if n_dipole_circuits_per_sector < 1:
        raise ValueError(
            f"n_dipole_circuits_per_sector must be at least 1, got "
            f"{n_dipole_circuits_per_sector}"
        )

    taperable = set(taperable_elements(line))
    type_by_name = _type_by_name(line)
    s_by_name = _s_by_name(line)
    cells = dipole_cells(line, regions, dipoles_per_cell=dipoles_per_cell)

    rows = []

    def add(circuit, family, sector, elements):
        for element in elements:
            rows.append(
                {
                    "circuit": circuit,
                    "family": family,
                    "sector": sector,
                    "element": str(element),
                    "s_m": float(s_by_name[str(element)]),
                }
            )

    for sector in regions["arcs"]:
        arc_elements = regions["arcs"][sector]
        arc_ds_elements = regions["arc_ds"][sector]

        sector_cells = cells[sector]
        if n_dipole_circuits_per_sector > len(sector_cells):
            raise ValueError(
                f"{sector} has {len(sector_cells)} dipole cells, which cannot be "
                f"split into {n_dipole_circuits_per_sector} circuits."
            )
        for index, group in enumerate(
            np.array_split(sector_cells, n_dipole_circuits_per_sector)
        ):
            add(f"DIP_{index + 1}_{sector}", "DIP", sector, group.reshape(-1))

        for family, prefixes in QUADRUPOLE_FAMILIES.items():
            elements = _select(
                line,
                arc_ds_elements,
                taperable,
                type_by_name,
                element_type="Quadrupole",
                prefix=prefixes,
            )
            for index, group in enumerate(
                np.array_split(elements, QUADRUPOLE_CIRCUITS_PER_SECTOR)
            ):
                add(f"{family}_{index + 1}_{sector}", family, sector, group)

        for family, prefixes in SEXTUPOLE_FAMILIES.items():
            elements = _select(
                line,
                arc_elements,
                taperable,
                type_by_name,
                element_type="Sextupole",
                prefix=prefixes,
            )
            add(f"{family}_{sector}", family, sector, elements)

    circuits = pd.DataFrame(rows, columns=list(CIRCUIT_TABLE_COLUMNS))
    return circuits[circuits["element"].notna()].reset_index(drop=True)


def circuit_groups(circuits: pd.DataFrame, families=None) -> dict:
    """Element names per circuit, optionally restricted to some families.

    Circuit order follows first appearance along the ring, so the grouping is
    reproducible.
    """
    selected = circuits
    if families is not None:
        selected = selected[selected["family"].isin(list(families))]
    return {
        circuit: np.asarray(group["element"], dtype=str)
        for circuit, group in selected.groupby("circuit", sort=False)
    }


def circuit_summary(circuits: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit: size and span. Mirrors the notebooks' overview."""
    return (
        circuits.groupby(["circuit", "family", "sector"], sort=False)
        .agg(
            number_of_elements=("element", "size"),
            first_element=("element", "first"),
            last_element=("element", "last"),
            s_start_m=("s_m", "min"),
            s_end_m=("s_m", "max"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------
# Dipole trims
# --------------------------------------------------------------------------


def select_trim_dipoles(
    line,
    regions,
    *,
    position: int = 1,
    dipoles_per_cell: int = DIPOLES_PER_CELL,
) -> tuple[np.ndarray, pd.DataFrame]:
    """One dipole per arc+DS cell, used as a horizontal orbit corrector.

    `position` chooses which dipole of the cell carries the trim: 1 for the
    first, 2 for the second, 3 for the third. The choice is uniform across the
    ring, so the trims stay one per cell and evenly spaced whichever is used.

    Returns the trim element names and a layout table naming all the dipoles
    of every cell, so the selection can be checked by eye.
    """
    if not 1 <= position <= dipoles_per_cell:
        raise ValueError(
            f"position must be in 1..{dipoles_per_cell}, got {position}"
        )

    cells = dipole_cells(line, regions, dipoles_per_cell=dipoles_per_cell)
    s_by_name = _s_by_name(line)

    selected = []
    rows = []
    for sector, sector_cells in cells.items():
        for cell_number, cell in enumerate(sector_cells):
            trim = str(cell[position - 1])
            selected.append(trim)
            row = {
                "sector": sector,
                "cell": cell_number,
                "trim_dipole": trim,
                "s_m": float(s_by_name[trim]),
                "length_m": float(line[trim].length),
            }
            row.update(
                {f"dipole_{i + 1}": str(name) for i, name in enumerate(cell)}
            )
            rows.append(row)

    return np.asarray(selected, dtype=str), pd.DataFrame(rows)


def validate_trim_selection(
    line,
    regions,
    trim_dipoles,
    layout,
    *,
    position: int = 1,
    dipoles_per_cell: int = DIPOLES_PER_CELL,
) -> pd.DataFrame:
    """Check that exactly one dipole per cell was picked, at the right slot."""
    cells = dipole_cells(line, regions, dipoles_per_cell=dipoles_per_cell)

    expected = []
    sector_rows = []
    for sector, sector_cells in cells.items():
        expected_sector = sector_cells[:, position - 1]
        expected.extend(expected_sector)

        selected_sector = layout.loc[layout["sector"] == sector, "trim_dipole"].to_numpy(str)
        sector_rows.append(
            {
                "sector": sector,
                "dipoles_in_arc_ds": sector_cells.size,
                "cells": len(sector_cells),
                "selected_trims": len(selected_sector),
                "selection_is_correct": np.array_equal(selected_sector, expected_sector),
            }
        )

    selected = np.asarray(trim_dipoles, dtype=str)
    expected = np.asarray(expected, dtype=str)

    if len(selected) != len(np.unique(selected)):
        raise ValueError("the same dipole was selected as a trim more than once")
    if not np.array_equal(selected, expected):
        raise ValueError(
            f"the selection is not dipole {position} of every "
            f"{dipoles_per_cell}-dipole cell in each arc+DS sector"
        )

    summary = pd.DataFrame(sector_rows)
    if not summary["selection_is_correct"].all():
        raise ValueError(
            f"trim selection is wrong in sector(s) "
            f"{list(summary.loc[~summary['selection_is_correct'], 'sector'])}"
        )
    return summary
