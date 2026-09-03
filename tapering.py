"""
Custom tapering / energy-loss compensation.

Two entry points, sharing one RF-phase solver:

``compensate_energy_loss``
    Replacement for xsuite's built-in ``Line.compensate_radiation_energy_loss``
    that additionally supports *group* tapering: magnets can be tapered by the
    mean of their group, or excluded from tapering entirely. Solves the magnet
    tapering and the RF phase together.

``get_rf_phase``
    Sets only the RF phase, for the machine *as it is currently tapered*. Use
    it after installing a coarse/grouped tapering model by hand. Unlike xsuite's
    own routine it never enables ``XS_FLAG_SR_TAPER``, so the stored
    ``delta_taper`` is honoured, and the closed orbit is searched on the real
    radiating lattice rather than an untapered stand-in.

Ported to xtrack 0.111.x (originally written against 0.99.6).
"""

import warnings

import numpy as np
import xtrack as xt
from typing import Callable, Optional

from tqdm import tqdm
from scipy.constants import c as clight
from xtrack.twiss import ClosedOrbitSearchError


# --------------------------------------------------------------------------
# Shared RF helpers
# --------------------------------------------------------------------------

def _rf_setters(line):
    """Multisetters and the cavity mask, gathered once."""
    c = line.attr._cache
    return {
        'v':           c['_own_voltage'].multisetter,
        'f':           c['_own_frequency'].multisetter,
        'lag':         c['_own_lag'].multisetter,
        'phase':       c['_own_phase'].multisetter,
        'phase_taper': c['_own_phase_taper'].multisetter,
        'lag_taper':   c['_own_lag_taper'].multisetter,
        'mask_cav':    c['_own_voltage'].mask,
    }


def _clear_lag_taper(rf):
    """
    The RF kernel applies BOTH corrections additively:
        sin(phase0 + DEG2RAD*(lag + lag_taper) + (phase + phase_taper) - ...)
    We drive phase_taper, so any lag_taper left over from a previous run or from
    xsuite's own compensate_radiation_energy_loss would be silently added on top.
    Zero it so phase_taper is the single source of truth. (xsuite itself marks
    lag_taper as legacy.)
    """
    rf['lag_taper'].set_values(np.zeros_like(rf['lag_taper'].get_values()))


def _base_phase(rf):
    """
    Untapered phase of each cavity, in radians.

    The kernel builds the RF argument as ``DEG2RAD*lag + phase``, so the base the
    taper works against is the sum of both -- read live rather than cached, since
    matching may move them. Lattices differ in where they put it (LCC_107.0.1
    uses lag=0, phase=pi), which is why this is derived instead of assumed.
    """
    return np.deg2rad(rf['lag'].get_values()) + rf['phase'].get_values()


def _check_rf_headroom(ratio, context, rf_warn_ratio):
    """
    Guard the arcsin argument.

    The voltage is fixed, so a cavity supplies at most q*V0, at crest.
    sin(phase_s) = (energy to replace)/(q*V0); |ratio| > 1 means no synchronous
    phase exists at all -- numpy would otherwise return NaN silently.
    """
    r = float(np.max(np.abs(np.atleast_1d(ratio))))
    if r >= 1.0:
        raise ValueError(
            f"No synchronous RF phase exists ({context}): the energy to replace "
            f"per turn is {r:.4f}x what the installed voltage can supply at crest. "
            "Either the tapering is too coarse or the lattice needs more RF voltage.")
    if r > rf_warn_ratio:
        warnings.warn(
            f"RF near its limit ({context}): sin(phase_s) = {r:.4f}, "
            f"phase_s = {np.rad2deg(np.arcsin(r)):.2f} deg, "
            f"{100*(1.0-r):.1f}% headroom to crest.",
            RuntimeWarning, stacklevel=2)
    return ratio


def _solve_rf_phase(line, trco, rf, rf_warn_ratio, context):
    """
    Set phase_taper so the cavities replace exactly the energy lost over one turn.

    ``trco`` is a ONE_TURN_EBE record of a particle launched on the closed orbit.
    Everything is in radians: the target is the total phase the cavity must sit
    at, and phase_taper supplies whatever the lattice's own base phase does not.
    """
    v0 = rf['v'].get_values()
    f0 = rf['f'].get_values()

    zeta_at_cav = np.atleast_1d(np.squeeze(trco.zeta[0, :-1]))[rf['mask_cav']]
    zeta_at_cav -= np.mean(zeta_at_cav)

    dptau = np.diff(trco.ptau[0])
    egain = dptau[dptau > 1e-10] * line.particle_ref.p0c

    inst_phase = np.arcsin(_check_rf_headroom(egain / v0, context, rf_warn_ratio))
    target_phase = np.pi - inst_phase + 2*np.pi * f0 * zeta_at_cav / clight
    rf['phase_taper'].set_values(target_phase - _base_phase(rf))
    return target_phase


def _seed_phase_taper(line, rf, rf_warn_ratio, n_iter=3, verbose=False):
    """
    First guess for phase_taper, for machines where no closed orbit exists yet.

    At ttbar the cavities sit at phase = pi (zero crossing) by default: they
    replace no energy, delta sags several percent over the turn and the closed
    orbit search fails before any tapering can be computed. The synchronous
    phase, pi - arcsin(U0/sum(V)), is enough to close the ring on its own --
    no magnet tapering required.

    U0 is measured by tracking one turn from particle_ref, which needs no closed
    orbit. It is a fixed point (the orbit excursion feeds back into the
    radiation), so the measurement is repeated a few times.
    """
    v0 = rf['v'].get_values()
    p0c = float(np.asarray(line.particle_ref.p0c).ravel()[0])
    phi_s = None

    for it in range(n_iter):
        p = line.particle_ref.copy()
        line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
        if int(np.asarray(p.state).ravel()[0]) < 1:
            raise RuntimeError(
                "Particle lost while measuring the energy loss, cannot seed the RF "
                "phase. Check that radiation is configured and the lattice is sane.")

        dptau = np.diff(line.record_last_track.ptau[0])
        U0 = -float(np.sum(dptau[dptau < 0])) * p0c        # SR loss per turn, positive
        ratio = _check_rf_headroom(U0 / v0.sum(), "RF phase seed", rf_warn_ratio)

        # pi - arcsin, the STABLE synchronous point above transition, and the
        # same branch _solve_rf_phase targets. 2pi - arcsin also closes the ring
        # and balances the energy, but it is the unstable point: twiss fails on
        # it (LinAlgError / "Invalid n3") and at ttbar a single turn loses the
        # particle. The two must agree, or the refinement jumps by pi.
        phi_s = np.pi - np.arcsin(ratio)
        rf['phase_taper'].set_values(np.full_like(v0, phi_s) - _base_phase(rf))
        if verbose:
            print(f"    seed {it}: U0 = {U0/1e6:.2f} MeV, U0/sumV = {ratio:.6f}, "
                  f"phase_s = {phi_s:.6f} rad ({np.rad2deg(phi_s):.3f} deg)")
    return phi_s


def _one_turn_eloss(line, particle_on_co):
    """Track one turn from the closed orbit; return (record, eloss in ptau units)."""
    p = particle_on_co.copy()
    line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
    trco = line.record_last_track
    dptau = np.diff(trco.ptau[0])
    return trco, np.sum(dptau[dptau < 1e-10])


# --------------------------------------------------------------------------
# RF phase only, for an already-tapered machine
# --------------------------------------------------------------------------

def get_rf_phase(
    line,
    max_iter: int=10,           # Max iterations on the RF phase
    tolerance: float=1e-13,     # Tolerance on energy loss convergence
    rf_warn_ratio: float=0.9,   # Warn when sin(phase_s) exceeds this
    co_guess=None,              # Optional closed-orbit seed
    auto_seed: bool=True,       # Seed phase_taper if no closed orbit is found
    verbose: bool=False,
    ):
    """
    Set the cavity RF phase for the machine *as it is currently tapered*.

    Drop-in replacement for the notebook's ``get_rf_phase``, but it solves the
    machine you actually built:

      * ``delta_taper`` is **read, never written** -- your coarse/grouped
        tapering model is what determines the energy loss, and it is left
        exactly as you set it.
      * ``XS_FLAG_SR_TAPER`` is never enabled, so the stored ``delta_taper`` is
        honoured instead of being overwritten by the particle's live delta
        (which would silently solve a perfectly tapered machine).
      * the closed orbit is found on the real radiating lattice, not under
        ``XTRACK_MULTIPOLE_NO_SYNRAD`` -- so the orbit the phase is derived from
        belongs to the tapered lattice, not to an untapered stand-in.

    Uses the same ``_solve_rf_phase`` as ``compensate_energy_loss``, so both
    routines set the phase by identical arithmetic.

    ``auto_seed`` handles the extreme case. If the first closed-orbit search
    fails -- typical at ttbar, where the cavities default to phase = pi and
    replace no energy -- phase_taper is seeded with the synchronous phase
    measured from a single turn (no closed orbit needed) and the search is
    retried. Set it False to get the bare failure instead.

    Returns a dict with the energy profile of the current machine, the phase
    that was applied, and the residual energy error.

    Note: ``result['delta']`` is the delta profile of *this* (coarse) machine.
    The notebook's original returned the *ideal* profile instead, because it
    always tracked with perfect tapering.
    """
    rf = _rf_setters(line)
    _clear_lag_taper(rf)

    delta_taper_setter = line.attr._cache['delta_taper'].multisetter
    delta_taper_before = delta_taper_setter.get_values().copy()

    particle_on_co = co_guess
    eloss0 = 0.0
    history = []
    trco = None
    net_eV = None

    seeded = not auto_seed
    for i in range(0, max_iter):
        try:
            particle_on_co = line.find_closed_orbit(co_guess=particle_on_co)
        except ClosedOrbitSearchError as err:
            if seeded:
                raise ClosedOrbitSearchError(
                    f"Closed orbit search failed at iteration {i} on the tapered "
                    "lattice, and the RF phase seed did not recover it. The current "
                    "tapering model may be too coarse to close the ring -- run "
                    "compensate_energy_loss() first to obtain a viable starting "
                    f"point. Original error: {err}") from err
            # No closed orbit yet: the cavities are probably at the wrong phase
            # (default pi = zero crossing) rather than the tapering being at fault.
            # Put them on the synchronous phase and try once more.
            seeded = True
            if verbose:
                print("  closed orbit not found, seeding RF phase from one-turn energy loss")
            _seed_phase_taper(line, rf, rf_warn_ratio, verbose=verbose)
            try:
                particle_on_co = line.find_closed_orbit(co_guess=co_guess)
            except ClosedOrbitSearchError as err2:
                raise ClosedOrbitSearchError(
                    "Closed orbit search failed even after seeding the RF phase to "
                    "the synchronous value. The lattice cannot be closed with the "
                    "current tapering model. Original error: "
                    f"{err2}") from err2

        trco, eloss = _one_turn_eloss(line, particle_on_co)
        eloss_eV = float(np.asarray(eloss * line.particle_ref.p0c).ravel()[0])
        # net imbalance over the turn: what the RF failed to replace
        net_eV = float(np.asarray(
            (trco.ptau[0, -1] - trco.ptau[0, 0]) * line.particle_ref.p0c).ravel()[0])
        history.append(eloss_eV)
        if verbose:
            print(f"Iteration {i}: SR loss {eloss_eV/1e6:.6f} MeV, "
                  f"net {net_eV:+.3f} eV (diff {np.abs(eloss - eloss0):.3e})")

        if np.abs(eloss - eloss0) < tolerance:
            break
        eloss0 = eloss

        _solve_rf_phase(line, trco, rf, rf_warn_ratio, f"get_rf_phase iteration {i}")

    # delta_taper is never written above; assert it to make that contract explicit
    if not np.array_equal(delta_taper_before, delta_taper_setter.get_values()):
        raise RuntimeError("delta_taper changed during get_rf_phase -- this is a bug")

    phase_taper = rf['phase_taper'].get_values()
    if verbose:
        print("phase_taper (rad):", phase_taper)
        print("phase_taper (deg):", np.rad2deg(phase_taper))

    return {
        's': trco.s[0, :-1],
        'delta': trco.delta[0, :-1],          # profile of THIS machine, not the ideal one
        'phase_taper': phase_taper,
        'phase_taper_deg': np.rad2deg(phase_taper),
        'eloss_eV': history[-1] if history else None,   # SR loss per turn
        'energy_error_eV': net_eV,                      # residual the RF did not replace
        'eloss_history_eV': history,
        'particle_on_co': particle_on_co,
        'monitor': trco,
    }


# --------------------------------------------------------------------------
# Full tapering + RF phase
# --------------------------------------------------------------------------

def compensate_energy_loss(
    line,
    delta_taper_setters_mean: Optional[list]=None, # Groups for tapering
    delta_taper_setters_no_tapering: Optional[list]=None, # Groups to explicitly exclude from tapering
    apply_full_taper: bool=True, # Wheter to apply individual tapering
    matching_function_var: Optional[xt.VaryList] = None, # Matching variables
    matching_function_targets: Optional[list] = None, # Matching targets
    on_iteration: Optional[Callable]=None, # Hook called at the end of each iteration
    cavity_midpoint: str=None, # Optional midpoint to cycle the beamline
    max_iter: int=10, # Max iterations for tapering
    max_threading_iter: int=2, # Max attempts to fix orbit via threading
    tolerance: float=1e-13, # Tolerance on energy loss convergence
    rf_warn_ratio: float=0.9, # Warn when sin(phase_s) exceeds this
    verbose: bool=False
    ):

    """
    Main function that replace the current XSuite "compensate_radiation_energy_loss".
    Inplace modification of the line, the tapering correction is applied using the "delta_taper" variable and the RF phase is set to compensate the loss energy.

    Return the closed orbit

    delta_taper_setters_mean: list of xt.MultiSetter, one for each group of magnet in which mean tapering is applied
    delta_taper_setters_no_tapering: list of xt.MultiSetter, one for each group of magnet in which no tapering is applied
    matching_function_var: xt.VaryList for matching inside the tapering loop
    matching_function_targets: list of xt.Target for matching inside the tapering loop

    on_iteration: optional callback ``f(line, i, eloss)`` invoked at the end of
        every tapering iteration. Extension point for anything that should run
        per iteration without being hard-wired into this function.

    rf_warn_ratio: the cavity voltage is fixed; only the phase is adjusted, so
        sin(phase_s) = (energy to replace) / (q * V0) must stay below 1. Emit a
        warning once it exceeds this fraction of crest. Coarser tapering costs
        more energy per turn and pushes it up, so this is the early signal that
        a granularity setting is running out of RF.
    """

    def _threading(threading_iterations=2):
        """
        Iterative tracking with strength magnet correction (delta_taper) to the energy loss through the corresponding element.
        Inplace modification of the line.
        """
        if verbose:
            print("Performing threading...")

        full_delta_taper = np.zeros_like(full_delta_taper_setter.get_values())
        for _it in range(0, threading_iterations):
            j = 0
            p = line.particle_ref.copy()
            for i, element in enumerate(tqdm(line.element_names, )):
                # Tracking element by element and correction of magnet strength if element with delta_taper
                if i == 0:
                    continue
                p0 = p.copy()
                line.track(p, ele_start=i-1, ele_stop=i)

                # Set delta_taper based on particle momentum loss at current element
                if full_delta_taper_mask[i]:
                    full_delta_taper[j] = p.ptau[0]
                    full_delta_taper_setter.set_values(full_delta_taper)
                    j = j + 1

                # Re-track from the previous element with updated delta_taper
                line.track(p0, ele_start=i-1, ele_stop=i)
                p = p0

            _ = np.diff(full_delta_taper_setter.get_values()) # Relative delta value to calculate the total energy gained in the cavities (egain) and lost outside (eloss)
            if verbose:
                # Predict energy gain based on RF parameters and compare with actual
                egain = 0
                for voltage, base, taper in zip(rf['v'].get_values(), _base_phase(rf), rf['phase_taper'].get_values()):
                    egain += voltage * np.sin(base + taper)
                egain_actual = np.sum(_[_ > 0]) * p.p0c[0]
                print(f"Predicted energy gain from cavities: {egain / 1e6} MeV")
                print(f"Actual energy gain from cavities: {egain_actual / 1e6} MeV")

            eloss = np.sum(_[_ < 0]) * p.p0c[0]
            if verbose:
                print(f"Energy loss through the ring: {eloss / 1e6} MeV")

            # Adjust RF phase to match observed energy loss

            phi_taper = np.arcsin(_check_rf_headroom(
                -eloss / np.sum(rf['v'].get_values()), "threading", rf_warn_ratio))
            base_phase = _base_phase(rf)
            target_phase = np.pi - phi_taper            # total phase the cavity must sit at
            taper = target_phase - base_phase           # what phase_taper has to supply
            if verbose:
                print(f"Base phase (rad): {base_phase[0]}")
                print(f"Current tapering phase (rad): {rf['phase_taper'].get_values()[0]}")
                print(f"Current total phase (rad): {base_phase[0] + rf['phase_taper'].get_values()[0]}")
                print(f"Required total phase (rad): {target_phase[0] if np.ndim(target_phase) else target_phase}")
                print(f"Change of tapering phase applied (rad): {(taper - rf['phase_taper'].get_values())[0]}")
            rf['phase_taper'].set_values(taper)



    def _compensate_eloss_and_tapper(particle_on_co):
        """
        Tapering function that iteratively compute the magnet strength correction and the RF phase.
        Individual or group tapering can be performed.
        Additional machting in the tapering loop can be performed.
        """

        tmp = []  # BUGFIX: was only bound inside the loop -> NameError in the
                  # final verbose block if the loop broke on the first iteration.
        eloss0 = 0 # Initial energy loss in one turn to initiate the function
        for i in range(0, max_iter):
            if verbose:
                print(f"Iteration {i}")
            # Check if the closed orbit research is valid
            try:
                particle_on_co = line.find_closed_orbit()
            except ClosedOrbitSearchError:
                return particle_on_co
            if verbose:
                print(particle_on_co)

            # Track particle on closed orbit to have the delta value for each element
            trco, eloss = _one_turn_eloss(line, particle_on_co)
            if verbose:
                print(f"Energy loss (MeV): {eloss * line.particle_ref.p0c[0] / 1e6}")
                print(f"Energy loss diff.: {np.abs(eloss - eloss0)}")

            # Comparison with previous eloss value to identify if the fixed point has been found
            if np.abs(eloss - eloss0) < tolerance:
                break

            # Compute ideal tapering based on delta values
            full_delta_taper = (0.5*(trco.delta[0, :-1] + trco.delta[0, 1:]))[full_delta_taper_mask]
            full_delta_taper_setter.set_values(full_delta_taper)

            tmp = []
            if delta_taper_setters_mean is not None:
                tmp = [_.get_values() for _ in delta_taper_setters_mean]  # List of setter for each magnet group for mean tapering
                # print(tmp)

            # Set to zero delta_taper variable to remove the ideal tapering in case one only want group tapering in some elements
            if not apply_full_taper:
                full_delta_taper_setter.set_values(np.zeros_like(full_delta_taper_setter.get_values()))

            # For each group of element of the setter list, the delta_taper mean value is set for all elements
            # BUGFIX: this loop used `i`, shadowing the outer iteration counter that
            # the `i != max_iter-1` matching guard below depends on. Renamed to `g`.
            if delta_taper_setters_mean is not None:
                for g in range(0, len(tmp)):
                    vvv = np.zeros_like(tmp[g]) + np.mean(tmp[g])
                    delta_taper_setters_mean[g].set_values(vvv)

            # For each group, the delta_taper is set to zero for all elements
            if delta_taper_setters_no_tapering is not None:
                for _setter in delta_taper_setters_no_tapering:
                    _setter.set_values(np.zeros_like(_setter.get_values()))

            # Perform matching that cna influence the closed orbit and so the tapering.
            # The condition is set to avoid doing the matching when the closed orbit is not yet good enoug
            # The matching is not perform for the last iteration of the function to conserv the best parameters for the output line
            if (matching_function_var is not None and matching_function_targets is not None):
               if np.abs(eloss - eloss0) < tolerance*1e3:
                   if ((np.abs(eloss - eloss0) >= tolerance) and (i != max_iter-1)):
                        if verbose:
                            print('Matching with given conditions')

                        particle_on_co = line.find_closed_orbit()
                        opt= line.match(
                            vary = matching_function_var,
                            targets = matching_function_targets,
                            verbose = verbose,
                            co_guess = particle_on_co,
                            compute_chromatic_properties = False,
                            eneloss_and_damping=True,
                            compensate_radiation_energy_loss=False # Need to force the function of XSuite not the be applied
                        )

            eloss0 = eloss

            # Adjust RF phase based on required energy gain and voltage
            _solve_rf_phase(line, trco, rf, rf_warn_ratio, f"tapering iteration {i}")

            # Extension point. Anything that perturbs the optics (matching, orbit
            # correction) goes here or, preferably, in an outer loop -- not inline.
            if on_iteration is not None:
                on_iteration(line, i, eloss)

        if verbose:
            print("Final configuration:")
            print("phase_taper (rad):", rf['phase_taper'].get_values())
            print("phase_taper (deg):", np.rad2deg(rf['phase_taper'].get_values()))
            print([np.mean(_) for _ in tmp])
            print(particle_on_co)

        return particle_on_co

    # Prepare beamline by cycling to midpoint of cavity if given
    initial_starting_point = line.element_names[0]
    print(f"Initial starting point: {initial_starting_point}")
    if cavity_midpoint is not None:
        line.cycle(name_first_element=cavity_midpoint, inplace=True)

    # Extract RF and tapering multisetter/masks from line
    rf = _rf_setters(line)
    _clear_lag_taper(rf)

    full_delta_taper_setter = line.attr._cache['delta_taper'].multisetter
    full_delta_taper_mask = line.attr._cache['delta_taper'].mask

    # Attempt to find closed orbit; if fails, apply threading
    particle_on_co = None
    for i in range(max_threading_iter):
        try:
            particle_on_co = line.find_closed_orbit()
            if verbose:
                print("Closed orbit search succeeded. Proceeding with compensation and tapering.")
            break
        except ClosedOrbitSearchError:
            if verbose:
                print("Closed orbit search failed, this will require threading.")
            _threading()
            particle_on_co = line.find_closed_orbit()

    # Apply full compensation and tapering
    particle_on_co = _compensate_eloss_and_tapper(particle_on_co)
    try:
        particle_on_co = line.find_closed_orbit(co_guess=particle_on_co)
    except ClosedOrbitSearchError:
        return particle_on_co

    # Tracking to have the coordinates of the particle on closed orbit at the starting point of the lattice
    p = particle_on_co.copy()
    line.track(p, ele_stop=initial_starting_point)

    # Cycle to have the default starting point of the lattice
    if cavity_midpoint is not None:
        line.cycle(name_first_element=initial_starting_point, inplace=True)

    return line.find_closed_orbit(co_guess=p, co_search_at=0)
