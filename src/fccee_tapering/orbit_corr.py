"""Closed-orbit correction towards a reference orbit.

Two independent correctors are available:

    `correct_orbit_with_correctors`   SVD on the h/v steering correctors,
                                      both planes
    `correct_orbit_with_dipole_trims` least squares on one trim per arc cell,
                                      horizontal only

Both drive the orbit towards the orbit of a reference lattice rather than
towards zero: coarse tapering distorts the closed orbit, and the target is the
orbit the ideally tapered machine would have.

The corrector classes are adapted from xsuite's own `trajectory_correction`,
with a reference orbit added; the response matrices and the SVD solve are
otherwise unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xtrack as xt
from scipy.sparse.linalg import lsmr

# Conversion from a kick angle to an integrated field: B*l = kick * Brho.
CLIGHT_GEV = 0.299792458


def magnetic_rigidity(line) -> float:
    """B*rho in T m, for converting kick angles to integrated fields."""
    p0c_gev = float(np.asarray(line.particle_ref.p0c).ravel()[0]) * 1e-9
    charge = abs(float(np.asarray(line.particle_ref.q0).ravel()[0]))
    return p0c_gev / (CLIGHT_GEV * charge)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


def _compute_correction(
    x_iter, response_matrix, n_micado=None, rcond=None, n_singular_values=None, gain=1
):
    if isinstance(response_matrix, (list, tuple)):
        assert len(response_matrix) == 3  # U, S, Vt
        U, S, Vh = response_matrix
        if n_singular_values is not None:
            U = U[:, :n_singular_values]
            S = S[:n_singular_values]
            Vh = Vh[:n_singular_values, :]
        # The SVD factors are used directly below. Reconstruct the full
        # matrix only when MICADO needs to test individual correctors.
        n_hcorrectors = Vh.shape[1]
        if n_micado is not None:
            response_matrix = U @ np.diag(S) @ Vh
    else:
        assert n_singular_values is None
        U = None
        S = None
        Vh = None
        n_hcorrectors = response_matrix.shape[1]

    if n_micado is not None:
        used_correctors = []
        for _i_micado in range(n_micado):
            residuals = []
            for i_corr in range(n_hcorrectors):
                if i_corr in used_correctors:
                    residuals.append(np.nan)
                    continue
                mask_corr = np.zeros(n_hcorrectors, dtype=bool)
                mask_corr[i_corr] = True
                for i_used in used_correctors:
                    mask_corr[i_used] = True
                _, residual_x, _rank_x, _sval_x = np.linalg.lstsq(
                    response_matrix[:, mask_corr], -x_iter, rcond=rcond
                )
                residuals.append(residual_x[0])
            used_correctors.append(np.nanargmin(residuals))
        mask_corr = np.zeros(n_hcorrectors, dtype=bool)
        mask_corr[used_correctors] = True
    else:
        mask_corr = np.ones(n_hcorrectors, dtype=bool)
        mask_corr[:] = True

    if mask_corr.all() and S is not None:
        S_inv = np.zeros_like(S)
        S_inv[S > 0] = 1 / S[S > 0]
        if rcond is not None:
            S_inv[S < rcond * S[0]] = 0
        correction_masked = Vh.T.conj() @ (np.diag(S_inv) @ (U.T.conj() @ (-x_iter)))
    else:
        correction_masked, _residual_x, _rank_x, _sval_x = np.linalg.lstsq(
            response_matrix[:, mask_corr], -x_iter, rcond=rcond
        )
    correction_masked *= gain

    correction_x = np.zeros(n_hcorrectors)
    correction_x[mask_corr] = correction_masked
    return correction_x


def _build_response_matrix(tw, monitor_names, corrector_names, mode="closed", plane=None):
    assert mode in ["closed", "open"]
    assert plane in ["x", "y"]

    indices_monitors = tw.rows.indices[monitor_names]
    indices_correctors = tw.rows.indices[corrector_names]
    bet_monitors = tw["bet" + plane][indices_monitors]
    bet_correctors = tw["bet" + plane][indices_correctors]
    mu_monitor = tw["mu" + plane][indices_monitors]
    mux_correctors = tw["mu" + plane][indices_correctors]

    bet_prod = np.atleast_2d(bet_monitors).T @ np.atleast_2d(bet_correctors)
    # Broadcasting avoids two full-size temporary tiled arrays.
    mu_diff = mu_monitor[:, None] - mux_correctors[None, :]

    if mode == "open":
        mu_diff[mu_diff < 0] = 0
        response_matrix = np.sqrt(bet_prod) * np.sin(2 * np.pi * np.abs(mu_diff))
    else:
        tune = tw.qx if plane == "x" else tw.qy
        response_matrix = (
            np.sqrt(bet_prod)
            / 2
            / np.sin(np.pi * tune)
            * np.cos(np.pi * tune - 2 * np.pi * np.abs(mu_diff))
        )
    return response_matrix


class OrbitCorrectionSinglePlane:
    """Correction of one plane onto a stored reference orbit."""

    def __init__(
        self,
        line,
        plane,
        monitor_names,
        corrector_names,
        start=None,
        end=None,
        twiss_table=None,
        n_micado=None,
        n_singular_values=None,
        rcond=None,
        reference_orbit=None,
    ):
        assert plane in ["x", "y"]
        self.twiss_table = twiss_table
        if twiss_table is not None:
            assert twiss_table.reference_frame == "proper"
        self.reference_orbit = reference_orbit

        if start is None:
            assert end is None
            self.mode = "closed"
            if self.twiss_table is None:
                self.twiss_table = line.twiss4d(reverse=False)
        else:
            assert end is not None
            self.mode = "open"
            if self.twiss_table is None:
                self.twiss_table = line.twiss4d(
                    start=start,
                    end=end,
                    init=xt.TwissInit(
                        W_matrix=np.eye(6),
                        particle_on_co=line.build_particles(
                            x=0, y=0, px=0, py=0, zeta=0, delta=0
                        ),
                        element_name=start,
                    ),
                    reverse=False,
                )

        if corrector_names is None:
            corr_names_from_line = getattr(line, f"steering_correctors_{plane}")
            assert corr_names_from_line is not None
            if start is not None:
                corrector_names = [
                    nn for nn in corr_names_from_line if nn in self.twiss_table.name
                ]
            else:
                corrector_names = corr_names_from_line

        if monitor_names is None:
            monitor_names = getattr(line, f"steering_monitors_{plane}")
            assert monitor_names is not None
            if start is not None:
                monitor_names = [nn for nn in monitor_names if nn in self.twiss_table.name]

        assert len(monitor_names) > 0
        assert len(corrector_names) > 0

        self.line = line
        self.plane = plane
        # Plain str, never numpy str: these index element_refs and so reach the
        # serialised expressions, where numpy's own repr is not loadable.
        self.monitor_names = [str(name) for name in monitor_names]
        self.corrector_names = [str(name) for name in corrector_names]
        self.start = start
        self.end = end
        self.n_micado = n_micado
        self.rcond = rcond
        self.n_singular_values = n_singular_values

        self.response_matrix = _build_response_matrix(
            plane=self.plane,
            tw=self.twiss_table,
            monitor_names=self.monitor_names,
            corrector_names=self.corrector_names,
            mode=self.mode,
        )

        U, S, Vt = np.linalg.svd(self.response_matrix, full_matrices=False)
        self.singular_values = S
        self.singular_vectors_out = U
        self.singular_vectors_in = Vt

        tw_table_local = self.twiss_table.rows[start:end]
        self._indices_monitor = tw_table_local.rows.indices[self.monitor_names]
        self._indices_correctors = tw_table_local.rows.indices[self.corrector_names]
        self.s_correctors = tw_table_local.s[self._indices_correctors]
        self.s_monitors = tw_table_local.s[self._indices_monitor]

        self._add_correction_knobs()

    def set_reference_orbit_from_twiss(self, tw):
        """Take the target orbit at the monitors from a reference lattice."""
        indices = tw.rows.indices[self.monitor_names]
        self.reference_orbit = np.asarray(tw[self.plane][indices], dtype=float).copy()

    def _compute_tw_orbit(self):
        twinit = None
        if self.mode == "open":
            twinit = xt.TwissInit(
                W_matrix=np.eye(6),
                particle_on_co=self.line.build_particles(
                    x=0, y=0, px=0, py=0, zeta=0, delta=0
                ),
                element_name=self.start,
            )
        return self.line.twiss(
            only_orbit=True, start=self.start, end=self.end, init=twinit, reverse=False
        )

    def _measure_position(self, tw_orbit=None):
        if tw_orbit is None:
            tw_orbit = self._compute_tw_orbit()
        indices = tw_orbit.rows.indices[self.monitor_names]
        position = np.asarray(tw_orbit[self.plane][indices], dtype=float)
        if self.reference_orbit is not None:
            assert len(self.reference_orbit) == len(position)
            position = position - self.reference_orbit
        return position

    def _compute_correction(
        self, position, n_micado=None, n_singular_values=None, rcond=None, gain=1
    ):
        if rcond is None:
            rcond = self.rcond
        if n_singular_values is None:
            n_singular_values = self.n_singular_values
        if n_micado is None:
            n_micado = self.n_micado
        return _compute_correction(
            position,
            response_matrix=(
                self.singular_vectors_out,
                self.singular_values,
                self.singular_vectors_in,
            ),
            n_micado=n_micado,
            rcond=rcond,
            n_singular_values=n_singular_values,
            gain=gain,
        )

    def _add_correction_knobs(self):
        self.correction_knobs = []
        for nn_kick in self.corrector_names:
            corr_knob_name = f"orbit_corr_{nn_kick}_{self.plane}"
            assert hasattr(self.line[nn_kick], "knl")
            assert hasattr(self.line[nn_kick], "ksl")
            if corr_knob_name not in self.line.vars:
                self.line.vars[corr_knob_name] = 0
            if self.plane == "x":
                if self.line.element_refs[nn_kick].knl[0]._expr is None or (
                    self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].knl[0]._expr._get_dependencies()
                ):
                    if self.line.element_refs[nn_kick].knl[0]._expr is not None:
                        self.line.element_refs[nn_kick].knl[0] -= self.line.vars[
                            f"orbit_corr_{nn_kick}_x"
                        ]
                    else:
                        val = self.line.element_refs[nn_kick].knl[0]._value
                        if hasattr(val, "get"):
                            val = val.get()
                        self.line.element_refs[nn_kick].knl[0] = val - self.line.vars[
                            f"orbit_corr_{nn_kick}_x"
                        ]
            elif self.plane == "y":
                if self.line.element_refs[nn_kick].ksl[0]._expr is None or (
                    self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].ksl[0]._expr._get_dependencies()
                ):
                    if self.line.element_refs[nn_kick].ksl[0]._expr is not None:
                        self.line.element_refs[nn_kick].ksl[0] += self.line.vars[
                            f"orbit_corr_{nn_kick}_y"
                        ]
                    else:
                        val = self.line.element_refs[nn_kick].ksl[0]._value
                        if hasattr(val, "get"):
                            val = val.get()
                        self.line.element_refs[nn_kick].ksl[0] = val + self.line.vars[
                            f"orbit_corr_{nn_kick}_y"
                        ]
            self.correction_knobs.append(corr_knob_name)

    def _apply_correction(self, correction=None):
        if correction is None:
            correction = self.correction
        for nn_knob, kick in zip(self.correction_knobs, correction):
            self.line.vars[nn_knob] += kick

    def get_kick_values(self):
        return np.array([self.line.vv[nn_knob] for nn_knob in self.correction_knobs])

    def clear_correction_knobs(self):
        for nn_knob in self.correction_knobs:
            self.line.vars[nn_knob] = 0


class TrajectoryCorrection:
    """Both planes of an orbit correction."""

    def __init__(
        self,
        line,
        start=None,
        end=None,
        twiss_table=None,
        monitor_names_x=None,
        corrector_names_x=None,
        monitor_names_y=None,
        corrector_names_y=None,
        n_micado=None,
        n_singular_values=None,
        rcond=None,
    ):
        if isinstance(rcond, (tuple, list)):
            rcond_x, rcond_y = rcond
        else:
            rcond_x, rcond_y = rcond, rcond
        if isinstance(n_singular_values, (tuple, list)):
            n_singular_values_x, n_singular_values_y = n_singular_values
        else:
            n_singular_values_x, n_singular_values_y = n_singular_values, n_singular_values
        if isinstance(n_micado, (tuple, list)):
            n_micado_x, n_micado_y = n_micado
        else:
            n_micado_x, n_micado_y = n_micado, n_micado

        if (
            monitor_names_x is not None
            or corrector_names_x is not None
            or line.steering_correctors_x is not None
            or line.steering_monitors_x is not None
        ):
            self.x_correction = OrbitCorrectionSinglePlane(
                line=line,
                plane="x",
                monitor_names=monitor_names_x,
                corrector_names=corrector_names_x,
                start=start,
                end=end,
                twiss_table=twiss_table,
                n_micado=n_micado_x,
                n_singular_values=n_singular_values_x,
                rcond=rcond_x,
            )
        else:
            self.x_correction = None

        if (
            monitor_names_y is not None
            or corrector_names_y is not None
            or line.steering_correctors_y is not None
            or line.steering_monitors_y is not None
        ):
            self.y_correction = OrbitCorrectionSinglePlane(
                line=line,
                plane="y",
                monitor_names=monitor_names_y,
                corrector_names=corrector_names_y,
                start=start,
                end=end,
                twiss_table=twiss_table,
                n_micado=n_micado_y,
                n_singular_values=n_singular_values_y,
                rcond=rcond_y,
            )
        else:
            self.y_correction = None

    def clear_correction_knobs(self):
        if self.x_correction is not None:
            self.x_correction.clear_correction_knobs()
        if self.y_correction is not None:
            self.y_correction.clear_correction_knobs()

    @property
    def line(self):
        if self.x_correction is not None:
            return self.x_correction.line
        return self.y_correction.line

    @property
    def twiss_table(self):
        if self.x_correction is not None:
            return self.x_correction.twiss_table
        return self.y_correction.twiss_table


# --------------------------------------------------------------------------
# Correction with the steering correctors
# --------------------------------------------------------------------------


def correct_orbit_with_correctors(
    correction,
    tolerance_x=1e-6,
    tolerance_y=1e-6,
    max_iterations=10,
    n_singular_values=None,
    rcond=1e-6,
    gain=1.0,
    verbose=False,
):
    """Iteratively correct x and y towards the stored reference orbit.

    tolerance_x, tolerance_y: RMS monitor residual in metres.
    gain: fraction of the computed correction applied per iteration; reduce
        below 1 if the iteration oscillates.

    Returns the per-iteration history and whether the tolerance was reached.
    A run that stops improving above the tolerance has still found its floor;
    the caller decides what to make of that.
    """
    x_corr = correction.x_correction
    y_corr = correction.y_correction

    history = []
    reached_tolerance = False

    for iteration in range(max_iterations + 1):
        # Recompute the present orbit after the previous correction.
        twiss_orbit = x_corr._compute_tw_orbit()

        # Both already return current minus reference at the monitors.
        residual_x = x_corr._measure_position(twiss_orbit)
        residual_y = y_corr._measure_position(twiss_orbit)

        rms_x = float(np.sqrt(np.mean(residual_x**2)))
        rms_y = float(np.sqrt(np.mean(residual_y**2)))
        max_x = float(np.max(np.abs(residual_x)))
        max_y = float(np.max(np.abs(residual_y)))

        history.append(
            {
                "iteration": iteration,
                "x_rms_m": rms_x,
                "y_rms_m": rms_y,
                "x_max_m": max_x,
                "y_max_m": max_y,
            }
        )
        if verbose:
            print(
                f"Iteration {iteration}: x RMS = {rms_x * 1e6:.4f} um, "
                f"y RMS = {rms_y * 1e6:.4f} um"
            )

        converged_x = rms_x <= tolerance_x
        converged_y = rms_y <= tolerance_y

        if converged_x and converged_y:
            reached_tolerance = True
            break
        if iteration == max_iterations:
            break

        if not converged_x:
            x_corr._apply_correction(
                x_corr._compute_correction(
                    position=residual_x,
                    n_singular_values=n_singular_values,
                    rcond=rcond,
                    gain=gain,
                )
            )
        if not converged_y:
            y_corr._apply_correction(
                y_corr._compute_correction(
                    position=residual_y,
                    n_singular_values=n_singular_values,
                    rcond=rcond,
                    gain=gain,
                )
            )

    return pd.DataFrame(history), reached_tolerance


# --------------------------------------------------------------------------
# Correction with dipole trims
# --------------------------------------------------------------------------


def build_horizontal_response_matrix(twiss, monitor_names, trim_dipole_names):
    """Horizontal orbit response of the monitors to a kick at each trim."""
    monitor_indices = twiss.rows.indices[list(monitor_names)]
    trim_indices = twiss.rows.indices[list(trim_dipole_names)]

    beta_monitor = np.asarray(twiss.betx[monitor_indices], dtype=float)
    beta_trim = np.asarray(twiss.betx[trim_indices], dtype=float)
    mu_monitor = np.asarray(twiss.mux[monitor_indices], dtype=float)
    mu_trim = np.asarray(twiss.mux[trim_indices], dtype=float)
    tune = float(twiss.qx)

    beta_product = beta_monitor[:, None] * beta_trim[None, :]
    phase_difference = np.abs(mu_monitor[:, None] - mu_trim[None, :])

    return (
        np.sqrt(beta_product)
        / (2.0 * np.sin(np.pi * tune))
        * np.cos(np.pi * tune - 2.0 * np.pi * phase_difference)
    )


def set_dipole_trim_kicks(line, dipole_names, base_knl, trim_kicks_rad):
    """Apply trim kicks on top of each dipole's nominal bending strength."""
    for dipole_name, base_value, kick in zip(dipole_names, base_knl, trim_kicks_rad):
        # A positive horizontal kick corresponds to a negative knl[0].
        line.element_refs[str(dipole_name)].knl[0] = float(base_value - kick)


def prepare_dipole_trims(line, dipole_names) -> np.ndarray:
    """Record the dipoles' nominal strengths and set their trims to zero.

    The returned baseline is the only record of the untrimmed strength: the
    trim is applied by overwriting knl[0], which destroys what was there.
    """
    base_knl = np.asarray(
        [float(line[name].knl[0]) for name in dipole_names], dtype=float
    )
    set_dipole_trim_kicks(line, dipole_names, base_knl, np.zeros(len(dipole_names)))
    return base_knl


def verify_dipole_trims(line, dipole_names, base_knl, test_kick_rad=1e-9) -> None:
    """Check that a trim kick reaches knl[0], then restore zero trims."""
    test_kicks = np.full(len(dipole_names), test_kick_rad)
    set_dipole_trim_kicks(line, dipole_names, base_knl, test_kicks)

    installed_knl = np.asarray([float(line[name].knl[0]) for name in dipole_names])
    expected_knl = base_knl - test_kicks

    set_dipole_trim_kicks(line, dipole_names, base_knl, np.zeros(len(dipole_names)))

    if not np.allclose(installed_knl, expected_knl, rtol=1e-10, atol=1e-16):
        bad = np.flatnonzero(
            ~np.isclose(installed_knl, expected_knl, rtol=1e-10, atol=1e-16)
        )
        raise RuntimeError(
            f"{len(bad)} dipole trims failed the knl[0] test. "
            f"First failed element: {dipole_names[bad[0]]}"
        )


def correct_orbit_with_dipole_trims(
    line,
    reference,
    monitor_names,
    trim_dipole_names,
    trim_base_knl,
    tolerance_rms_m=1e-6,
    max_iterations=10,
    gain=1.0,
    regularization=1e-6,
    solver_max_iterations=400,
    verbose=False,
):
    """Correct the horizontal orbit onto the reference using dipole trims.

    Returns the per-iteration history, the total kick applied at each trim,
    and whether the tolerance was reached.
    """
    response = build_horizontal_response_matrix(
        reference, monitor_names, trim_dipole_names
    )

    # Normalise columns so the regularisation treats all trim locations fairly.
    column_scale = np.linalg.norm(response, axis=0)
    column_scale[column_scale == 0.0] = 1.0
    response_scaled = response / column_scale[None, :]

    reference_indices = reference.rows.indices[list(monitor_names)]
    reference_x = np.asarray(reference.x[reference_indices], dtype=float)

    history = []
    total_trim_kicks = np.zeros(len(trim_dipole_names), dtype=float)
    reached_tolerance = False

    for iteration in range(max_iterations + 1):
        twiss_orbit = line.twiss(only_orbit=True)
        orbit_indices = twiss_orbit.rows.indices[list(monitor_names)]
        residual = np.asarray(twiss_orbit.x[orbit_indices], dtype=float) - reference_x

        rms_residual = float(np.sqrt(np.mean(residual**2)))
        maximum_residual = float(np.max(np.abs(residual)))

        history.append(
            {
                "iteration": iteration,
                "x_rms_m": rms_residual,
                "x_max_m": maximum_residual,
            }
        )
        if verbose:
            print(
                f"Iteration {iteration}: x RMS = {rms_residual * 1e6:.4f} um, "
                f"max = {maximum_residual * 1e6:.4f} um"
            )

        if rms_residual <= tolerance_rms_m:
            reached_tolerance = True
            break
        if iteration == max_iterations:
            break

        solution = lsmr(
            response_scaled,
            -residual,
            damp=regularization,
            atol=1e-11,
            btol=1e-11,
            maxiter=solver_max_iterations,
        )
        delta_kick = gain * solution[0] / column_scale

        total_trim_kicks += delta_kick
        set_dipole_trim_kicks(line, trim_dipole_names, trim_base_knl, total_trim_kicks)

        if verbose:
            print(
                f"  LSMR iterations = {solution[2]}, "
                f"linear residual norm = {solution[3] * 1e6:.4f} um, "
                f"step RMS = {1e6 * np.sqrt(np.mean(delta_kick**2)):.4f} urad, "
                f"total trim RMS = {1e6 * np.sqrt(np.mean(total_trim_kicks**2)):.4f} urad"
            )

    return pd.DataFrame(history), total_trim_kicks, reached_tolerance
