"""Centroid addition utilities for microlensing / crowded-field simulations.

This module provides helper functionality for computing the composite (flux
weighted) centroid of multiple overlapping point sources (e.g., a magnified
source star plus lens or blend stars) and for propagating those centroid
calculations across a light curve to obtain an astrometric shift time series.

The primary class :class:`CentroidAddition` is a lightweight namespace whose
methods operate either on supplied ``numpy`` arrays or on ``pandas`` DataFrames.
"""

import numpy as np
import pandas as pd


class CentroidAddition:
    """Algorithms for adding centroids and simulating astrometric shifts.

    Notes
    -----
    The class stores no state; methods could be used as module-level
    functions. It is structured as a class to mirror potential future
    extensions (e.g., configuration objects for PSF models or blending
    prescriptions).
    """

    @staticmethod
    def add_centroids(
            x_pos_S: np.ndarray,
            y_pos_S: np.ndarray,
            x_pos_L: np.ndarray,
            y_pos_L: np.ndarray,
            magnified_S_flux: np.ndarray,
            L_flux: np.ndarray
        ) -> np.ndarray:
        """Compute the flux–weighted centroid of multiple sources.

        Parameters
        ----------
        x_pos_S : ndarray, shape (N,)
            x-coordinates of the source.
        y_pos_S : ndarray, shape (N,)
            y-coordinates of the source.
        x_pos_L : ndarray, shape (N,)
            x-coordinates of the lens (or blend) components.
        y_pos_L : ndarray, shape (N,)
            y-coordinates of the lens (or blend) components.
        magnified_S_flux : ndarray, shape (N,)
            Corresponding flux of the source in arbitrary but consistent
            units; negative values are allowed mathematically but typically
            unphysical and should be pre-filtered upstream.
        L_flux : ndarray, shape (N,)
            Corresponding flux of each lens (or blend) component.

        Returns
        -------
        ndarray, shape (2,)
            Flux–weighted centroid ``[x_c, y_c]``. If the total flux is zero
            a zero vector is returned to avoid division-by-zero.
       """
        total_flux = magnified_S_flux + L_flux

        weighted_L_x_pos = x_pos_L * L_flux / total_flux
        weighted_L_y_pos = y_pos_L * L_flux / total_flux
        weighted_S_x_pos = x_pos_S * magnified_S_flux / total_flux
        weighted_S_y_pos = y_pos_S * magnified_S_flux / total_flux

        cumulative_centroid_x = weighted_S_x_pos + weighted_L_x_pos
        cumulative_centroid_y = weighted_S_y_pos + weighted_L_y_pos

        return cumulative_centroid_x, cumulative_centroid_y

    @staticmethod
    def rotate(dx, dy, mu_rel_N, mu_rel_E):
        """Rotate the centroid shift vector by relative proper motion angle + alpha.

        Parameters
        ----------
        dx : float
            Shift in x-direction.
        dy : float
            Shift in y-direction.
        mu_rel_N : float
            Relative proper motion in the North direction.
        mu_rel_E : float
            Relative proper motion in the East direction.

        Returns
        -------
        ndarray, shape (2,)
            Rotated shift vector.
        """
        alpha = np.arctan2(mu_rel_N, mu_rel_E)
        rotation_matrix = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha), np.cos(alpha)]
        ])
        return rotation_matrix @ np.array([dx, dy])

    def simulate_astrometric_shift(
        self,
        light_curve_df: pd.DataFrame,
        source_position: np.ndarray,
        lens_positions: np.ndarray,
        lens_fluxes: np.ndarray,
    ) -> pd.DataFrame:
        """Augment a light curve with per-epoch astrometric centroid shifts.

        Parameters
        ----------
        light_curve_df : pandas.DataFrame
            DataFrame containing at least a ``relative_flux`` column describing
            the magnified flux of the source at each epoch.
        source_position : ndarray, shape (2,)
            Fixed (x, y) position of the unmagnified source in chosen angular
            units.
        lens_positions : ndarray, shape (N, 2)
            Positions of each lens (or blend) contributing additional flux.
        lens_fluxes : ndarray, shape (N,)
            Relative (or absolute) fluxes of the lens components. Blend flux
            is assumed zero unless encoded here.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with added columns
            ``astrometric_shift_x`` and ``astrometric_shift_y`` containing the
            shift of the composite centroid relative to ``source_position``.

        Notes
        -----
        The source flux is taken directly from ``relative_flux`` per epoch and
        used as the magnified source contribution. No error propagation is
        performed here.
        """
        shifts = []
        for _, row in light_curve_df.iterrows():
            relative_flux = row['relative_flux']
            if relative_flux <= 0:
                shifts.append(np.array([0.0, 0.0]))
                continue

            magnified_source_flux = relative_flux
            all_positions = np.vstack([source_position, lens_positions])
            all_fluxes = np.hstack([magnified_source_flux, lens_fluxes])
            cumulative_centroid = self.add_centroids(all_positions, all_fluxes)
            shift = cumulative_centroid - source_position
            shifts.append(shift)

        light_curve_df['astrometric_shift_x'] = [shift[0] for shift in shifts]
        light_curve_df['astrometric_shift_y'] = [shift[1] for shift in shifts]
        return light_curve_df

    def plot_astrometric_shifts(
        self,
        light_curve_df: pd.DataFrame,
        t_ref: float,
        theta_E: float,
        mu_rel: float,
    ):
        """Visualize astrometric shifts and contextual lens/source geometry.

        Parameters
        ----------
        light_curve_df : pandas.DataFrame
            Light curve DataFrame containing columns: ``Simulation_time``,
            ``astrometric_shift_x``, ``astrometric_shift_y``, ``source_x``,
            ``source_y``, ``lens1_x``, ``lens1_y``, ``lens2_x``, ``lens2_y``,
            and ``measured_relative_flux`` (for scaling / reference).
        t_ref : float
            Reference time at which to highlight / interpolate positions.
        theta_E : float
            Einstein radius in the same angular units as positions and shifts.
        mu_rel : array-like of length 2
            Relative proper motion components (N, E) or (x, y) consistent with
            the positional frame; used here only for an orientation angle.

        Notes
        -----
        The current implementation produces a simple quiver plot of shift
        vectors vs index. Additional diagnostic panels (e.g., PSF heat maps)
        are planned but not yet rendered.
        """
        import matplotlib.pyplot as plt  # local import to keep base namespace light

        psf_width = 0.1  # currently unused placeholder
        pixel_scale = 0.05  # currently unused placeholder

        times = light_curve_df['Simulation_time'].to_numpy()
        shift_x_vals = light_curve_df['astrometric_shift_x'].to_numpy()
        shift_y_vals = light_curve_df['astrometric_shift_y'].to_numpy()

        shift_x_tref = np.interp(t_ref, times, shift_x_vals)
        shift_y_tref = np.interp(t_ref, times, shift_y_vals)

        lens_positions = np.array([
            light_curve_df['lens1_x'].to_numpy(), light_curve_df['lens1_y'].to_numpy(),
            light_curve_df['lens2_x'].to_numpy(), light_curve_df['lens2_y'].to_numpy()
        ])
        lens_positions_tref = [np.interp(t_ref, times, lens_positions[i]) for i in range(lens_positions.shape[0])]  # noqa: F841

        source_position = np.array([
            light_curve_df['source_x'].to_numpy(),
            light_curve_df['source_y'].to_numpy()
        ])
        source_position_tref = [np.interp(t_ref, times, source_position[i]) for i in range(source_position.shape[0])]  # noqa: F841
        magnified_source_tref = np.array([
            light_curve_df['measured_relative_flux'].to_numpy()
        ])
        magnified_source_tref = np.interp(t_ref, times, magnified_source_tref)  # noqa: F841

        shifts = np.vstack([shift_x_vals, shift_y_vals]).T
        plt.figure(figsize=(10, 5))
        plt.quiver(
            np.arange(len(shifts)),
            np.zeros_like(shifts[:, 0]),
            shifts[:, 0],
            shifts[:, 1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color='tab:blue',
            width=0.004,
        )
        plt.scatter([np.interp(t_ref, times, np.arange(len(times)))], [0], color='red', label='t_ref')
        plt.title('Astrometric Shifts (Centroid Offset Vectors)')
        plt.xlabel('Epoch Index')
        plt.ylabel('Shift (angular units)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()