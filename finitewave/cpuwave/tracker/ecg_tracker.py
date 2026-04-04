from pathlib import Path
import numpy as np
from numba import njit, prange

from finitewave.core.tracker.tracker import Tracker


class ECGTracker(Tracker):
    """
    A class to compute and track electrocardiogram (ECG) signals from a
    cardiac tissue model simulation.

    This tracker calculates ECG signals at specified measurement points by
    computing the potential differences across the cardiac tissue mesh and
    considering the inverse of the distance from each measurement point.

    measure_coords: array-like of shape (n_points, 3)
        Measurement points in grid index coordinates (x, y, z).
        For 2D: z is interpreted as height above the 2D plane (k=0).
        For 3D: z is the z-index in the grid.
    """

    def __init__(self, measure_coords=None):
        super().__init__()
        self.measure_coords = measure_coords
        self.ecg = []
        self.file_name = "ecg.npy"
        self.u_tr = None
        self._compute = None  # selected numba kernel

    def initialize(self, model):
        self.model = model

        if self.measure_coords is None:
            raise ValueError("measure_coords must be provided (shape: (n_points, 3)).")

        self.measure_coords = np.asarray(self.measure_coords, dtype=np.float64)
        self.measure_coords = np.atleast_2d(self.measure_coords)

        if self.measure_coords.shape[1] != 3:
            raise ValueError(f"measure_coords must have 3 columns (x,y,z); got {self.measure_coords.shape}.")

        self.ecg = []
        self.u_tr = np.zeros_like(model.u)

        if model.u.ndim == 2:
            self._compute = _compute_ecg_2d
        elif model.u.ndim == 3:
            self._compute = _compute_ecg_3d
        else:
            raise ValueError(f"Unsupported model.u.ndim={model.u.ndim}. Expected 2 or 3.")

    def calc_ecg(self):
        # diffusion update into u_tr
        self.model.diffusion_kernel(
            self.u_tr,
            self.model.u,
            self.model.weights,
            self.model.cardiac_tissue.myo_indexes,
        )

        # compute ecg
        return self._compute(
            self.u_tr,
            self.model.u,
            self.measure_coords,
            float(self.model.dr),
            self.model.cardiac_tissue.myo_indexes,
        )

    def _track(self):
        self.ecg.append(self.calc_ecg())

    @property
    def output(self):
        return np.asarray(self.ecg)

    def write(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        np.save(Path(self.path).joinpath(self.file_name), self.output)


@njit(parallel=True)
def _compute_ecg_2d(u_tr, u, coords, dr, indexes):
    """
    2D: u is (ni, nj). coords is (n_points, 3) -> (x,y,z_height).
    Distance is (x-i)^2 + (y-j)^2 + (z_height)^2

    Parameters
    ----------
    u_tr : numpy.ndarray
        A 2D array to store the updated potential values after diffusion.
    u : numpy.ndarray
        A 2D array representing the current potential values before diffusion.
    coord : tuple
        The coordinates of the measurement point.
    dr : float
        The spatial resolution of the grid.
    indexes : numpy.ndarray
        A 1D array of indices of the healthy tissue points.
    """
    n_j = u.shape[1]
    n_c = coords.shape[0]
    ecg = np.zeros(n_c)

    for c in range(n_c):
        x = coords[c, 0]
        y = coords[c, 1]
        z = coords[c, 2]  # height above plane
        ecg_ = 0.0

        for ind in prange(len(indexes)):
            ii = indexes[ind]
            i = ii // n_j
            j = ii % n_j

            d = (x - i) * (x - i) + (y - j) * (y - j) + z * z
            if d > 0.0:
                ecg_ += (u_tr[i, j] - u[i, j]) / (d * dr)

        ecg[c] = ecg_

    return ecg


@njit(parallel=True)
def _compute_ecg_3d(u_tr, u, coords, dr, indexes):
    """
    3D: u is (ni, nj, nk). coords is (n_points, 3) -> (x,y,z).
    Distance is (x-i)^2 + (y-j)^2 + (z-k)^2

    Parameters
    ----------
    u_tr : numpy.ndarray
        A 3D array to store the updated potential values after diffusion.
    u : numpy.ndarray
        A 3D array representing the current potential values before diffusion.
    coord : tuple
        The coordinates of the measurement point.
    dr : float
        The spatial resolution of the grid.
    indexes : numpy.ndarray
        A 1D array of indices of the healthy tissue points.
    """
    n_j = u.shape[1]
    n_k = u.shape[2]
    n_jk = n_j * n_k

    n_c = coords.shape[0]
    ecg = np.zeros(n_c)

    for c in range(n_c):
        x = coords[c, 0]
        y = coords[c, 1]
        z = coords[c, 2]
        ecg_ = 0.0

        for ind in prange(len(indexes)):
            ii = indexes[ind]
            i = ii // n_jk
            jk = ii - i * n_jk
            j = jk // n_k
            k = jk - j * n_k

            d = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k)
            if d > 0.0:
                ecg_ += (u_tr[i, j, k] - u[i, j, k]) / (d * dr)

        ecg[c] = ecg_

    return ecg
