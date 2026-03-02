from pathlib import Path
import csv
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from numba import njit
from numba.typed import List
from finitewave.core.tracker.tracker import Tracker


class SpiralWaveCoreTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.file_name = "spiral_wave_core"
        self.spiral_wave_cores = []
        self.delta = 5
        self.u_prev = None
        self._ndim = None

    def initialize(self, model):
        self.model = model
        self.u_prev = self.model.u.copy()
        self._ndim = self.model.u.ndim
        self.spiral_wave_cores = []

        if self._ndim not in (2, 3):
            raise ValueError(f"Unsupported model.u.ndim={self._ndim}; expected 2 or 3.")

    def track_tip_line(self, u, u_new, threshold):
        return list(_track_tip_line(u, u_new, threshold, self.delta))

    def _track(self):
        t = float(self.model.t)
        step = int(self.model.step)

        if self._ndim == 2:
            tips = self.track_tip_line(self.u_prev, self.model.u, self.threshold)
            for x, y in tips:
                self.spiral_wave_cores.append((int(x), int(y), t, step))

        else:
            nz = self.model.u.shape[2]
            for k in range(nz):
                u_prev = self.u_prev[:, :, k]
                u = self.model.u[:, :, k]
                tips = self.track_tip_line(u_prev, u, self.threshold)
                for x, y in tips:
                    self.spiral_wave_cores.append((int(x), int(y), int(k), t, step))

        self.u_prev = self.model.u.copy()

    @property
    def output(self):
        cols = ["x", "y", "time", "step"] if self._ndim == 2 else ["x", "y", "z", "time", "step"]

        if not self.spiral_wave_cores:
            if pd is not None:
                return pd.DataFrame(columns=cols)
            return np.empty((0, len(cols)))

        if pd is not None:
            return pd.DataFrame(self.spiral_wave_cores, columns=cols)

        return np.asarray(self.spiral_wave_cores)

    def write(self):
        out_path = Path(self.path, self.file_name).with_suffix(".csv")
        cols = ["x", "y", "time", "step"] if self._ndim == 2 else ["x", "y", "z", "time", "step"]

        if pd is not None:
            self.output.to_csv(out_path, index=False)
            return

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(self.spiral_wave_cores)

@njit
def _correct_tip_pos(i, j, u, u_new, threshold):
    AC = u[i, j] - u[i, j + 1] + u[i + 1, j + 1] - u[i + 1, j]
    GC = u[i, j + 1] - u[i, j]
    BC = u[i + 1, j] - u[i, j]
    DC = u[i, j] - threshold

    AD = u_new[i, j] - u_new[i, j + 1] + u_new[i + 1, j + 1] - u_new[i + 1, j]
    GD = u_new[i, j + 1] - u_new[i, j]
    BD = u_new[i + 1, j] - u_new[i, j]
    DD = u_new[i, j] - threshold

    Q = BC * AD - BD * AC
    R = GC * AD - GD * AC
    S = DC * AD - DD * AC

    if R == 0.0:
        return

    QOnR = Q / R
    SOnR = S / R

    T = AC * QOnR
    U = AC * SOnR - BC + GC * QOnR
    V = GC * SOnR - DC

    discriminant = U * U - 4.0 * T * V
    if discriminant < 0.0:
        return

    T2 = 2.0 * T
    if T2 == 0.0:
        return

    xn = (-U - sqrt(discriminant)) / T2
    xp = (-U + sqrt(discriminant)) / T2
    yn = -QOnR * xn - SOnR
    yp = -QOnR * xp - SOnR

    if 0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0:
        return [xn, yn]
    if 0.0 <= xp <= 1.0 and 0.0 <= yp <= 1.0:
        return [xp, yp]
    return


@njit
def _apply_threshold(i, j, u, threshold):
    if (u[i, j] >= threshold and (u[i + 1, j] < threshold or u[i, j + 1] < threshold or u[i + 1, j + 1] < threshold)):
        return 1
    if (u[i, j] < threshold and (u[i + 1, j] >= threshold or u[i, j + 1] >= threshold or u[i + 1, j + 1] >= threshold)):
        return 1
    return 0


@njit
def _track_tip_line(u, u_new, threshold, delta):
    out = List()
    size_i, size_j = u.shape

    # Guard: if domain is too small for delta margin
    if size_i <= 2 * delta + 1 or size_j <= 2 * delta + 1:
        return out

    for i in range(delta, size_i - delta):
        for j in range(delta, size_j - delta):
            counter = _apply_threshold(i, j, u, threshold)
            if counter == 1:
                counter += _apply_threshold(i, j, u_new, threshold)

            if counter == 2:
                correction = _correct_tip_pos(i, j, u, u_new, threshold)
                if correction is not None:
                    out.append([j + correction[1], i + correction[0]])

    return out