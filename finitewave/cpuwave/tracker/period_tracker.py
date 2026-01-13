from pathlib import Path
import numpy as np
import pandas as pd
import json
from .local_activation_time_tracker import LocalActivationTimeTracker


class PeriodTracker(LocalActivationTimeTracker):
    """
    A class to track activation periods of cells in a cardiac tissue model
    using detectors.

    Attributes
    ----------
    cell_ind : list or list of lists with two indices
        The indices [i, j] of the cell where the variables are tracked.
        List of lists can be used to track multiple cells.
    file_name : str
        The name of the file to save the computed activation periods.
    """

    def __init__(self):
        super().__init__()
        self.cell_ind = []
        self.file_name = "period"
        self._inds = None  # cached indices (n_cells, ndim)

    def initialize(self, model):
        super().initialize(model)

        inds = np.asarray(self.cell_ind, dtype=int)
        inds = np.atleast_2d(inds)

        if inds.size == 0:
            raise ValueError("cell_ind must contain at least one detector coordinate.")

        if inds.shape[1] != model.u.ndim:
            raise ValueError(
                f"cell_ind has ndim={inds.shape[1]} but model.u.ndim={model.u.ndim}. "
                f"Provide indices with {model.u.ndim} coordinates."
            )

        self._inds = inds
        n_det = inds.shape[0]

        # We only care about detector cells, so act_t layers are 1D (n_det,)
        self.act_t = [-np.ones(n_det, dtype=float)]

    def _track(self):
        cross = self.cross_threshold()

        # restrict to detector locations
        coord = tuple(self._inds[:, d] for d in range(self._inds.shape[1]))
        cross_det = cross[coord]  # shape (n_det,)

        # start a new layer if the current layer already has something and new crossings appear
        if np.any(self.act_t[-1] > -1) and np.any(cross_det):
            self.act_t.append(-np.ones_like(self.act_t[-1]))

        self.act_t[-1][cross_det] = self.model.t

    @property
    def output(self):
        lats = np.asarray(self.act_t).T  # (n_det, n_layers)

        # periods per detector: diffs between successive non -1 times
        periods = []
        for row in lats:
            t = row[row != -1]
            periods.append(np.diff(t) if len(t) >= 2 else np.array([], dtype=float))

        # keep as a series-of-arrays like before, but in a dataframe for easy CSV
        return pd.DataFrame({"periods": periods})

    def write(self):
        self.output.to_csv(Path(self.path, self.file_name).with_suffix(".csv"), index=False)