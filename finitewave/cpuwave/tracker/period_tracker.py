from pathlib import Path
import numpy as np
from .local_activation_time_tracker import LocalActivationTimeTracker

try:
    import pandas as pd  # optional
except ImportError:  # pandas is optional
    pd = None



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
    @property
    def output(self):
        """
        Computed activation periods.

        Returns
        -------
        If pandas is installed:
            pd.DataFrame (each row: detector, each column: consecutive period)
        If pandas is not installed:
            list[np.ndarray] (periods per detector)
        """
        lats = np.array(self.act_t, dtype=float)  # shape (n_layers, n_det)
        # Compute periods per detector without pandas:
        periods_list = []
        for i in range(lats.shape[1]):  # detectors
            t = lats[:, i]
            t = t[t != -1]
            periods_list.append(np.diff(t))

        if pd is None:
            return periods_list

        # Keep old “DataFrame of variable-length rows” behavior:
        max_len = max((len(p) for p in periods_list), default=0)
        data = np.full((len(periods_list), max_len), np.nan, dtype=float)
        for i, p in enumerate(periods_list):
            if len(p):
                data[i, :len(p)] = p
        return pd.DataFrame(data)

    def write(self):
        """
        Saves the computed activation periods to a CSV file.

        Always works (with or without pandas).
        """
        out_path = Path(self.path, self.file_name).with_suffix(".csv")

        periods = self.output

        if pd is not None and hasattr(periods, "to_csv"):
            periods.to_csv(out_path, index=False)
            return

        # No pandas: write ragged list to CSV (NaN-padded)
        max_len = max((len(p) for p in periods), default=0)
        data = np.full((len(periods), max_len), np.nan, dtype=float)
        for i, p in enumerate(periods):
            if len(p):
                data[i, :len(p)] = p

        # header: period_0, period_1, ...
        header = ",".join([f"period_{k}" for k in range(max_len)])
        np.savetxt(out_path, data, delimiter=",", header=header, comments="")