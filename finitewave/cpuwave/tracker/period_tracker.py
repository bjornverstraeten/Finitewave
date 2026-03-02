from pathlib import Path
import numpy as np

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
        """
        Property to get the computed activation periods.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed activation periods.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "PeriodTracker.output requires pandas.\n\n"
                "Install one of the following:\n"
                "  • pip install finitewave\n"
                "  • pip install pandas\n"
            ) from e

        lats = np.array(self.act_t)
        lats = pd.DataFrame(lats.T)
        periods = lats.apply(lambda row: np.diff(row[row != -1]), axis=1)
        return periods

    def write(self):
        """
        Saves the computed activation periods to a CSV file.
        """
        periods = self.output
        periods.to_csv(Path(self.path, self.file_name).with_suffix(".csv"))