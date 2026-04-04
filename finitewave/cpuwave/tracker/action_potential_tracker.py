import numpy as np

from finitewave.core.tracker.tracker import Tracker


class ActionPotentialTracker(Tracker):
    """
    A class to track and record the action potential of a specific cell in
    a cardiac tissue model.

    This tracker monitors the membrane potential of a single cell at each time
    step and stores the data in an array for later analysis or visualization.

    Attributes
    ----------
    act_pot : np.ndarray
        Array to store the action potential values at each time step.
    cell_ind : list or list of lists with two indices
        Coordinates of the cell to be tracked in the 2D model grid.
    file_name : str
        Name of the file where the tracked action potential data will be saved.
    """

    def __init__(self):
        """
        Initializes the ActionPotentialTracker with default parameters.
        """
        Tracker.__init__(self)
        self.act_pot = []           # Initialize the array to store action potential
        self.cell_ind = [1, 1]      # user can set [i,j] or [i,j,k] or list of those:
        self.file_name = "act_pot"  # Default file name for saving data

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model, setting up
        the action potential array.

        Parameters
        ----------
        model : object
            The cardiac tissue model object that contains simulation parameters
            like `t_max` (maximum time) and `dt` (time step).
        """
        self.model = model

        # Better to validate in initialize than it will fail in _track during calculations.
        inds = np.asarray(self.cell_ind, dtype=int)
        inds = np.atleast_2d(inds)

        # If user provided a flat list like [i,j,k], np.atleast_2d - (1, ndim) ok
        # If user provided list of points - (n, ndim) ok
        if inds.shape[1] != model.u.ndim:
            raise ValueError(
                f"cell_ind has ndim={inds.shape[1]} but model.u.ndim={model.u.ndim}. "
                f"Provide indices with {model.u.ndim} coordinates."
            )

        self._inds = inds  # cache validated indices

    def _track(self):
        """
        Records the action potential (`u`) of the specified cell at the current
        time step.
        """
        # Make possible to track multiple cells
        coord = tuple(self._inds[:, d] for d in range(self._inds.shape[1]))
        self.act_pot.append(self.model.u[coord])

    @property
    def output(self):
        """
        Returns the tracked action potential data.

        Returns
        -------
        np.ndarray
            The array containing the tracked action potential values.
        """
        return np.squeeze(self.act_pot)
