from pathlib import Path
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class VariablesTracker(Tracker):
    """
    A class to track multiple variables at a specific cell in a cardiac
    tissue model simulation.

    This tracker monitors user-defined variables at a specified cell index and
    records their values over time.

    Attributes
    ----------
    var_list : list of str
        A list of variable names to be tracked.
    cell_ind : list or list of lists with two indices
        The indices [i, j] of the cell where the variables are tracked.
        List of lists can be used to track multiple cells.
    dir_name : str
        The directory name where tracked variables are saved.
    vars : dict
        A dictionary where each key is a variable name, and the value is
        an array of its tracked values over time.

    """

    def __init__(self):
        """
        Initializes the MultiVariableTracker with default parameters.
        """
        Tracker.__init__(self)
        self.var_list = []  # List of variables to be tracked
        self.cell_ind = [1, 1]  # Cell index to track variables
        self.dir_name = "multi_vars"  # Directory to save tracked variables
        self.vars = {}  # Dictionary to store tracked variables
        self._inds = None  # cached indices

    def initialize(self, model):
        """
        Initializes the tracker with the simulation model and precomputes
        necessary values for each variable.

        Parameters
        ----------
        model : object
            The cardiac tissue model object containing the data to be tracked.
        """
        self.model = model
        self.vars = {}

        # Validate indices against model.u.ndim
        inds = np.asarray(self.cell_ind, dtype=int)
        inds = np.atleast_2d(inds)

        if inds.shape[1] != model.u.ndim:
            raise ValueError(
                f"cell_ind has ndim={inds.shape[1]} but model.u.ndim={model.u.ndim}. "
                f"Provide indices with {model.u.ndim} coordinates."
            )
        self._inds = inds

        # Prepare storage + validate variables exist
        for var_ in self.var_list:
            if not hasattr(self.model, var_):
                raise ValueError(f"Variable '{var_}' not found in model.")
            self.vars[var_] = []

    def _track(self):
        """
        Tracks and stores the values of each specified variable at each time step.

        This method should be called at each time step of the simulation.
        """
        # Track the value of each variable at the specified cell index
        # Make possible to track multiple cells
        coord = tuple(self._inds[:, d] for d in range(self._inds.shape[1]))

        for var_ in self.var_list:
            arr = getattr(self.model, var_)

            # Enforce grid-shaped variables (recommended)
            if not hasattr(arr, "ndim") or arr.ndim != self.model.u.ndim:
                raise ValueError(
                    f"Variable '{var_}' must be a grid field with ndim={self.model.u.ndim}, "
                    f"got type={type(arr)} ndim={getattr(arr, 'ndim', None)}"
                )

            self.vars[var_].append(arr[coord])

    @property
    def output(self):
        """
        Returns the tracked variables data.

        Returns
        -------
        dict
            A dictionary where each key is a variable name, and the value is
            an array of its tracked values over time.
        """
        out = {}
        for var_ in self.var_list:
            out[var_] = np.squeeze(np.asarray(self.vars[var_]))
        return out

    def write(self):
        """
        Saves the tracked variables to disk as NumPy files.
        """
        out_dir = Path(self.path, self.dir_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        out = self.output  # compute once
        for var_ in self.var_list:
            np.save(out_dir / f"{var_}.npy", out[var_])
