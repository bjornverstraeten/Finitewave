import numpy as np
from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentArea(StimCurrent):
    """
    A class that applies a stimulation current to a cardiac tissue model
    based on a area coords.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    coords : numpy.ndarray
        The coordinates of the area to be stimulated.
    u_max : float
        The maximum value of the membrane potential.
    """

    def __init__(self, time, curr_value, duration, coords=None, u_max=None):
        """
        Initializes the StimCurrentArea instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        coords : numpy.ndarray
            The coordinates of the area to be stimulated.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration)
        self.coords = None if coords is None else np.asarray(coords, dtype=np.int64)
        self.u_max = u_max

        self._coords = None  # validated tissue coords

    def add_stim_point(self, coord, mesh, size=None):
        """
        Adds an stimulation point to the area to be stimulated.

        Parameters
        ----------
        coord : numpy.ndarray
            The coordinates of the stimulation point.
        mesh : numpy.ndarray
            The mesh of the cardiac tissue model.
        size : float, optional
            The size of the area to be stimulated. Default is None.
        """
        coord = np.asarray(coord, dtype=np.float64).reshape(-1)
        if coord.shape[0] != mesh.ndim:
            raise ValueError(f"coord has dim={coord.shape[0]} but mesh.ndim={mesh.ndim}")

        if size is None:
            self.coords = np.atleast_2d(coord).astype(np.int64)
            return

        tissue_points = np.argwhere(mesh == 1).astype(np.float64)  # (n, ndim)
        dist = np.linalg.norm(tissue_points - coord[None, :], axis=1)
        self.coords = tissue_points[dist < float(size)].astype(np.int64)

    def initialize(self, model):
        if self.coords is None:
            raise ValueError("coords must be provided (or call add_stim_point before initialize).")

        coords = np.asarray(self.coords)
        coords = np.atleast_2d(coords)

        if coords.shape[1] != model.u.ndim:
            raise ValueError(
                f"coords has ndim={coords.shape[1]} but model.u.ndim={model.u.ndim}. "
                f"Provide coords with {model.u.ndim} columns."
            )

        # keep only points that are healthy tissue
        inds = tuple(coords.T)
        mask = (model.cardiac_tissue.mesh[inds] == 1)

        if mask.sum() == 0:
            raise ValueError("The specified area does not have healthy cells.")

        self._coords = coords[mask].astype(np.int64)
        return super().initialize(model)

    def stimulate(self, model):
        inds = tuple(self._coords.T)
        model.u[inds] += model.dt * self.curr_value

        if self.u_max is not None:
            model.u[inds] = np.minimum(model.u[inds], self.u_max)