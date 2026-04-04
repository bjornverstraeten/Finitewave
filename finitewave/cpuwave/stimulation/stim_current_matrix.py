import numpy as np
from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentMatrix(StimCurrent):
    """
    A class that applies a stimulation current to a cardiac tissue model
    based on a binary matrix.

    Attributes
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    matrix : numpy.ndarray
        A binary matrix indicating the region of interest for stimulation. 
        Elements greater than 0 represent regions to be stimulated.
    
    """
    def __init__(self, time, curr_value, duration, matrix, u_max=None):
        """
        Initializes the StimCurrentMatrix instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        matrix : numpy.ndarray
            A binary matrix indicating the region of interest for
            stimulation.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration)
        self.matrix = matrix
        self.u_max = u_max

    def stimulate(self, model):
        """
        Applies the stimulation current to the cardiac tissue model based on
        the specified binary matrix.

        The stimulation is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : CardiacModel
            The cardiac tissue model.
        """
        if self.matrix.shape != model.u.shape:
            raise ValueError(f"matrix shape {self.matrix.shape} != u shape {model.u.shape}")
        if model.cardiac_tissue.mesh.shape != model.u.shape:
            raise ValueError("mesh and u shapes differ")

        mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
        model.u[mask] += model.dt * self.curr_value

        if self.u_max is not None:
            model.u[mask] = np.where(model.u[mask] > self.u_max, self.u_max,
                                     model.u[mask])
