import numpy as np
from finitewave.core.stimulation.stim_current import StimCurrent

class StimCurrentCoord(StimCurrent):
    """
    A class that applies a stimulation current to a rectangular region of a
    cardiac tissue model.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    curr_value : float
        The value of the stimulation current.
    duration : float
        The duration of the stimulation.
    x1 : int
        The x-coordinate of the lower-left corner of the rectangular region.
    x2 : int
        The x-coordinate of the upper-right corner of the rectangular region.
    y1 : int
        The y-coordinate of the lower-left corner of the rectangular region.
    y2 : int
        The y-coordinate of the upper-right corner of the rectangular region.
    z1 : int
        The z-coordinate of the lower-left corner of the rectangular region.
    z2 : int
        The z-coordinate of the upper-right corner of the rectangular region.
    u_max : float, optional
        The maximum value of the membrane potential. Default is None.
    """

    def __init__(self, time, curr_value, duration, x1, x2, y1, y2, z1=None, z2=None,
                 u_max=None):
        """
        Initializes the StimCurrentCoord instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        curr_value : float
            The value of the stimulation current.
        duration : float
            The duration of the stimulation.
        x1, x2, y1, y2, z1, z2 : int
            The coordinates of the rectangular region to which the stimulation
            current is applied.
        u_max : float, optional
            The maximum value of the membrane potential. Default is None.
        """
        super().__init__(time, curr_value, duration)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.u_max = u_max

        if (z1 is None) ^ (z2 is None):
            raise ValueError("Either provide both z1 and z2, or neither (for 2D).")


    def stimulate(self, model):
        """
        Applies the stimulation current to the specified rectangular region of
        the cardiac tissue model.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the stimulation current is 
            applied.
        """
        u = model.u
        mesh = model.cardiac_tissue.mesh
        if mesh.shape != u.shape:
            raise ValueError("mesh and u shapes differ")

        if u.ndim == 2:
            roi_mesh = mesh[self.x1:self.x2, self.y1:self.y2]
            mask = (roi_mesh == 1)

            roi_u = u[self.x1:self.x2, self.y1:self.y2]
            roi_u[mask] += model.dt * self.curr_value

            if self.u_max is not None:
                roi_u[mask] = np.minimum(roi_u[mask], self.u_max)

        elif u.ndim == 3:
            if self.z1 is None or self.z2 is None:
                raise ValueError("For 3D you must provide z1 and z2.")

            roi_mesh = mesh[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2]
            mask = (roi_mesh == 1)

            roi_u = u[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2]
            roi_u[mask] += model.dt * self.curr_value

            if self.u_max is not None:
                roi_u[mask] = np.minimum(roi_u[mask], self.u_max)
        else:
            raise ValueError(f"Unsupported 'u' ndim={u.ndim}")