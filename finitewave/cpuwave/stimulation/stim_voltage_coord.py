from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageCoord(StimVoltage):
    """
    A class that applies a voltage stimulus to a cardiac tissue model
    within a specified region of interest.

    Parameters
    ----------
    time : float
        The time at which the stimulation starts.
    volt_value : float
        The voltage value to apply to the region of interest.
    x1 : int
        The starting x-coordinate of the region of interest.
    x2 : int
        The ending x-coordinate of the region of interest.
    y1 : int
        The starting y-coordinate of the region of interest.
    y2 : int
        The ending y-coordinate of the region of interest.
    z1 : int
        The starting z-coordinate of the region of interest.
    z2 : int
        The ending z-coordinate of the region of interest.
    """

    def __init__(self, time, volt_value, x1, x2, y1, y2, z1=None, z2=None):
        """
        Initializes the StimVoltageCoord instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        x1, x2, y1, y2, z1, z2 : int
            The coordinates of the region of interest to which the voltage
            stimulus is applied.
        """
        super().__init__(time, volt_value)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

        if (z1 is None) ^ (z2 is None):
            raise ValueError("Either provide both z1 and z2, or neither (for 2D).")       

    def stimulate(self, model):
        """
        Applies the voltage stimulus to the cardiac tissue model within the
        specified region of interest.

        Parameters
        ----------
        model : object
            The cardiac tissue model to which the voltage stimulus is applied.
        """
        u = model.u
        mesh = model.cardiac_tissue.mesh
        if mesh.shape != u.shape:
            raise ValueError("mesh and u shapes differ")

        if u.ndim == 2:
            roi_mesh = mesh[self.x1:self.x2, self.y1:self.y2]
            mask = (roi_mesh == 1)

            roi_u = u[self.x1:self.x2, self.y1:self.y2]
            roi_u[mask] = self.volt_value
        elif u.ndim == 3:
            if self.z1 is None or self.z2 is None:
                raise ValueError("For 3D you must provide z1 and z2.")

            roi_mesh = mesh[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2]
            mask = (roi_mesh == 1)

            roi_u = u[self.x1:self.x2, self.y1:self.y2, self.z1:self.z2]
            roi_u[mask] = self.volt_value
        else:
            raise ValueError(f"Unsupported 'u' ndim={u.ndim}")
