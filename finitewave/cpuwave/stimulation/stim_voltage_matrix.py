from finitewave.core.stimulation.stim_voltage import StimVoltage


class StimVoltageMatrix(StimVoltage):
    """
    A class that applies a voltage stimulus to a cardiac tissue model
    according to a specified matrix.
    """
    def __init__(self, time, volt_value, matrix):
        """
        Initializes the StimVoltageMatrix instance.

        Parameters
        ----------
        time : float
            The time at which the stimulation starts.
        volt_value : float
            The voltage value to apply.
        matrix : numpy.ndarray
            A 2D array where the voltage stimulus is applied to locations with
            values greater than 0.
        """
        super().__init__(time, volt_value)
        self.matrix = matrix

    def stimulate(self, model):
        """
        Applies the voltage stimulus to the cardiac tissue model based on the
        specified matrix.

        The voltage is applied only if the current time is within the
        stimulation period and the stimulation has not been previously applied.

        Parameters
        ----------
        model : CardiacModel
            The 2D cardiac tissue model.

        Notes
        -----
        The voltage value is applied to the positions in the cardiac tissue
        where the corresponding value in ``matrix`` is greater than 0,
        and the ``model.cardiac_tissue.mesh`` value is 1.
        """
        if self.matrix.shape != model.u.shape:
            raise ValueError(f"matrix shape {self.matrix.shape} != u shape {model.u.shape}")
        if model.cardiac_tissue.mesh.shape != model.u.shape:
            raise ValueError("mesh and u shapes differ")
        
        mask = (self.matrix > 0) & (model.cardiac_tissue.mesh == 1)
        model.u[mask] = self.volt_value
