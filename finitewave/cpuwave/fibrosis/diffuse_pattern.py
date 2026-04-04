import numpy as np
from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class DiffusePattern(FibrosisPattern):
    """
    Class for generating a diffuse fibrosis pattern in a given mesh area.

    Attributes
    ----------
    density : float
        The density of the fibrosis in the specified area
    x1 : int
        The starting x-coordinate of the fibrosis area.
    x2 : int
        The ending x-coordinate of the fibrosis area.
    y1 : int
        The starting y-coordinate of the fibrosis area.
    y2 : int
        The ending y-coordinate of the fibrosis area.
    z1 : int, optional
        The starting z-coordinate of the fibrosis area (for 3D meshes).
    z2 : int, optional
        The ending z-coordinate of the fibrosis area (for 3D meshes).
    """

    def __init__(self, density, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None):
        self.density = float(density)
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2

        if (z1 is None) ^ (z2 is None):
            raise ValueError("Either provide both z1 and z2, or neither.")

    def generate(self, shape=None, mesh=None):
        """
        Generates a diffuse 2D fibrosis pattern for the given shape and mesh.
        The resulting pattern is applied to the mesh within the specified
        area.

        Parameters
        ----------
        shape : tuple
            The shape of the mesh.
        mesh : numpy.ndarray, optional
            The existing mesh to base the pattern on. Default is None.

        Returns
        -------
        numpy.ndarray
            A new mesh array with the applied fibrosis pattern.

        Notes
        -----
        If both parameters are provided, first non-None parameter is used.
        """
        if shape is None and mesh is None:
            raise ValueError("Either shape or mesh must be provided.")

        if mesh is None:
            mesh = np.ones(shape, dtype=np.int8)

        # generate candidate fibrosis mask for whole domain
        fibr = self._generate(mesh.shape)

        roi = self._roi(mesh.ndim)
        mesh[roi] = fibr[roi]
        return mesh

    def _roi(self, ndim):
        if ndim == 2:
            return (slice(self.x1, self.x2), slice(self.y1, self.y2))
        elif ndim == 3:
            zsl = slice(self.z1, self.z2) if self.z1 is not None else slice(None)
            return (slice(self.x1, self.x2), slice(self.y1, self.y2), zsl)
        else:
            raise ValueError(f"Unsupported ndim={ndim}. Expected 2 or 3.")

    def _generate(self, shape):
        # 1 = healthy, 2 = fibrotic (as in your original)
        return 1 + (np.random.random(shape) <= self.density).astype(np.int8)
