import numpy as np
import random

from finitewave.core.fibrosis.fibrosis_pattern import FibrosisPattern


class StructuralPattern(FibrosisPattern):
    """
    Class for generating a structural fibrosis pattern in a mesh.

    The pattern consists of rectangular blocks distributed throughout a specified region of the mesh,
    with the density controlling the likelihood of each block being present.

    Attributes
    ----------
    density : float
        The density of the fibrosis blocks, represented as a probability.
    length_i : int
        The width of each block.
    length_j : int
        The height of each block.
    x1 : int
        The starting x-coordinate of the area where blocks can be placed.
    x2 : int
        The ending x-coordinate of the area where blocks can be placed.
    y1 : int
        The starting y-coordinate of the area where blocks can be placed.
    y2 : int
        The ending y-coordinate of the area where blocks can be placed.
    z1 : int, optional
        The starting z-coordinate of the area where blocks can be placed (for 3D meshes).
    z2 : int, optional
        The ending z-coordinate of the area where blocks can be placed (for 3D meshes).
    length_k : int, optional
        The depth of each block (for 3D meshes).
    """

    def __init__(
        self,
        density,
        length_i,
        length_j,
        x1,
        x2,
        y1,
        y2,
        z1=None,
        z2=None,
        length_k=None,
    ):
        self.density = float(density)

        self.length_i = int(length_i)
        self.length_j = int(length_j)
        self.length_k = None if length_k is None else int(length_k)

        self.x1, self.x2 = int(x1), int(x2)
        self.y1, self.y2 = int(y1), int(y2)
        self.z1, self.z2 = z1, z2

        if (z1 is None) ^ (z2 is None):
            raise ValueError("Either provide both z1 and z2, or neither.")

    def generate(self, shape=None, mesh=None):
        """
        Generates and applies a structural fibrosis pattern to the mesh.

        The mesh is divided into blocks of size `length_i` x `length_j` x `length_k`, with each block having 
        a probability `density` of being filled with fibrosis. The function ensures that blocks do not
        extend beyond the specified region.

        Parameters
        ----------
        shape : tuple
            The shape of the mesh.
        mesh : numpy.ndarray, optional
            The existing mesh to base the pattern on. Default is None..

        Returns
        -------
        numpy.ndarray
            A new mesh array with the applied fibrosis pattern.
        """
        if shape is None and mesh is None:
            raise ValueError("Either shape or mesh must be provided.")

        if mesh is None:
            mesh = np.ones(shape, dtype=np.int8)

        ndim = mesh.ndim
        if ndim == 2:
            self._apply_2d(mesh)
        elif ndim == 3:
            # for true 3D blocks, z-range and length_k must be provided
            if self.z1 is None or self.z2 is None:
                raise ValueError("For 3D mesh you must provide z1 and z2.")
            if self.length_k is None:
                raise ValueError("For 3D mesh you must provide length_k.")
            self._apply_3d(mesh)
        else:
            raise ValueError(f"Unsupported mesh.ndim={ndim}. Expected 2 or 3.")

        return mesh

    def _apply_2d(self, mesh):
        for i in range(self.x1, self.x2, self.length_i):
            i_s = min(self.length_i, self.x2 - i)
            for j in range(self.y1, self.y2, self.length_j):
                j_s = min(self.length_j, self.y2 - j)
                if random.random() <= self.density:
                    mesh[i:i + i_s, j:j + j_s] = 2

    def _apply_3d(self, mesh):
        z1, z2 = int(self.z1), int(self.z2)
        for i in range(self.x1, self.x2, self.length_i):
            i_s = min(self.length_i, self.x2 - i)
            for j in range(self.y1, self.y2, self.length_j):
                j_s = min(self.length_j, self.y2 - j)
                for k in range(z1, z2, self.length_k):
                    k_s = min(self.length_k, z2 - k)
                    if random.random() <= self.density:
                        mesh[i:i + i_s, j:j + j_s, k:k + k_s] = 2