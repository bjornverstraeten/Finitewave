import numpy as np
from numba import njit, prange

from finitewave.cpuwave2D.model.luo_rudy91_2d import (
    LuoRudy912D, calc_dm, calc_dh, calc_dj, calc_dd, calc_df, calc_dx,
    calc_dcai, calc_ina, calc_isk, calc_ik, calc_ik1, calc_ikp, calc_ib,
    calc_rhs
)
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import (
    IsotropicStencil3D
)
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)


class LuoRudy913D(LuoRudy912D):
    """
    Implements the 3D Luo-Rudy 1991 cardiac model.

    See LuoRudy912D for the 2D model description.
    """
    def __init__(self):
        super().__init__()

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel to update the state variables and membrane
        potential.
        """
        ionic_kernel_3d(self.u_new, self.u, self.m, self.h, self.j, self.d,
                        self.f, self.x, self.cai,
                        self.cardiac_tissue.myo_indexes, self.dt, 
                        self.gna, self.gsi, self.gk, self.gk1, self.gkp, self.gb, 
                        self.ko, self.ki, self.nai, self.nao, self.cao, 
                        self.R, self.T, self.F, self.PR_NaK, self.E_Na, self.E_K1)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue2D
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            return IsotropicStencil3D()

        return AsymmetricStencil3D()


@njit(parallel=True)
def ionic_kernel_3d(u_new, u, m, h, j_, d, f, x, cai, indexes, dt, gna, gsi, gk, gk1, gkp, gb, ko, ki, nai, nao, cao, R, T, F, PR_NaK, E_Na, E_K1):
    """
    Computes the ionic currents and updates the state variables in the 3D
    Luo-Rudy 1991 cardiac model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated membrane potential.
    u : np.ndarray
        Array of the current membrane potential values.
    m : np.ndarray
        Array for the gating variable `m`.
    h : np.ndarray
        Array for the gating variable `h`.
    j_ : np.ndarray
        Array for the gating variable `j`.
    d : np.ndarray
        Array for the gating variable `d`.
    f : np.ndarray
        Array for the gating variable `f`.
    x : np.ndarray
        Array for the gating variable `x`.
    Cai_c : np.ndarray
        Array for the intracellular calcium concentration.
    mesh : np.ndarray
        Mesh array indicating the tissue types.
    dt : float
        Time step for the simulation.
    """

    E_Na = (R*T/F)*np.log(nao/nai)

    n_j = u.shape[1]
    n_k = u.shape[2]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k

        # Fast sodium current:
        m[i, j, k]  += dt*calc_dm(u[i, j, k], m[i, j, k])
        h[i, j, k]  += dt*calc_dh(u[i, j, k], h[i, j, k])
        j_[i, j, k] += dt*calc_dj(u[i, j, k], j_[i, j, k])

        ina = calc_ina(u[i, j, k], m[i, j, k], h[i, j, k], j_[i, j, k], E_Na, gna)

        # Slow inward current:
        d[i, j, k] += dt*calc_dd(u[i, j, k], d[i, j, k])
        f[i, j, k] += dt*calc_df(u[i, j, k], f[i, j, k])

        isi = calc_isk(u[i, j, k], d[i, j, k], f[i, j, k], cai[i, j, k], gsi)

        cai[i, j, k] += dt*calc_dcai(cai[i, j, k], isi)

        # Time-dependent potassium current:
        x[i, j, k] += dt*calc_dx(u[i, j, k], x[i, j, k])

        ik = calc_ik(u[i, j, k], x[i, j, k], ko, ki, nao, nai, PR_NaK, R, T, F, gk)

        # Time-independent potassium current:
        ik1 = calc_ik1(u[i, j, k], ko, E_K1, gk1)

        # Plateau potassium current:
        ikp = calc_ikp(u[i, j, k], E_K1, gkp)

        # Background current:
        ib = calc_ib(u[i, j, k], gb)

        u_new[i, j, k] += dt*calc_rhs(ina, isi, ik, ik1, ikp, ib)
