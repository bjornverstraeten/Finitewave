import numpy as np
from numba import njit, prange
from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)

from finitewave.cpuwave2D.model._registry import load_ops
from finitewave.cpuwave2D.model._jitwrap import wrap_calc

ops = load_ops("luo_rudy_91")
jit_ops = wrap_calc(ops)

calc_rhs  = jit_ops["calc_rhs"]
calc_dm   = jit_ops["calc_dm"]
calc_dh   = jit_ops["calc_dh"]
calc_dj   = jit_ops["calc_dj"]
calc_dd   = jit_ops["calc_dd"]
calc_df   = jit_ops["calc_df"]
calc_dx   = jit_ops["calc_dx"]
calc_dcai = jit_ops["calc_dcai"]
calc_ina  = jit_ops["calc_ina"]
calc_isk  = jit_ops["calc_isk"]
calc_ik   = jit_ops["calc_ik"]
calc_ik1  = jit_ops["calc_ik1"]
calc_ikp  = jit_ops["calc_ikp"]
calc_ib   = jit_ops["calc_ib"]



class LuoRudy912D(CardiacModel):
    """
    Implements the Luo-Rudy 1991 ventricular action potential model.

    This biophysically detailed model simulates the ionic currents and membrane potential 
    of a ventricular cardiac cell based on Hodgkin-Huxley-type formalism. It was one of 
    the first to incorporate realistic ionic channel kinetics, calcium dynamics, and 
    multiple potassium currents to reproduce key phases of the action potential.

    The model includes:
    - Fast Na⁺ current (I_Na)
    - Slow inward Ca²⁺ current (I_Si)
    - Time-dependent K⁺ current (I_K)
    - Time-independent K⁺ current (I_K1)
    - Plateau K⁺ current (I_Kp)
    - Background/leak current (I_b)

    Attributes
    ----------
    state_vars : list of str
        List of state variable names to save and restore (`u`, `m`, `h`, `j`, `d`, `f`, `x`, `cai`).
    D_model : float
        Diffusion coefficient representing electrical conductivity in the medium (typically set to 0.1).
    gna, gsi, gk, gk1, gkp, gb : float
        Maximum conductances for Na⁺, Ca²⁺, K⁺, and background channels [mS/μF].
    ko, ki, nao, nai, cao : float
        Ion concentrations in mM (extracellular and intracellular for Na⁺, K⁺, Ca²⁺).
    R, T, F : float
        Physical constants: gas constant, temperature in Kelvin, and Faraday constant.
    PR_NaK : float
        Sodium/potassium permeability ratio (used in reversal potential calculation for I_K).

    Paper
    -----
    Luo CH, Rudy Y. 
    A model of the ventricular cardiac action potential. Depolarization, repolarization, and their interaction. 
    Circ Res. 1991 Jun;68(6):1501-26. 
    doi: 10.1161/01.res.68.6.1501. 
    PMID: 1709839.

    """

    def __init__(self):
        """
        Initializes the LuoRudy912D instance, setting up the state variables and parameters.
        """
        CardiacModel.__init__(self)
        self.D_model = 0.1

        self.m   = np.ndarray
        self.h   = np.ndarray
        self.j   = np.ndarray
        self.d   = np.ndarray
        self.f   = np.ndarray
        self.x   = np.ndarray
        self.cai = np.ndarray
        
        self.state_vars = ["u", "m", "h", "j", "d", "f", "x", "cai"]
        self.npfloat = 'float64'

        # model parameters
        parameters = ops.get_parameters()
        self.gna  = parameters["gna"]
        self.gsi  = parameters["gsi"]
        self.gk   = parameters["gk"]
        self.gk1  = parameters["gk1"]
        self.gkp  = parameters["gkp"]
        self.gb   = parameters["gb"]
        self.ko   = parameters["ko"]
        self.ki   = parameters["ki"]
        self.nao  = parameters["nao"]
        self.nai  = parameters["nai"]
        self.cao  = parameters["cao"]
        self.R    = parameters["R"]
        self.T    = parameters["T"]
        self.F    = parameters["F"]
        self.PR_NaK = parameters["PR_NaK"]
        self.E_Na   = parameters["E_Na"]
        self.E_K1   = parameters["E_K1"]

        # initial conditions
        variables = ops.get_variables()
        self.init_u = variables["u"]
        self.init_m = variables["m"]
        self.init_h = variables["h"]
        self.init_j = variables["j"]
        self.init_d = variables["d"]
        self.init_f = variables["f"]
        self.init_x = variables["x"]
        self.init_cai = variables["cai"]

    def initialize(self):
        """
        Initializes the state variables.

        This method sets the initial values for the membrane potential ``u``,
        gating variables ``m``, ``h``, ``j``, ``d``, ``f``, ``x``,
        and intracellular calcium concentration ``cai``.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u     = self.init_u * np.ones(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()

        self.m   = self.init_m * np.ones(shape, dtype=self.npfloat)
        self.h   = self.init_h * np.ones(shape, dtype=self.npfloat)
        self.j   = self.init_j * np.ones(shape, dtype=self.npfloat)
        self.d   = self.init_d * np.ones(shape, dtype=self.npfloat)
        self.f   = self.init_f * np.ones(shape, dtype=self.npfloat)
        self.x   = self.init_x * np.ones(shape, dtype=self.npfloat)
        self.cai = self.init_cai * np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel to update the state variables and membrane
        potential.
        """
        ionic_kernel_2d(self.u_new, self.u, 
                        self.m, self.h, self.j, self.d,self.f, self.x, self.cai, 
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
            return IsotropicStencil2D()

        return AsymmetricStencil2D()


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, m, h, j_, d, f, x, cai, indexes, dt, gna, gsi, gk, gk1, gkp, gb, ko, ki, nai, nao, cao, R, T, F, PR_NaK, E_Na, E_K1):
    """
    Computes the ionic currents and updates the state variables in the 2D
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
        Array for the gating variable `j_`.
    d : np.ndarray
        Array for the gating variable `d`.
    f : np.ndarray
        Array for the gating variable `f`.
    x : np.ndarray
        Array for the gating variable `x`.
    cai : np.ndarray
        Array for the intracellular calcium concentration.
    indexes : np.ndarray
        Array of indexes where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """

    E_Na = (R*T/F)*np.log(nao/nai)

    n_j = u.shape[1]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        # Fast sodium current:
        m[i, j]  += dt*calc_dm(u[i, j], m[i, j])
        h[i, j]  += dt*calc_dh(u[i, j], h[i, j])
        j_[i, j] += dt*calc_dj(u[i, j], j_[i, j])

        ina = calc_ina(u[i, j], m[i, j], h[i, j], j_[i, j], E_Na, gna)

        # Slow inward current:
        d[i, j] += dt*calc_dd(u[i, j], d[i, j])
        f[i, j] += dt*calc_df(u[i, j], f[i, j])

        isi = calc_isk(u[i, j], d[i, j], f[i, j], cai[i, j], gsi)

        cai[i, j] += dt*calc_dcai(cai[i, j], isi)

        # Time-dependent potassium current:
        x[i, j] += dt*calc_dx(u[i, j], x[i, j])

        ik = calc_ik(u[i, j], x[i, j], ko, ki, nao, nai, PR_NaK, R, T, F, gk)

        # Time-independent potassium current:
        ik1 = calc_ik1(u[i, j], ko, E_K1, gk1)

        # Plateau potassium current:
        ikp = calc_ikp(u[i, j], E_K1, gkp)

        # Background current:
        ib = calc_ib(u[i, j], gb)

        u_new[i, j] += dt*calc_rhs(ina, isi, ik, ik1, ikp, ib)




