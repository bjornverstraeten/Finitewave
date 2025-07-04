import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class AlievPanfilov2D(CardiacModel):
    """
    Two-dimensional implementation of the Aliev–Panfilov model of cardiac excitation.

    The Aliev–Panfilov model is a phenomenological two-variable model designed to
    reproduce basic features of cardiac excitation, including wave propagation and
    reentry, while remaining computationally efficient. It uses a single recovery
    variable coupled with a cubic nonlinearity to simulate action potential dynamics
    in excitable media.

    Attributes
    ----------
    u : np.ndarray
        Transmembrane potential (dimensionless, normalized to [0,1]).
    v : np.ndarray
        Recovery variable describing refractoriness.
    D_model : float
        Diffusion coefficient used for simulating spatial propagation.
    state_vars : list of str
        Names of the state variables to be saved and restored.
    npfloat : str
        Floating-point precision used in the simulation (default: 'float64').

    Model Parameters
    ----------------
    a : float
        Excitability threshold parameter.
    k : float
        Strength of the nonlinear source term (governs spike shape).
    eap : float
        Baseline recovery rate.
    mu_1 : float
        Recovery rate coefficient (scales v feedback).
    mu_2 : float
        Recovery rate offset (modulates u-dependence of recovery).

    Paper
    -----
    Rubin R. Aliev, Alexander V. Panfilov,
    A simple two-variable model of cardiac excitation,
    Chaos, Solitons & Fractals,
    Volume 7, Issue 3,
    1996,
    Pages 293-301,
    ISSN 0960-0779,
    https://doi.org/10.1016/0960-0779(95)00089-5.

    Attributes
    ----------
    v : np.ndarray
        Array for the recovery variable.
    w : np.ndarray
        Array for diffusion weights.
    D_model : float
        Model specific diffusion coefficient
    state_vars : list
        List of state variables to be saved and restored.
    npfloat : str
        Data type used for floating-point operations, default is 'float64'.
    """

    def __init__(self):
        """
        Initializes the AlievPanfilov2D instance with default parameters.
        """
        super().__init__()
        self.v = np.ndarray
        
        self.D_model = 1.
    
        self.state_vars = ["u", "v"]
        self.npfloat    = 'float64'

        # model parameters
        self.a    = 0.1
        self.k    = 8.0
        self.eap  = 0.01
        self.mu_1 = 0.2
        self.mu_2 = 0.3

        # initial conditions
        self.init_u = 0.0
        self.init_v = 0.0

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()
        self.u = self.init_u * np.ones_like(self.u, dtype=self.npfloat)
        self.v = self.init_v * np.ones_like(self.u, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        ionic_kernel_2d(self.u_new, self.u, self.v, self.cardiac_tissue.myo_indexes, self.dt, 
                        self.a, self.k, self.eap, self.mu_1, self.mu_2)

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

@njit
def calc_v(v, u, dt, a, k, eap, mu_1, mu_2):
    """
    Computes the update of the recovery variable v for the Aliev–Panfilov model.

    This function implements the ordinary differential equation governing the
    evolution of the recovery variable `v`, which models the refractoriness of
    the cardiac tissue. The rate of recovery depends on both `v` and `u`, with a
    nonlinear interaction term involving a cubic expression in `u`.

    Parameters
    ----------
    v : float
        Current value of the recovery variable.
    u : float
        Current value of the transmembrane potential.
    dt : float
        Time step for integration.
    a : float
        Excitability threshold.
    k : float
        Strength of the nonlinear source term.
    eap : float
        Baseline recovery rate.
    mu_1 : float
        Recovery scaling parameter.
    mu_2 : float
        Offset parameter for recovery rate.

    Returns
    -------
    float
        Updated value of the recovery variable `v`.
    """

    v += (- dt * (eap + (mu_1 * v) / (mu_2 + u)) *
            (v + k * u * (u - a - 1.)))
    return v


@njit(parallel=True)
def ionic_kernel_2d(u_new, u, v, indexes, dt, a, k, eap, mu_1, mu_2):
    """
    Computes the ionic kernel for the Aliev-Panfilov 2D model.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated action potential values.
    u : np.ndarray
        Current action potential array.
    v : np.ndarray
        Recovery variable array.
    indexes : np.ndarray
        Array of indices where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.
    """

    n_j = u.shape[1]

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii / n_j)
        j = ii % n_j

        v[i, j] = calc_v(v[i, j], u[i, j], dt, a, k, eap, mu_1, mu_2)

        u_new[i, j] += dt * (- k * u[i, j] * (u[i, j] - a) * (u[i, j] - 1.) -
                            u[i, j] * v[i, j])

