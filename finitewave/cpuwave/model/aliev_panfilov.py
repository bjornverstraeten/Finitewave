import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.core.model.ionic_kernel_generator import IonicKernelGenerator

from finitewave.cpuwave.stencil.sten2D.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave.stencil.sten2D.isotropic_stencil_2d import (
    IsotropicStencil2D
)
from finitewave.cpuwave.stencil.sten3D.asymmetric_stencil_3d import (
    AsymmetricStencil3D
)
from finitewave.cpuwave.stencil.sten3D.isotropic_stencil_3d import (
    IsotropicStencil3D
)

from finitewave.cpuwave.model._registry import load_ops, wrap_calc
from finitewave.cpuwave.model._kernel_builder import build_kernel


try:
    ops = load_ops("aliev_panfilov")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Aliev–Panfilov model ops not found. "
        # "Install model package: pip install aliev-panfilov-finitewave-model"
    ) from e


class AlievPanfilovKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "v", "a", "k", "mu1", "mu2", "eps"
        ]

    def generate_body(self) -> str:
        model_dict = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"
        
        return f"""\

        {model_dict['v']} += dt * calc_dv({model_dict['v']}, {model_dict['u']}, 
            {model_dict['a']}, {model_dict['k']}, {model_dict['eps']}, {model_dict['mu1']}, {model_dict['mu2']})

        {u_new} += dt * calc_rhs({model_dict['u']}, {model_dict['v']}, {model_dict['a']}, {model_dict['k']})
"""


class AlievPanfilov(CardiacModel):
    """
    Implementation of the Aliev–Panfilov model of cardiac excitation.

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
    eps : float
        Baseline recovery rate.
    mu1 : float
        Recovery rate coefficient (scales v feedback).
    mu2 : float
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
        Initializes the AlievPanfilov instance with default parameters.
        """
        super().__init__()
        self.D_model = 1.
        self.state_vars = ops.get_variables().keys()
        self.state_pars = [p for p in ops.get_parameters().keys()]
        self.npfloat    = 'float64'

        # declare arrays
        self.v = np.ndarray

        # parameters + variables from ops
        self.default_parameters = ops.get_parameters()
        self.default_variables = ops.get_variables()

        # set parameters
        for name, value in self.default_parameters.items():
            setattr(self, name, value)

        # expose initial conditions as init_*
        for name, value in self.default_variables.items():
            setattr(self, f"init_{name}", value)

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()

        self.u = self.init_u * np.ones_like(self.u, dtype=self.npfloat)
        self.v = self.init_v * np.ones_like(self.u, dtype=self.npfloat)

        gen = AlievPanfilovKernel()
        for name in self.default_variables.keys():
            gen.arrays.append(name)
        for name in self.default_parameters.keys():
            if np.isscalar(getattr(self, name)):
                gen.scalars.append(name)
            elif isinstance(getattr(self, name), np.ndarray):
                gen.arrays.append(name)

        glb = {
            "calc_dv": jit_ops["calc_dv"], 
            "calc_rhs": jit_ops["calc_rhs"]
        }

        self._kernel, _ = build_kernel(
            gen=gen,
            glb=glb,
            dimensions=self.cardiac_tissue.dimensions,
            observers=self.observers,
        )

        self._buffs = self._form_and_verify_observers()
        
    def run_ionic_kernel(self):
        """
        Executes the ionic kernel for the Aliev-Panfilov model.
        """
        self._kernel(self.u_new, self.cardiac_tissue.myo_indexes, self.dt, self.step,
                        self.u, self.v, self.a, self.k, self.mu1, self.mu2, self.eps,
                        *self._buffs)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            if cardiac_tissue.dimensions == 2:
                return IsotropicStencil2D()
            elif cardiac_tissue.dimensions == 3:
                return IsotropicStencil3D()
            else:
                raise ValueError("Unsupported number of dimensions")
        else:
            if cardiac_tissue.dimensions == 2:
                return AsymmetricStencil2D()
            elif cardiac_tissue.dimensions == 3:
                return AsymmetricStencil3D()
            else:
                raise ValueError("Unsupported number of dimensions")


