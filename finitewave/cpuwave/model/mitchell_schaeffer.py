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
    ops = load_ops("mitchell_schaeffer")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Mitchell-Schaeffer model ops not found. "
        # "Install model package: pip install mitchell-schaeffer-finitewave-model"
    ) from e


class  MitchellSchaefferKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "h", "tau_close", "tau_open", "tau_out", "tau_in", "u_gate"
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"
        
        return f"""\

        {model['h']} += dt*calc_dh(
            {model['h']},
            {model['u']},
            {model['tau_close']},
            {model['tau_open']},
            {model['u_gate']}
        )

        J_in = calc_J_in(
            {model['u']},
            {model['h']},
            {model['tau_in']}
        )   

        J_out = calc_J_out(
            {model['u']},
            {model['tau_out']}
        )

        {u_new} += dt*calc_rhs(J_in, J_out) 

"""


class MitchellSchaeffer(CardiacModel):
    """
    Implements the Mitchell-Schaeffer model of cardiac excitation.

    This is a phenomenological two-variable model capturing the essence of cardiac 
    action potential dynamics using a simplified formulation. It separates inward and 
    outward currents and uses a single gating variable to regulate excitability.

    It reproduces key features like:
    - Excitability and recovery
    - Action potential duration (APD)
    - Restitution and wave propagation

    Attributes
    ----------
    h : np.ndarray
        Gating variable controlling the availability of inward current.
    D_model : float
        Diffusion coefficient for spatial propagation.
    state_vars : list
        Names of the dynamic variables for saving/restoring state.
    npfloat : str
        Floating-point type used (default: float64).

    Paper
    -----
    Mitchell, C. C., & Schaeffer, D. G. (2003).
    A two-current model for the dynamics of cardiac membrane
    potential. Bulletin of Mathematical Biology, 65, 767–793.
    https://doi.org/10.1016/S0092-8240(03)00041-7
        
    """

    def __init__(self):
        """
        Initializes the Mitchell-Schaeffer instance with default parameters.
        """
        super().__init__()
        self.D_model = 1.
        self.npfloat    = 'float64'

        self.default_parameters = ops.get_parameters()
        self.default_variables = ops.get_variables()

        self.state_vars = self.default_variables.keys()
        self.state_pars = list(self.default_parameters.keys())

        # expose parameters as direct attributes (scalar or array)
        for name, value in self.default_parameters.items():
            setattr(self, name, value)

        # expose initial conditions as init_*
        for name, value in self.default_variables.items():
            setattr(self, f"init_{name}", value)

        # declare arrays (optional, for readability/debug)
        for name in self.default_variables.keys():
            setattr(self, name, np.ndarray)

    def initialize(self):
        """
        Initializes the model for simulation.
        """
        super().initialize()

        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            setattr(self, name, init_val * np.ones_like(self.u, dtype=self.npfloat))
            if name == 'u':
                self.u_new = self.u.copy()

        # validate parameter fields shapes if they are arrays
        tissue_shape = self.cardiac_tissue.mesh.shape
        for name in self.default_parameters.keys():
            par = getattr(self, name)
            if isinstance(par, np.ndarray):
                if par.shape != tissue_shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape {tissue_shape}"
                    )

        gen = MitchellSchaefferKernel()
        self._kernel_args_order = gen.args_order[:]

        # args_order: state vars first, then all parameters (stable order for call site)
        param_names = list(self.default_parameters.keys())
        var_names = list(self.default_variables.keys())

        # Tell generator which names are arrays vs scalars (for indexing decisions)
        for name in var_names:
            gen.arrays.append(name)

        for name in param_names:
            par = getattr(self, name)
            if np.isscalar(par):
                gen.scalars.append(name)
            elif isinstance(par, np.ndarray):
                gen.arrays.append(name)


        glb = {
            "calc_dh": jit_ops["calc_dh"],
            "calc_J_out": jit_ops["calc_J_out"],
            "calc_J_in": jit_ops["calc_J_in"],
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
        Executes the ionic kernel for the Mitchell-Schaeffer model.
        """
        args = [getattr(self, name) for name in self._kernel_args_order]
        self._kernel(
            self.u_new,
            self.cardiac_tissue.myo_indexes,
            self.dt,
            self.step,
            *args,
            *self._buffs,
        )

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


