import numpy as np

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.core.model.ionic_kernel_generator import IonicKernelGenerator

from finitewave.cpuwave.stencil.sten2D.asymmetric_stencil_2d import AsymmetricStencil2D
from finitewave.cpuwave.stencil.sten2D.isotropic_stencil_2d import IsotropicStencil2D
from finitewave.cpuwave.stencil.sten3D.asymmetric_stencil_3d import AsymmetricStencil3D
from finitewave.cpuwave.stencil.sten3D.isotropic_stencil_3d import IsotropicStencil3D

from finitewave.cpuwave.model._registry import load_ops, wrap_calc
from finitewave.cpuwave.model._kernel_builder import build_kernel


try:
    ops = load_ops("barkley")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Barkley model ops not found. "
        "Install model package: pip install barkley-finitewave-model"
    ) from e


class BarkleyKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "v", "a", "b", "eps"
        ]

    def generate_body(self) -> str:
        model_dict = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        {model_dict['v']} += dt * calc_dv({model_dict['v']}, {model_dict['u']})

        {u_new} += dt * calc_rhs({model_dict['u']}, {model_dict['v']}, {model_dict['a']}, 
            {model_dict['b']}, {model_dict['eps']})
"""


class Barkley(CardiacModel):
    """
    Two-dimensional implementation of the Barkley model for excitable media.

    The Barkley model is a simplified two-variable reaction–diffusion system
    originally developed to study wave propagation in excitable media. While it is 
    not biophysically detailed, it captures essential qualitative features of 
    cardiac-like excitation dynamics such as spiral waves, wave break, and reentry.

    This implementation is included for benchmarking, educational purposes, 
    and comparison against more detailed cardiac models.

    Attributes
    ----------
    u : np.ndarray
        Excitation variable (analogous to membrane potential).
    v : np.ndarray
        Recovery variable controlling excitability.
    D_model : float
        Diffusion coefficient for excitation variable.
    state_vars : list of str
        Names of variables saved during simulation.
    npfloat : str
        Floating-point precision (default: 'float64').

    Model Parameters
    ----------------
    a : float
        Threshold-like parameter controlling excitability.
    b : float
        Recovery time scale.
    eap : float
        Controls sharpness of the activation term (nonlinear gain).

    Paper
    -----
    Barkley, D. (1991).
    A model for fast computer simulation of waves in excitable media.
    Physica D: Nonlinear Phenomena, 61-70.
    https://doi.org/10.1016/0167-2789(86)90198-1.

    """
    def __init__(self):
        super().__init__()
        self.D_model = 1.0
        self.state_vars = ["u", "v"]
        self.npfloat = "float64"

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

        gen = BarkleyKernel()
        for name in self.default_variables.keys():
            gen.arrays.append(name)
        for name in self.default_parameters.keys():
            if np.isscalar(getattr(self, name)):
                gen.scalars.append(name)
            elif isinstance(getattr(self, name), np.ndarray):
                gen.arrays.append(name)
        glb = {
            "calc_dv": jit_ops["calc_dv"],
            "calc_rhs": jit_ops["calc_rhs"],
        }

        self._kernel, _ = build_kernel(
            gen=gen,
            glb=glb,
            dimensions=self.cardiac_tissue.dimensions,
            observers=self.observers,
        )

        self._buffs = self._form_and_verify_observers()

    def run_ionic_kernel(self):
        self._kernel(
            self.u_new,
            self.cardiac_tissue.myo_indexes,
            self.dt,
            self.step,
            self.u,
            self.v,
            self.a,
            self.b,
            self.eps,
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


