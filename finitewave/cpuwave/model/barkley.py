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
        self.arrays = ["u", "v"]
        self.scalars = ["a", "b", "eap"]

    def generate_body(self) -> str:
        u_idx = self._indexing("u")
        v_idx = self._indexing("v")
        v_set = self._indexing("v")
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        u_idx = {u_idx}
        v_idx = {v_idx}

        v_new = v_idx + dt * calc_v(v_idx, u_idx)
        {v_set} = v_new

        {u_new} += dt * calc_rhs(u_idx, v_new, a, b, eap)
"""


class Barkley(CardiacModel):
    def __init__(self):
        super().__init__()
        self.v = np.ndarray

        self.D_model = 1.0
        self.state_vars = ["u", "v"]
        self.npfloat = "float64"

        self.parameters = ops.get_parameters()
        self.variables = ops.get_variables()

        self.par_a = self.parameters["a"]
        self.par_b = self.parameters["b"]
        self.par_eap = self.parameters["eap"]

        self.var_u = self.variables["u"]
        self.var_v = self.variables["v"]

    def initialize(self):
        super().initialize()

        self.u = self.var_u * np.ones_like(self.u, dtype=self.npfloat)
        self.v = self.var_v * np.ones_like(self.u, dtype=self.npfloat)

        scalar_params, array_params = [], []
        for par_name in self.parameters.keys():
            val = getattr(self, f"par_{par_name}")
            (scalar_params if np.isscalar(val) else array_params).append(par_name)

        gen = BarkleyKernel()
        glb = {
            "calc_v": jit_ops["calc_v"],
            "calc_rhs": jit_ops["calc_rhs"],
        }

        self._kernel, _ = build_kernel(
            gen=gen,
            glb=glb,
            dimensions=self.cardiac_tissue.dimensions,
            scalar_params=tuple(sorted(scalar_params)),
            array_params=tuple(sorted(array_params)),
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
            self.par_a,
            self.par_b,
            self.par_eap,
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


