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
    ops = load_ops("courtemanche")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Courtemanche model ops not found. "
        "Install model package: pip install finitewave-model-courtemanche"
    ) from e


class CourtemancheKernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()

        self.args_order = [
            "u", "nai", "ki", "cai",
            "m", "h", "j",
            "oa", "oi", "ua", "ui", "xr", "xs", "d", "f", "fca",
            "urel", "vrel", "wrel", "irel",
            "caup", "carel",
            "nao", "ko", "cao", "R", "T", "F",
            "gna", "gk1", "gto", "gcal", "gnab", "gcab",
            "gkr", "gks", "inakmax", "kmnai", "kmko",
            "inacamax", "kmnancx", "kmcancx", "ksatncx",
            "ipcamax", "iupmax", "kup", "caupmax",
            "krel", "Vrel", "Vup", "Vj",
            "kq10", "gkur_coeff",
            "ibk",
            "trpnmax", "kmtrpn", "cmdnmax", "kmcmdn",
            "csqnmax", "kmcsqn",
        ]

    def generate_body(self) -> str:
        model = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        u_loc = {model['u']}

        # Equilibrium potentials
        ena, ek, eca = calc_equilibrum_potentials(
            {model['nai']}, {model['nao']},
            {model['ki']}, {model['ko']},
            {model['cai']}, {model['cao']},
            {model['R']}, {model['T']}, {model['F']}
        )

        # Fast Na gating
        {model['m']} = calc_gating_m({model['m']}, u_loc, dt)
        {model['h']} = calc_gating_h({model['h']}, u_loc, dt)
        {model['j']} = calc_gating_j({model['j']}, u_loc, dt)

        # Currents
        ina = calc_ina(u_loc, {model['m']}, {model['h']}, {model['j']}, {model['gna']}, ena)
        ik1 = calc_ik1(u_loc, {model['gk1']}, ek)

        ito, {model['oa']}, {model['oi']} = calc_ito(
            u_loc, dt, {model['kq10']}, {model['oa']}, {model['oi']}, {model['gto']}, ek
        )
        ikur, {model['ua']}, {model['ui']} = calc_ikur(
            u_loc, dt, {model['kq10']}, {model['ua']}, {model['ui']}, ek, {model['gkur_coeff']}
        )

        ikr, {model['xr']} = calc_ikr(u_loc, dt, {model['xr']}, {model['gkr']}, ek)
        iks, {model['xs']} = calc_iks(u_loc, dt, {model['xs']}, {model['gks']}, ek)

        ical, {model['d']}, {model['f']}, {model['fca']} = calc_ical(
            u_loc, dt, {model['d']}, {model['f']}, {model['cai']}, {model['gcal']}, {model['fca']}
        )

        inak = calc_inak(
            {model['inakmax']}, {model['nai']}, {model['nao']}, {model['ko']},
            {model['kmnai']}, {model['kmko']},
            {model['F']}, u_loc, {model['R']}, {model['T']}
        )
        inaca = calc_inaca(
            {model['inacamax']}, {model['nai']}, {model['nao']}, {model['cai']}, {model['cao']},
            {model['kmnancx']}, {model['kmcancx']}, {model['ksatncx']},
            {model['F']}, u_loc, {model['R']}, {model['T']}
        )

        ibca = calc_ibca({model['gcab']}, eca, u_loc)
        ibna = calc_ibna({model['gnab']}, ena, u_loc)
        ipca = calc_ipca({model['ipcamax']}, {model['cai']})

        # SR release / uptake
        {model['irel']}, {model['urel']}, {model['vrel']}, {model['wrel']} = calc_irel(
            dt,
            {model['urel']}, {model['vrel']}, {model['irel']}, {model['wrel']},
            ical, inaca,
            {model['krel']}, {model['carel']}, {model['cai']}, u_loc,
            {model['F']}, {model['Vrel']}
        )

        itr = calc_itr({model['caup']}, {model['carel']})
        iup = calc_iup({model['iupmax']}, {model['cai']}, {model['kup']})
        iupleak = calc_iupleak({model['caup']}, {model['caupmax']}, {model['iupmax']})

        # Concentrations / buffers
        {model['caup']} += dt * calc_dcaup(iup, iupleak, itr, {model['Vrel']}, {model['Vup']})
        {model['nai']}  += dt * calc_dnai(inak, inaca, ibna, ina, {model['F']}, {model['Vj']})
        {model['ki']}   += dt * calc_dki(inak, ik1, ito, ikur, ikr, iks, {model['ibk']}, {model['F']}, {model['Vj']})

        {model['cai']}  += dt * calc_dcai(
            {model['cai']}, inaca, ipca, ical, ibca, iup, iupleak, {model['irel']},
            {model['Vrel']}, {model['Vup']},
            {model['trpnmax']}, {model['kmtrpn']},
            {model['cmdnmax']}, {model['kmcmdn']},
            {model['F']}, {model['Vj']}
        )

        {model['carel']} += dt * calc_dcarel(
            {model['carel']}, itr, {model['irel']}, {model['csqnmax']}, {model['kmcsqn']}
        )

        # Membrane potential update:
        # in 0D: u += dt * (-rhs + stim)
        # in tissue: stim already applied earlier, so only -rhs here
        {u_new} += dt * (-calc_rhs(ina, ik1, ito, ikur, ikr, iks, ical, ipca, inak, inaca, ibna, ibca))
    """


class Courtemanche(CardiacModel):
    """
    A class to represent the Courtemanche cardiac model in 2D.

    Attributes
    ----------
    D_model : float
        Model specific diffusion coefficient.
    state_vars : list of str
        List of state variable names.
    """
    def __init__(self):
        super().__init__()
        self.D_model = 1.0
        self.npfloat = "float64"

        self._initialize_variables_and_parameters(ops)

    def initialize(self):
        super().initialize()

        self._allocate_state_arrays()

        gen = self._initialize_kernel(CourtemancheKernel)
    
        glb = {
            "np": np,
            "calc_equilibrum_potentials": jit_ops["calc_equilibrum_potentials"],
            "calc_gating_m": jit_ops["calc_gating_m"],
            "calc_gating_h": jit_ops["calc_gating_h"],
            "calc_gating_j": jit_ops["calc_gating_j"],
            "calc_ina": jit_ops["calc_ina"],
            "calc_ik1": jit_ops["calc_ik1"],
            "calc_ito": jit_ops["calc_ito"],
            "calc_ikur": jit_ops["calc_ikur"],
            "calc_ikr": jit_ops["calc_ikr"],
            "calc_iks": jit_ops["calc_iks"],
            "calc_ical": jit_ops["calc_ical"],
            "calc_inak": jit_ops["calc_inak"],
            "calc_inaca": jit_ops["calc_inaca"],
            "calc_ibca": jit_ops["calc_ibca"],
            "calc_ibna": jit_ops["calc_ibna"],
            "calc_ipca": jit_ops["calc_ipca"],
            "calc_irel": jit_ops["calc_irel"],
            "calc_itr": jit_ops["calc_itr"],
            "calc_iup": jit_ops["calc_iup"],
            "calc_iupleak": jit_ops["calc_iupleak"],
            "calc_dcaup": jit_ops["calc_dcaup"],
            "calc_dnai": jit_ops["calc_dnai"],
            "calc_dki": jit_ops["calc_dki"],
            "calc_dcai": jit_ops["calc_dcai"],
            "calc_dcarel": jit_ops["calc_dcarel"],
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

