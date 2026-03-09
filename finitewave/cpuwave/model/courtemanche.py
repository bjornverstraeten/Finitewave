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
            "nao", "ko", "cao", "R", "T", "F", "Cm",
            "gna", "gk1", "gto", "gcal", "gnab", "gcab",
            "gkr", "gks", "inakmax", "kmnai", "kmko",
            "inacamax", "kmnancx", "kmcancx", "ksatncx",
            "ipcamax", "iupmax", "kup", "caupmax",
            "krel", "Vrel", "Vup", "Vj",
            "kq10", "ibk",
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
        ina = calc_ina(u_loc, {model['m']}, {model['h']}, {model['j']}, {model['gna']}, ena, {model['Cm']})
        ik1 = calc_ik1(u_loc, {model['gk1']}, ek, {model['Cm']})

        ito, {model['oa']}, {model['oi']} = calc_ito(
            u_loc, dt, {model['kq10']}, {model['oa']}, {model['oi']}, {model['gto']}, ek, {model['Cm']}
        )
        ikur, {model['ua']}, {model['ui']} = calc_ikur(
            u_loc, dt, {model['kq10']}, {model['ua']}, {model['ui']}, ek, {model['Cm']}
        )

        ikr, {model['xr']} = calc_ikr(u_loc, dt, {model['xr']}, {model['gkr']}, ek, {model['Cm']})
        iks, {model['xs']} = calc_iks(u_loc, dt, {model['xs']}, {model['gks']}, ek, {model['Cm']})

        ical, {model['d']}, {model['f']}, {model['fca']} = calc_ical(
            u_loc, dt, {model['d']}, {model['f']}, {model['cai']}, {model['gcal']}, {model['fca']}, {model['Cm']}
        )

        inak = calc_inak(
            {model['inakmax']}, {model['nai']}, {model['nao']}, {model['ko']},
            {model['kmnai']}, {model['kmko']},
            {model['F']}, u_loc, {model['R']}, {model['T']}, {model['Cm']}
        )
        inaca = calc_inaca(
            {model['inacamax']}, {model['nai']}, {model['nao']}, {model['cai']}, {model['cao']},
            {model['kmnancx']}, {model['kmcancx']}, {model['ksatncx']},
            {model['F']}, u_loc, {model['R']}, {model['T']}, {model['Cm']}
        )

        ibca = calc_ibca({model['gcab']}, eca, u_loc, {model['Cm']})
        ibna = calc_ibna({model['gnab']}, ena, u_loc, {model['Cm']})
        ipca = calc_ipca({model['ipcamax']}, {model['cai']}, {model['Cm']})

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
        {u_new} += dt * (-calc_rhs(ina, ik1, ito, ikur, ikr, iks, ical, ipca, inak, inaca, ibna, ibca, {model['Cm']}))
    """


class Courtemanche(CardiacModel):
    """
    This model describes the ionic currents and action potential dynamics of human atrial myocytes. 
    It includes detailed formulations for major ionic currents (fast sodium current, L-type calcium current, 
    inward rectifier potassium current, transient outward potassium current, rapid and slow delayed rectifier potassium currents, 
    and Na⁺/Ca²⁺ exchanger), as well as calcium handling mechanisms.

    The Courtemanche model is widely used as a reference atrial electrophysiology model. 
    It has served as the basis for many subsequent atrial modeling studies, including investigations of atrial fibrillation and drug effects.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient for the model, used in tissue simulations.
    npfloat : str
        String specifying the floating-point precision to use (e.g., 'float64').

    Model Variables
    ---------------
    u : np.ndarray
        Transmembrane potential in mV.
    nai : np.ndarray
        Intracellular sodium concentration in mM.
    ki : np.ndarray
        Intracellular potassium concentration in mM.
    cai : np.ndarray
        Intracellular calcium concentration in mM.
    m, h, j : np.ndarray
        Gating variables for the fast sodium current.
    oa, oi : np.ndarray
        Gating variables for the transient outward potassium current (I_to).
    ua, ui : np.ndarray
        Gating variables for the ultrarapid delayed rectifier potassium current (I_kur).
    xr : np.ndarray
        Gating variable for the rapid delayed rectifier potassium current (I_kr).
    xs : np.ndarray
        Gating variable for the slow delayed rectifier potassium current (I_ks).
    d, f : np.ndarray
        Gating variables for the L-type calcium current (I_cal).
    fca : np.ndarray
        Calcium-dependent inactivation variable for I_cal.
    urel, vrel, wrel : np.ndarray
        Gating variables for sarcoplasmic reticulum (SR) calcium release.
    ire : np.ndarray
        SR calcium release current.
    caup : np.ndarray
        SR calcium uptake variable.
    carel : np.ndarray
        SR calcium release variable.

    Model Parameters
    ----------------
    nao : float
        Extracellular sodium concentration in mM.
    ko : float
        Extracellular potassium concentration in mM.
    cao : float
        Extracellular calcium concentration in mM.
    R : float
        Universal gas constant in J/(mol*K).
    T : float
        Absolute temperature in K.
    F : float
        Faraday's constant in C/mol.
    Cm : float
        Membrane capacitance in μF/cm².
    gna : float
        Maximum conductance for the fast sodium current in mS/μF.
    gk1 : float
        Maximum conductance for the inward rectifier potassium current in mS/μF.
    gto : float
        Maximum conductance for the transient outward potassium current in mS/μF.
    gcal : float
        Maximum conductance for the L-type calcium current in mS/μF.
    gnab : float
        Maximum conductance for the background sodium current in mS/μF.
    gcab : float
        Maximum conductance for the background calcium current in mS/μF.
    gkr : float
        Maximum conductance for the rapid delayed rectifier potassium current in mS/μF.
    gks : float
        Maximum conductance for the slow delayed rectifier potassium current in mS/μF.
    inakmax : float
        Maximum current for the Na⁺/K⁺ pump in μA/μF.
    kmnai : float
        Half-saturation concentration for intracellular sodium in mM (Na⁺/K⁺ pump).
    kmko : float
        Half-saturation concentration for extracellular potassium in mM (Na⁺/K⁺ pump).
    inacamax : float
        Maximum current for the Na⁺/Ca²⁺ exchanger in μA/μF.
    kmnancx : float
        Half-saturation concentration for intracellular sodium in mM (Na⁺/Ca²⁺ exchanger).
    kmcancx : float
        Half-saturation concentration for intracellular calcium in mM (Na⁺/Ca²⁺ exchanger).
    ksatncx : float
        Saturation factor for the Na⁺/Ca²⁺ exchanger.
    ipcamax : float
        Maximum current for the plasma membrane Ca²⁺ ATPase in μA/μF.
    iupmax : float
        Maximum uptake rate for the SR calcium pump in mM/ms.
    kup : float
        Half-saturation concentration for calcium uptake in mM.
    caupmax : float
        Maximum SR calcium uptake variable in mM.
    krel : float
        Rate constant for SR calcium release in ms⁻¹.
    Vrel : float
        Volume ratio for SR release.
    Vup : float
        Volume ratio for SR uptake.
    Vj : float
        Volume ratio for junctional space.
    kq10 : float
        Temperature coefficient for gating kinetics.
    ibk : float
        Background potassium current in μA/μF.
    trpnmax : float
        Maximum concentration of troponin C in mM.
    kmtrpn : float
        Half-saturation concentration for troponin C in mM.
    cmdnmax : float
        Maximum concentration of calmodulin in mM.
    kmcmdn : float
        Half-saturation concentration for calmodulin in mM.
    csqnmax : float
        Maximum concentration of calsequestrin in mM.
    kmcsqn : float
        Half-saturation concentration for calsequestrin in mM.
    

    Paper
    -----
    Courtemanche M, Ramirez RJ, Nattel S. 
    Ionic mechanisms underlying human atrial action potential properties: insights from a mathematical model. 
    Am J Physiol. 1998 Jul;275(1):H301-21.
    https://doi.org/10.1152/ajpheart.1998.275.1.H301
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

