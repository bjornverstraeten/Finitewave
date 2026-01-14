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
    ops = load_ops("luo_rudy_91")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError(
        "Luo–Rudy 1991 model ops not found."
        # "Install model package: pip install luo-rudy91-finitewave-model"
    ) from e


class LuoRudy91Kernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
            "u", "m", "h", "j", "d", "f", "x", "cai",
            "gna", "gsi", "gk", "gk1", "gkp", "gb",
            "ko", "ki", "nai", "nao", "cao",
            "R", "T", "F", "PR_NaK"
        ]

    def generate_body(self) -> str:
        model_dict = {var: self._indexing(var) for var in (self.arrays + self.scalars)}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        u_loc = {model_dict['u']}

        E_Na = ({model_dict['R']}*{model_dict['T']}/{model_dict['F']}) * np.log({model_dict['nao']}/{model_dict['nai']})
        E_K1 = ({model_dict['R']}*{model_dict['T']}/{model_dict['F']}) * np.log({model_dict['ko']}/{model_dict['ki']})

        {model_dict['m']} = {model_dict['m']} + dt * calc_dm(u_loc, {model_dict['m']})
        {model_dict['h']} = {model_dict['h']} + dt * calc_dh(u_loc, {model_dict['h']})
        {model_dict['j']} = {model_dict['j']} + dt * calc_dj(u_loc, {model_dict['j']})

        {model_dict['d']} = {model_dict['d']} + dt * calc_dd(u_loc, {model_dict['d']})
        {model_dict['f']} = {model_dict['f']} + dt * calc_df(u_loc, {model_dict['f']})
        {model_dict['x']} = {model_dict['x']} + dt * calc_dx(u_loc, {model_dict['x']})

        ina = calc_ina(u_loc, {model_dict['m']}, {model_dict['h']}, {model_dict['j']}, E_Na, {model_dict['gna']})
        isi = calc_isk(u_loc, {model_dict['d']}, {model_dict['f']}, {model_dict['cai']}, {model_dict['gsi']})
        {model_dict['cai']} = {model_dict['cai']} + dt * calc_dcai({model_dict['cai']}, isi)

        ik  = calc_ik(u_loc, {model_dict['x']}, {model_dict['ko']}, {model_dict['ki']}, {model_dict['nao']}, {model_dict['nai']}, 
            {model_dict['PR_NaK']}, {model_dict['R']}, {model_dict['T']}, {model_dict['F']}, {model_dict['gk']})
        ik1 = calc_ik1(u_loc, {model_dict['ko']}, E_K1, {model_dict['gk1']})
        ikp = calc_ikp(u_loc, E_K1, {model_dict['gkp']})
        ib  = calc_ib(u_loc, {model_dict['gb']})

        {u_new} += dt * calc_rhs(ina, isi, ik, ik1, ikp, ib)
    """



class LuoRudy91(CardiacModel):
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
        super().__init__()

        self.D_model = 0.1
        self.state_vars = ops.get_variables().keys()
        self.state_pars = [p for p in ops.get_parameters().keys()]
        self.npfloat = "float64"

        # declare arrays for clarity
        self.m = np.ndarray
        self.h = np.ndarray
        self.j = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.cai = np.ndarray

        # parameters + variables from ops
        self.default_parameters = ops.get_parameters()
        self.default_variables = ops.get_variables()

        # expose parameters as par_*
        for name, value in self.default_parameters.items():
            setattr(self, name, value)
        # remove E_Na, E_K1 from parameters as they are computed internally
        delattr(self, "E_Na")
        delattr(self, "E_K1")

        # expose initial conditions as init_*
        for name, value in self.default_variables.items():
            setattr(self, f"init_{name}", value)

    def initialize(self):
        super().initialize()

        self.u   = self.init_u   * np.ones_like(self.u, dtype=self.npfloat)
        self.m   = self.init_m   * np.ones_like(self.u, dtype=self.npfloat)
        self.h   = self.init_h   * np.ones_like(self.u, dtype=self.npfloat)
        self.j   = self.init_j   * np.ones_like(self.u, dtype=self.npfloat)
        self.d   = self.init_d   * np.ones_like(self.u, dtype=self.npfloat)
        self.f   = self.init_f   * np.ones_like(self.u, dtype=self.npfloat)
        self.x   = self.init_x   * np.ones_like(self.u, dtype=self.npfloat)
        self.cai = self.init_cai * np.ones_like(self.u, dtype=self.npfloat)

        for name in self.default_parameters.keys():
            if name in ("E_Na", "E_K1"):
                continue
            par = getattr(self, name)
            if isinstance(par, np.ndarray):
                if par.shape != self.cardiac_tissue.mesh.shape:
                    raise ValueError(f"par_{name} shape {par.shape} != tissue shape {self.cardiac_tissue.mesh.shape}")

        gen = LuoRudy91Kernel()
        for name in self.default_variables.keys():
            gen.arrays.append(name)
        for name in self.default_parameters.keys():
            if name in ("E_Na", "E_K1"): # computed internally
                continue
            if np.isscalar(getattr(self, name)):
                gen.scalars.append(name)
            elif isinstance(getattr(self, name), np.ndarray):
                gen.arrays.append(name)

        glb = {
            "np": np,
            "calc_dm": jit_ops["calc_dm"],
            "calc_dh": jit_ops["calc_dh"],
            "calc_dj": jit_ops["calc_dj"],
            "calc_dd": jit_ops["calc_dd"],
            "calc_df": jit_ops["calc_df"],
            "calc_dx": jit_ops["calc_dx"],
            "calc_dcai": jit_ops["calc_dcai"],
            "calc_ina": jit_ops["calc_ina"],
            "calc_isk": jit_ops["calc_isk"],
            "calc_ik": jit_ops["calc_ik"],
            "calc_ik1": jit_ops["calc_ik1"],
            "calc_ikp": jit_ops["calc_ikp"],
            "calc_ib": jit_ops["calc_ib"],
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
            self.u, self.m, self.h, self.j, self.d, self.f, self.x, self.cai,
            self.gna, self.gsi, self.gk, self.gk1, self.gkp, self.gb,
            self.ko, self.ki, self.nai, self.nao, self.cao,
            self.R, self.T, self.F, self.PR_NaK,
            # observers
            *self._buffs,
        )

    def select_stencil(self, cardiac_tissue):
        if cardiac_tissue.fibers is None:
            if cardiac_tissue.dimensions == 2:
                return IsotropicStencil2D()
            if cardiac_tissue.dimensions == 3:
                return IsotropicStencil3D()
            raise ValueError("Unsupported number of dimensions")
        else:
            if cardiac_tissue.dimensions == 2:
                return AsymmetricStencil2D()
            if cardiac_tissue.dimensions == 3:
                return AsymmetricStencil3D()
            raise ValueError("Unsupported number of dimensions")
