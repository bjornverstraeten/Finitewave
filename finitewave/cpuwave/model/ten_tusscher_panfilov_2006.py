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
    ops = load_ops("ten_tusscher_panfilov_2006")
    jit_ops = wrap_calc(ops)
except KeyError as e:
    raise ImportError("TP06 model ops not found.") from e


class TenTusscherPanfilov2006Kernel(IonicKernelGenerator):
    def __init__(self):
        super().__init__()
        self.args_order = [
         "u", "cai", "casr", "cass", "nai", "Ki",
            "m", "h", "j", "xr1", "xr2", "xs", "r", "s",
            "d", "f", "f2", "fcass", "rr", "oo",
            "ko", "cao", "nao", "Vc", "Vsr", "Vss", "Bufc", 
            "Kbufc", "Bufsr", "Kbufsr", "Bufss", "Kbufss",
            "Vmaxup", "Kup", "Vrel", "k1", "k2", "k3", "k4", "EC",
            "maxsr", "minsr", "Vleak", "Vxfer", "R", "F", "T", "RTONF", "CAPACITANCE",
            "gkr", "pKNa", "gk1", "gna", "gbna", "KmK", "KmNa", "knak", "gcal", "gbca",
            "knaca", "KmNai", "KmCa", "ksat", "n_", "gpca", "KpCa", "gpk", "gto", "gks"
        ]

    def generate_body(self) -> str:
        model = {name: self._indexing(name) for name in self.args_order}
        u_new = f"u_new{self._raw_indexing()}"

        return f"""\
        u_loc = {model['u']}

        # --- Handy constants (local) ---
        inverseVcF2  = 1.0 / (2.0 * {model['Vc']}  * {model['F']})
        inverseVcF   = 1.0 / ({model['Vc']}  * {model['F']})
        inversevssF2 = 1.0 / (2.0 * {model['Vss']} * {model['F']})

        # --- Reversal potentials ---
        Ek  = {model['RTONF']} * np.log({model['ko']} / {model['Ki']})
        Ena = {model['RTONF']} * np.log({model['nao']} / {model['nai']})
        Eks = {model['RTONF']} * np.log(({model['ko']} + {model['pKNa']} * {model['nao']}) /
                                        ({model['Ki']} + {model['pKNa']} * {model['nai']}))
        Eca = 0.5 * {model['RTONF']} * np.log({model['cao']} / {model['cai']})

        # --- Currents + gating updates ---
        ina, {model['m']}, {model['h']}, {model['j']} = calc_ina(
            u_loc, {model['m']}, {model['h']}, {model['j']},
            {model['gna']}, Ena
        )

        ical, {model['d']}, {model['f']}, {model['f2']}, {model['fcass']} = calc_ical(
            u_loc, dt, {model['d']}, {model['f']}, {model['f2']}, {model['fcass']},
            {model['cao']}, {model['cass']},
            {model['gcal']}, {model['F']}, {model['R']}, {model['T']}
        )

        ito, {model['r']}, {model['s']} = calc_ito(
            u_loc, dt, {model['r']}, {model['s']}, Ek, {model['gto']}
        )

        ikr, {model['xr1']}, {model['xr2']} = calc_ikr(
            u_loc, dt, {model['xr1']}, {model['xr2']}, Ek, {model['gkr']}, {model['ko']}
        )

        iks, {model['xs']} = calc_iks(
            u_loc, dt, {model['xs']}, Eks, {model['gks']}
        )

        ik1 = calc_ik1(u_loc, Ek, {model['gk1']})

        inaca = calc_inaca(
            u_loc, {model['nao']}, {model['nai']}, {model['cao']}, {model['cai']},
            {model['KmNai']}, {model['KmCa']}, {model['knaca']}, {model['ksat']}, {model['n_']},
            {model['F']}, {model['R']}, {model['T']}
        )

        inak = calc_inak(
            u_loc, {model['nai']}, {model['ko']},
            {model['KmK']}, {model['KmNa']}, {model['knak']},
            {model['F']}, {model['R']}, {model['T']}
        )

        ipca = calc_ipca({model['cai']}, {model['KpCa']}, {model['gpca']})
        ipk  = calc_ipk(u_loc, Ek, {model['gpk']})

        ibna = calc_ibna(u_loc, Ena, {model['gbna']})
        ibca = calc_ibca(u_loc, Eca, {model['gbca']})

        irel, {model['rr']}, {model['oo']} = calc_irel(
            dt, {model['rr']}, {model['oo']},
            {model['casr']}, {model['cass']},
            {model['Vrel']}, {model['k1']}, {model['k2']}, {model['k3']}, {model['k4']},
            {model['maxsr']}, {model['minsr']}, {model['EC']}
        )

        ileak = calc_ileak({model['casr']}, {model['cai']}, {model['Vleak']})
        iup   = calc_iup({model['cai']}, {model['Vmaxup']}, {model['Kup']})
        ixfer = calc_ixfer({model['cass']}, {model['cai']}, {model['Vxfer']})

        # --- Concentrations ---
        {model['casr']} = calc_casr(
            dt, {model['casr']},
            {model['Bufsr']}, {model['Kbufsr']},
            iup, irel, ileak
        )

        {model['cass']} = calc_cass(
            dt, {model['cass']},
            {model['Bufss']}, {model['Kbufss']},
            ixfer, irel, ical,
            {model['CAPACITANCE']}, {model['Vc']}, {model['Vss']}, {model['Vsr']},
            inversevssF2
        )

        {model['cai']} = calc_cai(
            dt, {model['cai']},
            {model['Bufc']}, {model['Kbufc']},
            ibca, ipca, inaca, iup, ileak, ixfer,
            {model['CAPACITANCE']}, {model['Vsr']}, {model['Vc']},
            inverseVcF2
        )

        {model['nai']} += dt * calc_dnai(
            ina, ibna, inak, inaca, {model['CAPACITANCE']}, inverseVcF
        )

        {model['Ki']}  += dt * calc_dki(
            ik1, ito, ikr, iks, inak, ipk, inverseVcF, {model['CAPACITANCE']}
        )

        # --- Voltage ---
        {u_new} -= dt * (calc_rhs(
            ikr, iks, ik1, ito,
            ina, ibna,
            ical, ibca,
            inak, inaca,
            ipca, ipk
        ))
"""


class TenTusscherPanfilov2006(CardiacModel):
    def __init__(self):
        super().__init__()

        self.D_model = 0.154
        self.npfloat = "float64"

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
        super().initialize()

        # allocate state arrays
        for name in self.default_variables.keys():
            init_val = getattr(self, f"init_{name}")
            setattr(self, name, init_val * np.ones_like(self.u, dtype=self.npfloat))

        # validate parameter fields shapes if they are arrays
        tissue_shape = self.cardiac_tissue.mesh.shape
        for name in self.default_parameters.keys():
            par = getattr(self, name)
            if isinstance(par, np.ndarray):
                if par.shape != tissue_shape:
                    raise ValueError(
                        f"param '{name}' shape {par.shape} != tissue shape {tissue_shape}"
                    )

        gen = TenTusscherPanfilov2006Kernel()
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
            "np": np,
            # currents/gates
            "calc_ina":  jit_ops["calc_ina"],
            "calc_ical": jit_ops["calc_ical"],
            "calc_ito":  jit_ops["calc_ito"],
            "calc_ikr":  jit_ops["calc_ikr"],
            "calc_iks":  jit_ops["calc_iks"],
            "calc_ik1":  jit_ops["calc_ik1"],
            "calc_inaca": jit_ops["calc_inaca"],
            "calc_inak":  jit_ops["calc_inak"],
            "calc_ipca":  jit_ops["calc_ipca"],
            "calc_ipk":   jit_ops["calc_ipk"],
            "calc_ibna":  jit_ops["calc_ibna"],
            "calc_ibca":  jit_ops["calc_ibca"],
            "calc_irel":  jit_ops["calc_irel"],
            "calc_ileak": jit_ops["calc_ileak"],
            "calc_iup":   jit_ops["calc_iup"],
            "calc_ixfer": jit_ops["calc_ixfer"],
            # concentrations + voltage rhs
            "calc_casr": jit_ops["calc_casr"],
            "calc_cass": jit_ops["calc_cass"],
            "calc_cai":  jit_ops["calc_cai"],
            "calc_dnai": jit_ops["calc_dnai"],
            "calc_dki":  jit_ops["calc_dki"],
            "calc_rhs":  jit_ops["calc_rhs"],
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
