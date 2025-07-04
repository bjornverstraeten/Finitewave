import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class TP062D(CardiacModel):
    """
    Implements the ten Tusscher–Panfilov 2006 (TP06) human ventricular ionic model in 2D.

    The TP06 model is a detailed biophysical model of the human ventricular 
    action potential, designed to simulate realistic electrical behavior in 
    tissue including alternans, reentrant waves, and spiral wave breakup.

    This model includes:
    - 18 dynamic state variables (voltage, ion concentrations, channel gates, buffers)
    - Full calcium handling with subspace (cass) and sarcoplasmic reticulum (casr)
    - Sodium, potassium, and calcium currents including background, exchanger, and pumps
    - Buffering effects and intracellular transport

    Finitewave provides this model in 2D form for efficient simulation and 
    reproducible experimentation with custom spatial setups.

    Attributes
    ----------
    D_model : float
        Diffusion coefficient specific to this model (cm²/ms).
    state_vars : list of str
        List of all state variable names, used for checkpointing and logging.
    ko : float
        Extracellular potassium concentration (mM).
    cao : float
        Extracellular calcium concentration (mM).
    nao : float
        Extracellular sodium concentration (mM).
    Vc : float
        Cytoplasmic volume (μL).
    Vsr : float
        Sarcoplasmic reticulum volume (μL).
    Vss : float
        Subsarcolemmal space volume (μL).
    R, T, F : float
        Universal gas constant, absolute temperature, and Faraday constant.
    RTONF : float
        Precomputed RT/F value for Nernst equation.
    CAPACITANCE : float
        Membrane capacitance per unit area (μF/cm²).
    gna, gcal, gkr, gks, gk1, gto : float
        Conductances for major ionic channels.
    gbna, gbca : float
        Background sodium and calcium conductances.
    gpca, gpk : float
        Pump-related conductances.
    knak, knaca : float
        Maximal Na⁺/K⁺ pump and Na⁺/Ca²⁺ exchanger rates.
    Km*, Kbuf*, Vmaxup, Vrel, etc.
        Numerous kinetic constants for buffering, pump activity, and calcium handling.

    Paper
    -----
    ten Tusscher KH, Panfilov AV. 
    Alternans and spiral breakup in a human ventricular tissue model.
    Am J Physiol Heart Circ Physiol. 2006 Sep;291(3):H1088–H1100.
    https://doi.org/10.1152/ajpheart.00109.2006

    """

    def __init__(self):
        super().__init__()
        self.D_model    = 0.154

        self.state_vars = ["u", "cai", "casr", "cass", "nai", "Ki",
                           "m", "h", "j", "xr1", "xr2", "xs", "r",
                           "s", "d", "f", "f2", "fcass", "rr", "oo"]
        self.npfloat    = 'float64'

        # Extracellular Ion Concentrations (mM)
        self.ko  = 5.4     # Potassium extracellular concentration
        self.cao = 2.0     # Calcium extracellular concentration
        self.nao = 140.0   # Sodium extracellular concentration

        # Cell Volume (in uL)
        self.Vc  = 0.016404   # Cytoplasmic volume
        self.Vsr = 0.001094   # Sarcoplasmic reticulum volume
        self.Vss = 0.00005468 # Subsarcolemmal space volume

        # Buffering Parameters
        self.Bufc   = 0.2     # Cytoplasmic buffer concentration
        self.Kbufc  = 0.001   # Cytoplasmic buffer affinity
        self.Bufsr  = 10.0    # SR buffer concentration
        self.Kbufsr = 0.3     # SR buffer affinity
        self.Bufss  = 0.4     # Subsarcolemmal buffer concentration
        self.Kbufss = 0.00025 # Subsarcolemmal buffer affinity

        # Calcium Handling Parameters
        self.Vmaxup = 0.006375  # Maximal calcium uptake rate
        self.Kup    = 0.00025   # Calcium uptake affinity
        self.Vrel   = 0.102     # Calcium release rate from SR
        self.k1_    = 0.15      # Transition rate for SR calcium release
        self.k2_    = 0.045
        self.k3     = 0.060
        self.k4     = 0.005      # Alternative transition rate
        self.EC     = 1.5        # Calcium-induced calcium release sensitivity
        self.maxsr  = 2.5        # Maximum SR calcium release permeability
        self.minsr  = 1.0        # Minimum SR calcium release permeability
        self.Vleak  = 0.00036    # SR calcium leak rate
        self.Vxfer  = 0.0038     # Calcium transfer rate from subspace to cytosol

        # Physical Constants
        self.R     = 8314.472   # Universal gas constant (J/(kmol·K))
        self.F     = 96485.3415 # Faraday constant (C/mol)
        self.T     = 310.0      # Temperature (Kelvin, 37°C)
        self.RTONF = 26.71376   # RT/F constant for Nernst equation

        # Membrane Capacitance
        self.CAPACITANCE = 0.185 # Membrane capacitance (μF/cm²)

        # Ion Channel Conductances
        self.gkr  = 0.153       # Rapid delayed rectifier K+ conductance
        self.gks  = 0.392       # Slow delayed rectifier K+ conductance
        self.gk1  = 5.405       # Inward rectifier K+ conductance
        self.gto  = 0.294       # Transient outward K+ conductance
        self.gna  = 14.838      # Fast Na+ conductance
        self.gbna = 0.00029     # Background Na+ conductance
        self.gcal = 0.00003980  # L-type Ca2+ channel conductance
        self.gbca = 0.000592    # Background Ca2+ conductance
        self.gpca = 0.1238      # Sarcolemmal Ca2+ pump current conductance
        self.KpCa = 0.0005      # Sarcolemmal Ca2+ pump affinity
        self.gpk  = 0.0146      # Na+/K+ pump current conductance

        # Na+/K+ Pump Parameters
        self.pKNa = 0.03        # Na+/K+ permeability ratio
        self.KmK  = 1.0         # Half-saturation for K+ activation
        self.KmNa = 40.0        # Half-saturation for Na+ activation
        self.knak = 2.724       # Maximal Na+/K+ pump rate

        # Na+/Ca2+ Exchanger Parameters
        self.knaca = 1000       # Maximal Na+/Ca2+ exchanger current
        self.KmNai = 87.5       # Half-saturation for Na+ binding
        self.KmCa  = 1.38       # Half-saturation for Ca2+ binding
        self.ksat  = 0.1        # Saturation factor
        self.n_   = 0.35        # Exponent for Na+ dependence

        # initial conditions
        self.init_u     = -84.5
        self.init_cai   = 0.00007
        self.init_casr  = 1.3
        self.init_cass  = 0.00007
        self.init_nai   = 7.67
        self.init_Ki    = 138.3
        self.init_m     = 0.0
        self.init_h     = 0.75
        self.init_j     = 0.75
        self.init_xr1   = 0.0
        self.init_xr2   = 1.0
        self.init_xs    = 0.0
        self.init_r     = 0.0
        self.init_s     = 1.0
        self.init_d     = 0.0
        self.init_f     = 1.0
        self.init_f2    = 1.0
        self.init_fcass = 1.0
        self.init_rr    = 1.0
        self.init_oo    = 0.0

    def initialize(self):
        """
        Initializes the model's state variables and diffusion/ionic kernels.

        Sets up the initial values for membrane potential, ion concentrations,
        gating variables, and assigns the appropriate kernel functions.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = self.init_u * np.ones(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.cai = self.init_cai * np.ones(shape, dtype=self.npfloat)
        self.casr = self.init_casr * np.ones(shape, dtype=self.npfloat)
        self.cass = self.init_cass * np.ones(shape, dtype=self.npfloat)
        self.nai = self.init_nai * np.ones(shape, dtype=self.npfloat)
        self.Ki = self.init_Ki * np.ones(shape, dtype=self.npfloat)
        self.m = self.init_m * np.ones(shape, dtype=self.npfloat)
        self.h = self.init_h * np.ones(shape, dtype=self.npfloat)
        self.j = self.init_j * np.ones(shape, dtype=self.npfloat)
        self.xr1 = self.init_xr1 * np.ones(shape, dtype=self.npfloat)
        self.xr2 = self.init_xr2 * np.ones(shape, dtype=self.npfloat)
        self.xs = self.init_xs * np.ones(shape, dtype=self.npfloat)
        self.r = self.init_r * np.ones(shape, dtype=self.npfloat)
        self.s = self.init_s * np.ones(shape, dtype=self.npfloat)
        self.d = self.init_d * np.ones(shape, dtype=self.npfloat)
        self.f = self.init_f * np.ones(shape, dtype=self.npfloat)
        self.f2 = self.init_f2 * np.ones(shape, dtype=self.npfloat)
        self.fcass = self.init_fcass * np.ones(shape, dtype=self.npfloat)
        self.rr = self.init_rr * np.ones(shape, dtype=self.npfloat)
        self.oo = self.init_oo * np.ones(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.cai, self.casr, self.cass,
                        self.nai, self.Ki, self.m, self.h, self.j, self.xr1,
                        self.xr2, self.xs, self.r, self.s, self.d, self.f,
                        self.f2, self.fcass, self.rr, self.oo,
                        self.cardiac_tissue.myo_indexes, self.dt,
                        self.ko, self.cao, self.nao, self.Vc, self.Vsr, self.Vss, self.Bufc, self.Kbufc, self.Bufsr, self.Kbufsr,
                        self.Bufss, self.Kbufss, self.Vmaxup, self.Kup, self.Vrel, self.k1_, self.k2_, self.k3, self.k4, self.EC,
                        self.maxsr, self.minsr, self.Vleak, self.Vxfer, self.R, self.F, self.T, self.RTONF, self.CAPACITANCE,
                        self.gkr, self.pKNa, self.gk1, self.gna, self.gbna, self.KmK, self.KmNa, self.knak, self.gcal, self.gbca,
                        self.knaca, self.KmNai, self.KmCa, self.ksat, self.n_, self.gpca, self.KpCa, self.gpk, self.gto, self.gks)

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
def calc_ina(u, dt, m, h, j, gna, Ena):
    """
    Calculates the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    m : np.ndarray
        Gating variable for sodium channels (activation).
    h : np.ndarray
        Gating variable for sodium channels (inactivation).
    j : np.ndarray
        Gating variable for sodium channels (inactivation).
    gna : float
        Sodium conductance.
    Ena : float
        Sodium reversal potential.

    Returns
    -------
    np.ndarray
        Updated fast sodium current array.
    """

    alpha_m = 1./(1.+np.exp((-60.-u)/5.))
    beta_m = 0.1/(1.+np.exp((u+35.)/5.)) + \
        0.10/(1.+np.exp((u-50.)/200.))
    tau_m = alpha_m*beta_m
    m_inf = 1./((1.+np.exp((-56.86-u)/9.03))
                * (1.+np.exp((-56.86-u)/9.03)))

    alpha_h = 0.
    beta_h = 0.
    if u >= -40.:
        alpha_h = 0.
        beta_h = 0.77/(0.13*(1.+np.exp(-(u+10.66)/11.1)))
    else:
        alpha_h = 0.057*np.exp(-(u+80.)/6.8)
        beta_h = 2.7*np.exp(0.079*u)+(3.1e5)*np.exp(0.3485*u)

    tau_h = 1.0/(alpha_h + beta_h)

    h_inf = 1./((1.+np.exp((u+71.55)/7.43))
                * (1.+np.exp((u+71.55)/7.43)))

    alpha_j = 0.
    beta_j = 0.
    if u >= -40.:
        alpha_j = 0.
        beta_j = 0.6*np.exp((0.057)*u)/(1.+np.exp(-0.1*(u+32.)))
    else:
        alpha_j = ((-2.5428e4)*np.exp(0.2444*u)-(6.948e-6) *
                np.exp(-0.04391*u))*(u+37.78) /\
            (1.+np.exp(0.311*(u+79.23)))
        beta_j = 0.02424*np.exp(-0.01052*u) / \
            (1.+np.exp(-0.1378*(u+40.14)))

    tau_j = 1.0/(alpha_j + beta_j)

    j_inf = h_inf

    m = m_inf-(m_inf-m)*np.exp(-dt/tau_m)
    h = h_inf-(h_inf-h)*np.exp(-dt/tau_h)
    j = j_inf-(j_inf-j)*np.exp(-dt/tau_j)

    return gna*m*m*m*h*j*(u-Ena), m, h, j

@njit
def calc_ical(u, dt, d, f, f2, fcass, cao, cass, gcal, F, R, T):
    """
    Calculates the L-type calcium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    d : np.ndarray
        Gating variable for L-type calcium channels.
    f : np.ndarray
        Gating variable for calcium-dependent calcium channels.
    f2 : np.ndarray
        Secondary gating variable for calcium-dependent calcium channels.
    fcass : np.ndarray
        Gating variable for calcium-sensitive current.
    cao : float
        Extracellular calcium concentration.
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    gcal : float
        Calcium conductance.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float

    Returns
    -------
    np.ndarray
        Updated L-type calcium current array.
    """

    d_inf = 1./(1.+np.exp((-8-u)/7.5))
    Ad = 1.4/(1.+np.exp((-35-u)/13))+0.25
    Bd = 1.4/(1.+np.exp((u+5)/5))
    Cd = 1./(1.+np.exp((50-u)/20))
    tau_d = Ad*Bd+Cd
    f_inf = 1./(1.+np.exp((u+20)/7))
    Af = 1102.5*np.exp(-(u+27)*(u+27)/225)
    Bf = 200./(1+np.exp((13-u)/10.))
    Cf = (180./(1+np.exp((u+30)/10)))+20
    tau_f = Af+Bf+Cf
    f2_inf = 0.67/(1.+np.exp((u+35)/7))+0.33
    Af2 = 600*np.exp(-(u+25)*(u+25)/170)
    Bf2 = 31/(1.+np.exp((25-u)/10))
    Cf2 = 16/(1.+np.exp((u+30)/10))
    tau_f2 = Af2+Bf2+Cf2
    fcass_inf = 0.6/(1+(cass/0.05)*(cass/0.05))+0.4
    tau_fcass = 80./(1+(cass/0.05)*(cass/0.05))+2.

    d = d_inf-(d_inf-d)*np.exp(-dt/tau_d)
    f = f_inf-(f_inf-f)*np.exp(-dt/tau_f)
    f2 = f2_inf-(f2_inf-f2)*np.exp(-dt/tau_f2)
    fcass = fcass_inf-(fcass_inf-fcass)*np.exp(-dt/tau_fcass)

    return gcal*d*f*f2*fcass*4*(u-15)*(F*F/(R*T)) *\
        (0.25*np.exp(2*(u-15)*F/(R*T))*cass-cao) / \
        (np.exp(2*(u-15)*F/(R*T))-1.), d, f, f2, fcass

@njit
def calc_ito(u, dt, r, s, Ek, gto):
    """
    Calculates the transient outward current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    r : np.ndarray
        Gating variable for ryanodine receptors.
    s : np.ndarray
        Gating variable for calcium-sensitive current.
    ek : float
        Potassium reversal potential.

    Returns
    -------
    np.ndarray
        Updated transient outward current array.
    """

    r_inf = 1./(1.+np.exp((20-u)/6.))
    s_inf = 1./(1.+np.exp((u+20)/5.))
    tau_r = 9.5*np.exp(-(u+40.)*(u+40.)/1800.)+0.8
    tau_s = 85.*np.exp(-(u+45.)*(u+45.)/320.) + \
        5./(1.+np.exp((u-20.)/5.))+3.

    s = s_inf-(s_inf-s)*np.exp(-dt/tau_s)
    r = r_inf-(r_inf-r)*np.exp(-dt/tau_r)

    return gto*r*s*(u-Ek), r, s

@njit
def calc_ikr(u, dt, xr1, xr2, Ek, gkr, ko):
    """
    Calculates the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    xr1 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    xr2 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    Ek : float
        Potassium reversal potential.
    gkr : float
        Potassium conductance.

    Returns
    -------
    np.ndarray
        Updated rapid delayed rectifier potassium current array.
    """

    xr1_inf = 1./(1.+np.exp((-26.-u)/7.))
    axr1 = 450./(1.+np.exp((-45.-u)/10.))
    bxr1 = 6./(1.+np.exp((u-(-30.))/11.5))
    tau_xr1 = axr1*bxr1
    xr2_inf = 1./(1.+np.exp((u-(-88.))/24.))
    axr2 = 3./(1.+np.exp((-60.-u)/20.))
    bxr2 = 1.12/(1.+np.exp((u-60.)/20.))
    tau_xr2 = axr2*bxr2

    xr1 = xr1_inf-(xr1_inf-xr1)*np.exp(-dt/tau_xr1)
    xr2 = xr2_inf-(xr2_inf-xr2)*np.exp(-dt/tau_xr2)

    return gkr*np.sqrt(ko/5.4)*xr1*xr2*(u-Ek), xr1, xr2

@njit
def calc_iks(u, dt, xs, Eks, gks):
    """
    Calculates the slow delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    dt : float
        Time step for the simulation.
    xs : np.ndarray
        Gating variable for slow delayed rectifier potassium channels.
    Eks : float
        Potassium reversal potential.
    gks : float
        Potassium conductance.
    
    Returns
    -------
    np.ndarray
        Updated slow delayed rectifier potassium current array.
    """
    xs_inf = 1./(1.+np.exp((-5.-u)/14.))
    Axs = (1400./(np.sqrt(1.+np.exp((5.-u)/6))))
    Bxs = (1./(1.+np.exp((u-35.)/15.)))
    tau_xs = Axs*Bxs+80
    xs_inf = 1./(1.+np.exp((-5.-u)/14.))

    xs = xs_inf-(xs_inf-xs)*np.exp(-dt/tau_xs)

    return gks*xs*xs*(u-Eks), xs

@njit
def calc_ik1(u, Ek, gk1):
    """
    Calculates the inward rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    gk1 : float
        Inward rectifier potassium conductance.

    Returns
    -------
    np.ndarray
        Updated inward rectifier potassium current array.
    """

    ak1 = 0.1/(1.+np.exp(0.06*(u-Ek-200)))
    bk1 = (3.*np.exp(0.0002*(u-Ek+100)) +
           np.exp(0.1*(u-Ek-10)))/(1.+np.exp(-0.5*(u-Ek)))
    rec_iK1 = ak1/(ak1+bk1)

    return gk1*rec_iK1*(u-Ek)

@njit
def calc_inaca(u, nao, nai, cao, cai, KmNai, KmCa, knaca, ksat, n_, F, R, T):
    """
    Calculates the sodium-calcium exchanger current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    nao : float
        Sodium ion concentration in the extracellular space.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    cao : float
        Calcium ion concentration in the extracellular space.
    cai : np.ndarray
        Calcium ion concentration in the submembrane space.
    KmNai : float
        Michaelis constant for sodium.
    KmCa : float
        Michaelis constant for calcium.
    knaca : float
        Sodium-calcium exchanger conductance.
    ksat : float
        Saturation factor.
    n_ : float
        Exponent for sodium dependence.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float
        Temperature.
    
    Returns
    -------
    np.ndarray
        Updated sodium-calcium exchanger current array.
    """

    return knaca*(1./(KmNai*KmNai*KmNai+nao*nao*nao))*(1./(KmCa+cao)) *\
            (1./(1+ksat*np.exp((n_-1)*u*F/(R*T)))) *\
            (np.exp(n_*u*F/(R*T))*nai*nai*nai*cao -
                np.exp((n_-1)*u*F/(R*T))*nao*nao*nao*cai*2.5)

@njit
def calc_inak(u, nai, ko, KmK, KmNa, knak, F, R, T):
    """
    Calculates the sodium-potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    ko : float
        Potassium ion concentration in the extracellular space.
    KmK : float
        Michaelis constant for potassium.
    KmNa : float
        Michaelis constant for sodium.
    knak : float
        Sodium-potassium pump conductance.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float
        Temperature.

    Returns
    -------
    np.ndarray
        Updated sodium-potassium pump current array.
    """

    rec_iNaK = (
        1./(1.+0.1245*np.exp(-0.1*u*F/(R*T))+0.0353*np.exp(-u*F/(R*T))))

    return knak*(ko/(ko+KmK))*(nai/(nai+KmNa))*rec_iNaK

@njit
def calc_ipca(cai, KpCa, gpca):
    """
    Calculates the calcium pump current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    KpCa : float
        Michaelis constant for calcium pump.
    gpca : float
        Calcium pump conductance.

    Returns
    -------
    np.ndarray
        Updated calcium pump current array.
    """

    return gpca*cai/(KpCa+cai)

@njit
def calc_ipk(u, Ek, gpk):
    """
    Calculates the potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    gpk : float
        Potassium pump conductance.
    
    Returns
    -------
    np.ndarray
        Updated potassium pump current array.
    """
    rec_ipK = 1./(1.+np.exp((25-u)/5.98))

    return gpk*rec_ipK*(u-Ek)

@njit
def calc_ibna(u, Ena, gbna):
    """
    Calculates the background sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ena : float
        Sodium reversal potential.
    gbna : float
        Background sodium conductance.

    Returns
    -------
    np.ndarray
        Updated background sodium current array.
    """

    return gbna*(u-Ena)

@njit
def calc_ibca(u, Eca, gbca):
    """
    Calculates the background calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Eca : float
        Calcium reversal potential.
    gbca : float
        Background calcium conductance.

    Returns
    -------
    np.ndarray
        Updated background calcium current array.
    """

    return gbca*(u-Eca)

@njit
def calc_irel(dt, rr, oo, casr, cass, vrel, k1, k2, k3, k4, maxsr, minsr, EC):
    """
    Calculates the ryanodine receptor current.

    Parameters
    ----------
    dt : float
        Time step for the simulation.
    rr : np.ndarray
        Ryanodine receptor gating variable for calcium release.
    oo : np.ndarray
        Ryanodine receptor gating variable for calcium release.
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    vrel : float
        Release rate of calcium from the sarcoplasmic reticulum.
    k1 : float
        Transition rate for SR calcium release.
    k2 : float
        Transition rate for SR calcium release.
    k3 : float
        Transition rate for SR calcium release.
    k4 : float
        Alternative transition rate.
    maxsr : float
        Maximum SR calcium release permeability.
    minsr : float
        Minimum SR calcium release permeability.
    EC : float
        Calcium-induced calcium release sensitivity.
    
    Returns
    -------
    np.ndarray
        Updated ryanodine receptor current array.
    """

    kCaSR = maxsr-((maxsr-minsr)/(1+(EC/casr)*(EC/casr)))
    k1_ = k1/kCaSR
    k2_ = k2*kCaSR
    drr = k4*(1-rr)-k2_*cass*rr
    rr += dt*drr
    oo = k1_*cass*cass * rr/(k3+k1_*cass*cass)

    return vrel*oo*(casr-cass), rr, oo

@njit
def calc_ileak(casr, cai, vleak):
    """
    Calculates the calcium leak current.

    Parameters
    ----------
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    vleak : float
        Leak rate of calcium from the sarcoplasmic reticulum.

    Returns
    -------
    np.ndarray
        Updated calcium leak current array.
    """

    return vleak*(casr-cai)

@njit
def calc_iup(cai, vmaxup, Kup):
    """
    Calculates the calcium uptake current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    vmaxup : float
        Uptake rate of calcium into the sarcoplasmic reticulum.
    Kup : float
        Michaelis constant for calcium uptake.

    Returns
    -------
    np.ndarray
        Updated calcium uptake current array.
    """

    return vmaxup/(1.+((Kup*Kup)/(cai*cai)))

@njit
def calc_ixfer(cass, cai, vxfer):
    """
    Calculates the calcium transfer current.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    vxfer : float
        Transfer rate of calcium between the submembrane space and cytosol.

    Returns
    -------
    np.ndarray
        Updated calcium transfer current array.
    """

    return vxfer*(cass-cai)

@njit
def calc_casr(dt, caSR, bufsr, Kbufsr, iup, irel, ileak):
    """
    Calculates the calcium concentration in the sarcoplasmic reticulum.

    Parameters
    ----------
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    bufsr : float
        Buffering capacity of the sarcoplasmic reticulum.
    Kbufsr : float
        Buffering constant of the sarcoplasmic reticulum.
    iup : float
        Calcium uptake current.
    irel : float
        Calcium release current.
    ileak : float
        Leak rate of calcium from the sarcoplasmic reticulum.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the sarcoplasmic reticulum.
    """

    CaCSQN = bufsr*caSR/(caSR+Kbufsr)
    dCaSR = dt*(iup-irel-ileak)
    bjsr = bufsr-CaCSQN-dCaSR-caSR+Kbufsr
    cjsr = Kbufsr*(CaCSQN+dCaSR+caSR)
    return (np.sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2

@njit
def calc_cass(dt, caSS, bufss, Kbufss, ixfer, irel, ical, capacitance, Vc, Vss, Vsr, inversevssF2):
    """
    Calculates the calcium concentration in the submembrane space.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    bufss : float
        Buffering capacity of the submembrane space.
    Kbufss : float
        Buffering constant of the submembrane space.
    ixfer : float
        Calcium transfer current.
    irel : float
        Calcium release current.
    ical : float
        L-type calcium current.
    capacitance : float
        Membrane capacitance.
    Vc : float
        Volume of the cytosol.
    Vss : float
        Volume of the submembrane space.
    Vsr : float
        Volume of the sarcoplasmic reticulum.
    inversevssF2 : float
        Inverse of the product of 2
        times the volume of the submembrane space and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the submembrane space.
    """

    CaSSBuf = bufss*caSS/(caSS+Kbufss)
    dCaSS = dt*(-ixfer*(Vc/Vss)+irel*(Vsr/Vss) +
                (-ical*inversevssF2*capacitance))
    bcss = bufss-CaSSBuf-dCaSS-caSS+Kbufss
    ccss = Kbufss*(CaSSBuf+dCaSS+caSS)
    return (np.sqrt(bcss*bcss+4*ccss)-bcss)/2

@njit
def calc_cai(dt, cai, bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer, capacitance, vsr, vc, inverseVcF2):
    """
    Calculates the calcium concentration in the cytosol.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the cytosol.
    bufc : float
        Buffering capacity of the cytosol.
    Kbufc : float
        Buffering constant of the cytosol.
    ibca : float
        Background calcium current.
    ipca : float
        Calcium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    iup : float
        Calcium uptake current.
    ileak : float
        Calcium leak current.
    ixfer : float
        Calcium transfer current.
    capacitance : float
        Membrane capacitance.
    vsr : float
        Volume of the sarcoplasmic reticulum.
    vc : float
        Volume of the cytosol.
    inverseVcF2 : float
        Inverse of the product of 2
        times the volume of the cytosol and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the cytosol.
    """

    CaCBuf = bufc*cai/(cai+Kbufc)
    dCai = dt*((-(ibca+ipca-2*inaca)*inverseVcF2*capacitance) -
                   (iup-ileak)*(vsr/vc)+ixfer)
    bc = bufc-CaCBuf-dCai-cai+Kbufc
    cc = Kbufc*(CaCBuf+dCai+cai)
    return (np.sqrt(bc*bc+4*cc)-bc)/2, cai

@njit
def calc_nai(dt, ina, ibna, inak, inaca, capacitance, inverseVcF):
    """
    Calculates the sodium concentration in the cytosol.

    Parameters
    ----------
    dt : float
        Time step for the simulation.
    ina : float
        Fast sodium current.
    ibna : float
        Background sodium current.
    inak : float
        Sodium-potassium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    capacitance : float
        Membrane capacitance.
    inverseVcF : float
        Inverse of the product of the volume of the cytosol and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated sodium concentration in the cytosol.
    """

    dNai = -(ina+ibna+3*inak+3*inaca)*inverseVcF*capacitance
    return dt*dNai

@njit
def calc_ki(dt, ik1, ito, ikr, iks, inak, ipk, inverseVcF, capacitance):
    """
    Calculates the potassium concentration in the cytosol.

    Parameters
    ----------
    ik1 : float
        Inward rectifier potassium current.
    ito : float
        Transient outward current.
    ikr : float
        Rapid delayed rectifier potassium current.
    iks : float
        Slow delayed rectifier potassium current.
    inak : float
        Sodium-potassium pump current.
    ipk : float
        Potassium pump current.
    capacitance : float
        Membrane capacitance.
    inverseVcF : float
        Inverse of the product of the volume of the cytosol and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated potassium concentration in the cytosol.
    """

    dKi = -(ik1+ito+ikr+iks-2*inak+ipk)*inverseVcF*capacitance
    return dt*dKi

# tp06 epi kernel
@njit(parallel=True)
def ionic_kernel_2d(u_new, u, cai, casr, cass, nai, Ki, m, h, j_, xr1, xr2,
                    xs, r, s, d, f, f2, fcass, rr, oo, indexes, dt, 
                    ko, cao, nao, Vc, Vsr, Vss, Bufc, Kbufc, Bufsr, Kbufsr,
                    Bufss, Kbufss, Vmaxup, Kup, Vrel, k1_, k2_, k3, k4, EC,
                    maxsr, minsr, Vleak, Vxfer, R, F, T, RTONF, CAPACITANCE,
                    gkr, pKNa, gk1, gna, gbna, KmK, KmNa, knak, gcal, gbca,
                    knaca, KmNai, KmCa, ksat, n_, gpca, KpCa, gpk, gto, gks):
    """
    Compute the ionic currents and update the state variables for the 2D TP06
    cardiac model.

    This function calculates the ionic currents based on the TP06 cardiac
    model, updates ion concentrations, and modifies gating variables in the
    2D grid. The calculations are performed in parallel to enhance performance.

    Parameters
    ----------
    u_new : numpy.ndarray
        Array to store the updated membrane potential values.
    u : numpy.ndarray
        Array of current membrane potential values.
    cai : numpy.ndarray
        Array of calcium concentration in the cytosol.
    casr : numpy.ndarray
        Array of calcium concentration in the sarcoplasmic reticulum.
    cass : numpy.ndarray
        Array of calcium concentration in the submembrane space.
    nai : numpy.ndarray
        Array of sodium ion concentration in the intracellular space.
    Ki : numpy.ndarray
        Array of potassium ion concentration in the intracellular space.
    m : numpy.ndarray
        Array of gating variable for sodium channels (activation).
    h : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    j_ : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    xr1 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    xr2 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    xs : numpy.ndarray
        Array of gating variable for slow delayed rectifier potassium channels.
    r : numpy.ndarray
        Array of gating variable for ryanodine receptors.
    s : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    d : numpy.ndarray
        Array of gating variable for L-type calcium channels.
    f : numpy.ndarray
        Array of gating variable for calcium-dependent calcium channels.
    f2 : numpy.ndarray
        Array of secondary gating variable for calcium-dependent calcium
        channels.
    fcass : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    rr : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    oo : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    indexes: numpy.ndarray
        Array of indexes where the kernel should be computed (``mesh == 1``).
    dt : float
        Time step for the simulation.

    Returns
    -------
    None
        The function updates the state variables in place. No return value is
        produced.
    """
    n_j = u.shape[1]

    inverseVcF2 = 1./(2*Vc*F)
    inverseVcF = 1./(Vc*F)
    inversevssF2 = 1./(2*Vss*F)

    for ind in prange(len(indexes)):
        ii = indexes[ind]
        i = int(ii/n_j)
        j = ii % n_j

        Ek = RTONF*(np.log((ko/Ki[i, j])))
        Ena = RTONF*(np.log((nao/nai[i, j])))
        Eks = RTONF*(np.log((ko+pKNa*nao)/(Ki[i, j]+pKNa*nai[i, j])))
        Eca = 0.5*RTONF*(np.log((cao/cai[i, j])))

        # Compute currents
        ina, m[i, j], h[i, j], j_[i, j] = calc_ina(u[i, j], dt, m[i, j], h[i, j], j_[i, j], gna, Ena)
        ical, d[i, j], f[i, j], f2[i, j], fcass[i, j] = calc_ical(u[i, j], dt, d[i, j], f[i, j], f2[i, j], fcass[i, j], cao, cass[i, j], gcal, F, R, T)
        ito, r[i, j], s[i, j] = calc_ito(u[i, j], dt, r[i, j], s[i, j], Ek, gto)
        ikr, xr1[i, j], xr2[i, j] = calc_ikr(u[i, j], dt, xr1[i, j], xr2[i, j], Ek, gkr, ko)
        iks, xs[i, j] = calc_iks(u[i, j], dt, xs[i, j], Eks, gks)
        ik1 = calc_ik1(u[i, j], Ek, gk1)
        inaca = calc_inaca(u[i, j], nao, nai[i, j], cao, cai[i, j], KmNai, KmCa, knaca, ksat, n_, F, R, T) 
        inak = calc_inak(u[i, j], nai[i, j], ko, KmK, KmNa, knak, F, R, T)
        ipca = calc_ipca(cai[i, j], KpCa, gpca)
        ipk = calc_ipk(u[i, j], Ek, gpk)
        ibna = calc_ibna(u[i, j], Ena, gbna)
        ibca = calc_ibca(u[i, j], Eca, gbca)
        irel, rr[i, j], oo[i, j] = calc_irel(dt, rr[i, j], oo[i, j], casr[i, j], cass[i, j], Vrel, k1_, k2_, k3, k4, maxsr, minsr, EC)
        ileak = calc_ileak(casr[i, j], cai[i, j], Vleak)
        iup = calc_iup(cai[i, j], Vmaxup, Kup)
        ixfer = calc_ixfer(cass[i, j], cai[i, j], Vxfer)

        # Compute concentrations
        casr[i, j] = calc_casr(dt, casr[i, j], Bufsr, Kbufsr, iup, irel, ileak)
        cass[i, j] = calc_cass(dt, cass[i, j], Bufss, Kbufss, ixfer, irel, ical, CAPACITANCE, Vc, Vss, Vsr, inversevssF2)
        cai[i, j], cai[i, j] = calc_cai(dt, cai[i, j], Bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer, CAPACITANCE, Vsr, Vc, inverseVcF2)
        nai[i, j] += calc_nai(dt, ina, ibna, inak, inaca, CAPACITANCE, inverseVcF)
        Ki[i, j] += calc_ki(dt, ik1, ito, ikr, iks, inak, ipk, inverseVcF, CAPACITANCE)

        # Update membrane potential
        u_new[i, j] -= dt * (ikr + iks + ik1 + ito + ina + ibna + ical + ibca + inak + inaca + ipca + ipk)
        
