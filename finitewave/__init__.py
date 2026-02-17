
"""
finitewave
==========

A Python package for simulating cardiac electrophysiology in 2D and 3D using
the finite difference method.

This package provides a set of tools for simulating cardiac electrophysiology
in 2D and 3D using the finite difference method. The package includes classes
for creating cardiac tissue models, tracking electrical activity, and
visualizing simulation results. The package is designed to be flexible and
extensible, allowing users to create custom models and trackers for their
specific research needs.

"""

from finitewave.core import (
    Command,
    CommandSequence,
    FibrosisPattern,
    CardiacModel,
    StateLoader,
    StateSaver,
    StateSaverCollection,
    Stencil,
    StimCurrent,
    StimSequence,
    StimVoltage,
    Stim,
    CardiacTissue,
    Tracker,
    TrackerSequence
)

from finitewave.cpuwave import (
    # IncorrectWeightsModeError2D,
    DiffusePattern,
    StructuralPattern,
    AlievPanfilov,
    Barkley,
    MitchellSchaeffer,
    FentonKarma,
    BuenoOrovio,
    LuoRudy91,
    TenTusscherPanfilov2006,
    Courtemanche,
    AsymmetricStencil2D,
    # SymmetricStencil2D,
    IsotropicStencil2D,
    StimCurrentArea,
    StimCurrentCoord,
    StimVoltageCoord,
    StimCurrentMatrix,
    StimVoltageMatrix,
    ActionPotentialTracker,
    ActivationTimeTracker,
    AnimationTracker,
    ECGTracker,
    LocalActivationTimeTracker,
    PeriodTracker,
    # PeriodAnimationTracker,
    SpiralWaveCoreTracker,
    VariablesTracker,
)

from finitewave.tools import (
    Animation2DBuilder,
    Animation3DBuilder,
    VisMeshBuilder3D,
    Velocity2DCalculation,
    Velocity3DCalculation,
)


# compatibility with older versions:

CardiacTissue2D = CardiacTissue
CardiacTissue3D = CardiacTissue

AlievPanfilov2D = AlievPanfilov
AlievPanfilov3D = AlievPanfilov
Barkley2D = Barkley
Barkley3D = Barkley
MitchellSchaeffer2D = MitchellSchaeffer
MitchellSchaeffer3D = MitchellSchaeffer
FentonKarma2D = FentonKarma
FentonKarma3D = FentonKarma
BuenoOrovio2D = BuenoOrovio
BuenoOrovio3D = BuenoOrovio
LuoRudy912D = LuoRudy91
LuoRudy913D = LuoRudy91
TP062D = TenTusscherPanfilov2006
TP063D = TenTusscherPanfilov2006
Courtemanche2D = Courtemanche
Courtemanche3D = Courtemanche

StimCurrentArea2D = StimCurrentArea
StimCurrentArea3D = StimCurrentArea
StimCurrentCoord2D = StimCurrentCoord
StimCurrentCoord3D = StimCurrentCoord
StimVoltageCoord2D = StimVoltageCoord
StimVoltageCoord3D = StimVoltageCoord
StimCurrentMatrix2D = StimCurrentMatrix
StimCurrentMatrix3D = StimCurrentMatrix
StimVoltageMatrix2D = StimVoltageMatrix
StimVoltageMatrix3D = StimVoltageMatrix

ActionPotential2DTracker = ActionPotentialTracker
ActionPotential3DTracker = ActionPotentialTracker
ActivationTime2DTracker = ActivationTimeTracker
ActivationTime3DTracker = ActivationTimeTracker
Animation2DTracker = AnimationTracker
Animation3DTracker = AnimationTracker
ECG2DTracker = ECGTracker
ECG3DTracker = ECGTracker
LocalActivationTime2DTracker = LocalActivationTimeTracker
LocalActivationTime3DTracker = LocalActivationTimeTracker
Period2DTracker = PeriodTracker
Period3DTracker = PeriodTracker
SpiralWaveCore2DTracker = SpiralWaveCoreTracker
SpiralWaveCore3DTracker = SpiralWaveCoreTracker
Variables2DTracker = VariablesTracker
Variables3DTracker = VariablesTracker
MultiVariable2DTracker = VariablesTracker
MultiVariable3DTracker = VariablesTracker
