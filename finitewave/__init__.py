
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
    # Courtemanche,
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
