"""Numerical defaults and unit conversions used by MF6PQC.

Case-specific scientific parameters should be supplied through configuration;
constants here exist only for stable legacy defaults and numerical safeguards.
"""

SECONDS_PER_DAY = 86_400.0

# MF6PQC sends seconds explicitly through SetTime and SetTimeStep.
PHREEQCRM_TIME_CONVERSION = 1.0
PHREEQCRM_REBALANCE_FRACTION = 0.5

# PhreeqcRM entity amount-unit codes.  Code 0 for solid entities is the legacy
# MF6PQC contract required by the incremental d_<mineral> porosity calculation.
PHREEQCRM_UNITS = {
    "solution": 2,
    "ppassemblage": 0,
    "exchange": 0,
    "surface": 0,
    "gas_phase": 0,
    "ssassemblage": 0,
    "kinetics": 0,
}
# Seven blocks expected by InitialPhreeqc2Module(_mix).
MODULE_INDICES = {
    "solution": 0,
    "equilibrium_phases": 1,
    "exchange": 2,
    "surface": 3,
    "gas_phase": 4,
    "solid_solutions": 5,
    "kinetics": 6,
}
IC_DEFAULT = -1

MIN_POROSITY = 1.0e-4
MAX_POROSITY = 1.0
MIN_CONCENTRATION = 1.0e-20
# Coupling time steps are expressed in MODFLOW TDIS units (days in examples).
MIN_TIME_STEP = 1.0e-30

# Legacy vertical/horizontal conductivity ratio; configurable per simulation.
K33_RATIO = 0.6

# Compatibility defaults retained for external imports.
SOURCE_RELAXATION = 0.5
SIA_MAX_PICARD_ITER = 2000
SIA_RTOL = 1.0e-4
SIA_ATOL = 1.0e-9
DENSITY_RELAXATION = 0.5

# PhreeqcRM densities are kg/L; MODFLOW BUY expects kg/m3.
DENSITY_SCALE = 1000.0

# Default mineral molar volumes in L/mol.  Values are part of the model input,
# not universal software constants: cases should override them when their
# database, phase definition, or reference source differs.
VM_MINERALS = {
    "Calcite": 0.03693,
    "Dolomite": 0.0645,
    "Halite": 0.0271,
    "Carnallite": 0.1737,
    "Polyhalite": 0.2180,
    "Sylvite": 0.0375,
    "Gypsum": 0.07421,
    "Bischofite": 0.1271,
    "Syngenite": 0.1273,
    "Ferrihydrite": 0.02399,
    "Jarosite": 0.15463,
    "Gibbsite": 0.03319,
    "Siderite": 0.02926,
}
