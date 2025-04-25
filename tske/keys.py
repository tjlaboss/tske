"""
Keys

Hardcoding is bad. Use these constants instead.
"""

IMPLICIT_NAMES = ("implicit euler", "implicit", "backward euler", "backward")
EXPLICIT_NAMES = ("explicit euler", "explicit", "forward euler", "forward")
CRANKNIC_NAMES = ("crank-nicolson", "cranknic", "crank", "nicolson")
METH = "method"

# Reactivity functions
REAC = "reactivity"
REAC_TYPE = "type"
RHO = "rho"
STEP = "step"
RAMP = "ramp"
RAMP_SLOPE = "slope"
SINE = "sine"
SINE_OMEGA = "frequency"

# Time options
TIME = "time"
TIME_TOTAL = "total"
TIME_DELTA = "dt"

# Plot options
PLOT = "plots"
PLOT_SHOW = "show"
PLOT_SPY = "spy"
PLOT_PR = "power_reactivity"
PLOT_LOG = "plot_type"
PLOT_LINEAR = "linear"
PLOT_SEMLOG = "semilog"
PLOT_LOGLOG = "loglog"
PLOT_TYPES = (PLOT_LINEAR, PLOT_SEMLOG, PLOT_LOGLOG)

# Data inputs
DATA = "data"

# PKRE data
DATA_B = "delay_fractions"
DATA_L = "decay_constants"
DATA_BIG_L = "Lambda"
DATA_IV = "ivelocity"

# Diffusion data
# Cross sections
MATERIALS = "materials"
XS_SIGMA_A = "sigma_a"
XS_D = "diffusion_coefficient"
XS_NUSIGMA_FP = "prompt_nu_sigma_f"
# Geometry
GEOM = "geometry"
GEOM_DX = "delta_x"
NODES = "nodes"
NODE_MATERIAL = "material"
NODE_SWAPS = "swaps"
# Boundary conditions
NODE_BC = "boundary_condition"
BC_ZERO = "absorptive"      # Zero flux
BC_VACUUM = "vacuum"        # Zero incoming current
BC_MIRROR = "reflective"    # Zero current
BC_TYPES = (BC_ZERO, BC_VACUUM, BC_MIRROR)


# Plot names
EXT = ".pdf"  # consider making this user-configurable
FNAME_SPY = "spy" + EXT
FNAME_PR = "power_reactivity" + EXT
FNAME_FLUX2 = "flux2d" + EXT
FNAME_FLUX3 = "flux3d" + EXT
FNAME_CONVERGE = "timestep_study" + EXT

# Text names
FNAME_CFG = "config.yml"
FNAME_TIME = "times.txt"
FNAME_RHO = "reactivities.txt"
FNAME_P = "powers.txt"
FNAME_FMT_C = "concentrations{}.txt"
FNAME_C = FNAME_FMT_C.format("")
FNAME_MATRIX_A = "A.txt"
FNAME_MATRIX_B = "B.txt"
FNAME_DT = "dt.txt"
FNAME_REPORT = "timestep_report.txt"
