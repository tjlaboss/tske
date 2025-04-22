"""
Analytic

Analytically guess flux profile based on boundary conditions
"""

import enum
import numpy as np


class BoundaryConditions(enum.IntEnum):
	reflective = 1
	absorptive = 2



