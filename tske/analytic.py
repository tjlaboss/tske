"""
Analytic

Analytically guess flux profile based on boundary conditions.
Assumes a 1D slab with a uniform material and kinf > 1.
"""

import enum
import numpy as np
from tske.tping import T_arr


class BoundaryConditions(enum.IntEnum):
	reflective = 1
	absorptive = 2


def sine(nx) -> T_arr:
	"""Truncated sine"""
	theta_vals = np.linspace(0, np.pi, nx + 2)
	sine_wave = np.sin(theta_vals)[1:-1]
	sine_wave /= sine_wave.mean()
	return sine_wave


def half_cosine(nx) -> T_arr:
	"""Truncated half-cosine"""
	theta_vals = np.linspace(0, np.pi/2, nx + 1)
	cosine_wave = np.cos(theta_vals)[:-1]
	cosine_wave /= cosine_wave.mean()
	return cosine_wave


def guess_flux(bc_w: int, bc_e: int, nx: int) -> T_arr:
	"""Guess the flux analytically based on the boundary conditions.
	
	Parameters:
	-----------
	bc_w: BoundaryConditions
		West boundary condition
	
	bc_e: BoundaryConditions
		East boundary condition
	
	nx: int
		Number of spatial nodes.
	
	Returns:
	--------
	phi: np.ndarray
		Vector of len(nx) of the starting flux profile.
	"""
	if bc_w == BoundaryConditions.reflective and bc_e == BoundaryConditions.reflective:
		# Flat.
		return np.ones(nx)
	if bc_w == BoundaryConditions.reflective and bc_e == BoundaryConditions.absorptive:
		# Half cosine (forward)
		return half_cosine(nx)
	if bc_w == BoundaryConditions.absorptive and bc_e == BoundaryConditions.reflective:
		# Half cosine (backward)
		return half_cosine(nx)[::-1]
	if bc_w == BoundaryConditions.absorptive and bc_e == BoundaryConditions.absorptive:
		# (Co)sine
		return sine(nx)
	raise NotImplementedError(f"{bc_w} | {bc_e}")
