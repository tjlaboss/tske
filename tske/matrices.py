"""
Matrices

Matrix builders for Spatial Kinetics Equations
"""

import numpy as np
import typing
import enum
from tske import keys
from tske.tping import T_arr, T_nodearr


class Methods(enum.IntEnum):
	explicit = 0
	implicit = 1
	cranknic = 2


def __check_inputs(n, rho_vec, betas, lams):
	"""Check the lenghts of inputs for matrix builders."""
	len_rho = len(rho_vec)
	assert len_rho == n, f"Expected {n} reactivities; got {len_rho}."
	len_beta = len(betas)
	len_lams = len(lams)
	assert len_beta == len_lams, \
		(f"Number of delayed neutron fractions ({len_beta}) does not match"
		 f"number of delayed neutron decay constants ({len_lams}).")


def __crank_precursors(
		ijk, n, i, ndg, A, lams, betas, L, v1, dt,
		explicit, implicit, cranknic
):
	# Flux/power is at index 0
	ixc0 = ijk(n+0, i, 0)  # phi(i, n)
	ixc1 = ijk(n+1, i, 0)  # phi(i, n+1)
	for k in range(ndg):
		# Generate space, time, and precursor indices
		ic0 = ijk(n+0, i, k+1)  # i, n,   precursor k
		ic1 = ijk(n+1, i, k+1)  # i, n+1, precursor k
		# Delayed source in flux (TO flux, FROM precursors)
		A[ixc1, ic0] = 0 if implicit else -lams[k]*v1
		A[ixc1, ic1] = 0 if explicit else -lams[k]*v1
		# Precursor source (TO precursors, FROM flux)
		A[ic1, ixc0] = 0 if implicit else betas[k]/L
		A[ic1, ixc1] = 0 if explicit else betas[k]/L
		# Precursor sink (TO precursors, FROM precursors in time)
		invtime = 2/dt if cranknic else 1/dt
		n0_term = 0 if implicit else -lams[k]
		n1_term = 0 if explicit else -lams[k]
		# Include factor of 2 in time numerator
		A[ic1, ic0] = +invtime + n0_term
		A[ic1, ic1] = -invtime + n1_term

def crank_nicolson(
		dt: float,
		betas: T_arr,
		lams: T_arr,
		L: float,
		v1: float,
		dx: float,
		nodes: T_nodearr,
		P0: typing.Sequence[float],
		method=Methods.cranknic
) -> typing.Tuple[T_arr, T_arr]:
	"""Build A and B matrices using Crank-Nicolson.
	
	Crank-Nicolson = 1/2 implicit + 1/2 Explicit
	
	
	Parameters:
	-----------
	dt: float
		Timestep size (s).
	
	betas: np.ndarray(float)
		Array of delayed neutron precursor fission yields.
	
	lams: np.ndarray(float)
		Array of delayed neutron precursor decay constants (s^-1).
	
	L: float
		Prompt neutron lifetime (s).
	
	v1: float
		Inverse neutron velocity (s/cm).
	
	dx: float
		Spatial mesh size (cm).
	
	nodes: (nx, nt) array of Node1D
		Spatial nodes
		
	P0: sequence of float
		Starting flux. Must match len(nodes).
	
	method: int
		0 = explicit Euler
		1 = implicit Euler
		2 = Crank-Nicolson
	
	Returns:
	--------
	A: np.ndarray
		Square [NxN] array, for LHS of matrix solution.
	
	B: np.ndarray
		Vector [Nx1] array, for RHS of matrix solution.
	"""
	assert method in range(3)
	explicit = (method == Methods.explicit)
	implicit = (method == Methods.implicit)
	cranknic = (method == Methods.cranknic)
	
	# __check_inputs(n, rho_vec, betas, lams)
	ndg = len(betas)	# number of delayed groups
	beff = sum(betas)   # beta effective

	nx, nt = nodes.shape
	size = (1 + ndg)*nt*nx
	
	def ijk(index_t: int, index_x: int, index_c: int) -> int:
		"""convert 3 indices to 1 index"""
		return index_x + index_t*nx + index_c*nt*nx
		# return index_t + index_x*nt + index_c*nt*nx

	A = np.zeros((size, size))
	B = np.zeros(size)
	C0s = (P0*betas)/(lams*L)  # Initial precursor concentrations
	
	invdt = v1/dt
	if cranknic:
		invdt *= 2 # multiply this by 2 instead of dividing everything else by 2
	
	for n in range(nt - 1):
		def gen_precursors(i_x):
			__crank_precursors(ijk=ijk, n=n, i=i_x, ndg=ndg, A=A, lams=lams, betas=betas, v1=v1, L=L, dt=dt,
			                   explicit=explicit, implicit=implicit, cranknic=cranknic)
		# interior nodes
		for i in range(1, nx - 1):
			# Coupling coefficients: assumes uniform dx
			Dl0 = nodes[i, n+0].get_Dhat(nodes[i-1, n+0])/dx
			Dr0 = nodes[i, n+0].get_Dhat(nodes[i+1, n+0])/dx
			Dl1 = nodes[i, n+1].get_Dhat(nodes[i-1, n+1])/dx
			Dr1 = nodes[i, n+1].get_Dhat(nodes[i+1, n+1])/dx
			# other xs
			sig_a0 = nodes[i, n+0].sigmaA
			sig_a1 = nodes[i, n+1].sigmaA
			sig_f0 = nodes[i, n+0].nuSigmaF
			sig_f1 = nodes[i, n+1].nuSigmaF
			# Generate space and time indices
			ixl0 = ijk(n+0, i-1, 0)  # phi(i-1, n)
			ixc0 = ijk(n+0, i+0, 0)  # phi(i,   n)
			ixr0 = ijk(n+0, i+1, 0)  # phi(i+1, n)
			ixl1 = ijk(n+1, i-1, 0)  # phi(i-1, n+1)
			ixc1 = ijk(n+1, i+0, 0)  # phi(i,   n+1)
			ixr1 = ijk(n+1, i+1, 0)  # phi(i+1, n+1)
			# Center node
			n0_term = 0 if implicit else -Dl0 - Dr0 - sig_a0 + sig_f0
			n1_term = 0 if explicit else -Dl1 - Dr1 - sig_a1 + sig_f1
			A[ixc1, ixc0] = +invdt + n0_term  # i, n
			A[ixc1, ixc1] = -invdt + n1_term  # i, n+1
			# Adjacent nodes
			A[ixc1, ixl0] = 0 if implicit else Dl0  # i-1, n
			A[ixc1, ixr0] = 0 if implicit else Dr0  # i+1, n
			A[ixc1, ixl1] = 0 if explicit else Dl1  # i-1, n+1
			A[ixc1, ixr1] = 0 if explicit else Dr1  # i+1, n+1
			#
			gen_precursors(i_x=i)
		
		# Boundary conditions
		# Forgive the copy-paste
		
		# West boundary condition (hardcoded reflective for now)
		iwc0 = ijk(n+0, 0, 0)
		iwc1 = ijk(n+1, 0, 0)
		iwr0 = ijk(n+0, 1, 0)
		iwr1 = ijk(n+1, 1, 0)
		sig_a0 = nodes[0, n+0].sigmaA
		sig_a1 = nodes[0, n+1].sigmaA
		sig_f0 = nodes[0, n+0].nuSigmaF
		sig_f1 = nodes[0, n+1].nuSigmaF
		# Coupling coefficients: assumes uniform dx
		Dr0 = nodes[0, n+0].get_Dhat(nodes[1, n+0])/dx
		Dr1 = nodes[0, n+1].get_Dhat(nodes[1, n+1])/dx
		# West node
		n0_term = 0 if implicit else -Dr0 - sig_a0 + sig_f0
		n1_term = 0 if explicit else -Dr1 - sig_a1 + sig_f1
		A[iwc1, iwc0] = +invdt + n0_term  # 0, n
		A[iwc1, iwc1] = -invdt + n1_term  # 0, n+1
		# Node 1 (right)
		A[iwc1, iwr0] = 0 if implicit else Dr0
		A[iwc1, iwr1] = 0 if explicit else Dr1
		#
		gen_precursors(i_x=0)
		
		# East boundary condition (hardcoded reflective for now)
		X = nx - 1
		iec0 = ijk(n+0, X-0, 0)
		iec1 = ijk(n+1, X-0, 0)
		iel0 = ijk(n+0, X-1, 0)
		iel1 = ijk(n+1, X-1, 0)
		sig_a0 = nodes[X, n+0].sigmaA
		sig_a1 = nodes[X, n+1].sigmaA
		sig_f0 = nodes[X, n+0].nuSigmaF
		sig_f1 = nodes[X, n+1].nuSigmaF
		# Coupling coefficients: assumes uniform dx
		Dl0 = nodes[X, n+0].get_Dhat(nodes[X-1, n+0])/dx
		Dl1 = nodes[X, n+1].get_Dhat(nodes[X-1, n+1])/dx
		# East node
		n0_term = 0 if implicit else -Dl0 - sig_a0 + sig_f0
		n1_term = 0 if explicit else -Dl1 - sig_a1 + sig_f1
		A[iec1, iec0] = +invdt + n0_term  # X, n
		A[iec1, iec1] = -invdt + n1_term  # X, n+1
		# Node X-1 (left)
		A[iec1, iel0] = 0 if implicit else Dl0
		A[iec1, iel1] = 0 if explicit else Dl1
		#
		gen_precursors(i_x=X)
		
	# Initial condition
	for i in range(nx):
		i00 = ijk(0, i, 0)
		A[i00, i00] = 1
		B[i00] = P0[i]
		for k in range(ndg):
			ik0 = ijk(0, i, k+1)
			A[ik0, ik0] = 1
			B[ik0] = C0s[k]
				
	return A, B


METHODS = {
	key: Methods.implicit for key in keys.IMPLICIT_NAMES
} | {
	key: Methods.explicit for key in keys.EXPLICIT_NAMES
} | {
	key: Methods.cranknic for key in keys.CRANKNIC_NAMES
}

