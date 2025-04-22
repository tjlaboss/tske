"""
Matrices

Matrix builders for Point Kinetics Equations
"""

import numpy as np
import typing
from tske import keys
from tske.tping import T_arr, T_nodearr


def __check_inputs(n, rho_vec, betas, lams):
	"""Check the lenghts of inputs for matrix builders."""
	len_rho = len(rho_vec)
	assert len_rho == n, f"Expected {n} reactivities; got {len_rho}."
	len_beta = len(betas)
	len_lams = len(lams)
	assert len_beta == len_lams, \
		(f"Number of delayed neutron fractions ({len_beta}) does not match"
		 f"number of delayed neutron decay constants ({len_lams}).")


def __crank_precursors(ijk, n, i, ndg, A, lams, betas, L, dt):
	# Flux/power is at index 0
	ixc0 = ijk(n+0, i, 0)  # phi(i, n)
	ixc1 = ijk(n+1, i, 0)  # phi(i, n+1)
	for k in range(ndg):
		# Generate space, time, and precursor indices
		ic0 = ijk(n+0, i, k+1)  # i, n,   precursor k
		ic1 = ijk(n+1, i, k+1)  # i, n+1, precursor k
		# Delayed source in flux (TO flux, FROM precursors)
		A[ixc1, ic0] = lams[k]
		A[ixc1, ic1] = lams[k]
		# Precursor source (TO precursors, FROM flux)
		A[ic1, ixc0] = betas[k]/L
		A[ic1, ixc1] = betas[k]/L
		# Precursor sink (TO precursors, FROM precursors in time)
		# Include factor of 2 in time numerator
		A[ic1, ic0] = +2/dt - lams[k]
		A[ic1, ic1] = -2/dt - lams[k]

def crank_nicolson(
		dt: float,
		betas: T_arr,
		lams: T_arr,
		L: float,
		v1: float,
		dx: float,
		nodes: T_nodearr,
		P0: typing.Sequence[float]
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
	
	Returns:
	--------
	A: np.ndarray
		Square [NxN] array, for LHS of matrix solution.
	
	B: np.ndarray
		Vector [Nx1] array, for RHS of matrix solution.
	"""
	# __check_inputs(n, rho_vec, betas, lams)
	ndg = len(betas)	# number of delayed groups
	beff = sum(betas)   # beta effective
	# rho_vec *= beff	 # convert from $

	nx, nt = nodes.shape
	size = (1 + ndg)*nt*nx
	
	def ijk(index_t: int, index_x: int, index_c: int) -> int:
		"""convert 3 indices to 1 index"""
		#return index_t + index_x*nt + index_c*nt*(ndg+1)
		return index_t + index_x*nt + index_c*nt*nx

	A = np.zeros((size, size))
	B = np.zeros(size)
	C0s = (P0*betas)/(lams*L)  # Initial precursor concentrations
	
	# 2 / (v*dt) : multiply this by 2 instead of dividing everything else by 2
	invdt2 = 2*v1/dt
	
	for n in range(nt - 1):
		def gen_precursors(i_x):
			__crank_precursors(ijk=ijk, n=n, i=i_x, ndg=ndg, A=A, lams=lams, betas=betas, L=L, dt=dt)
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
			A[ixc1, ixc0] = +invdt2 - Dl0 - Dr0 - sig_a0 + sig_f0  # i, n
			A[ixc1, ixc1] = -invdt2 - Dl1 - Dr1 - sig_a1 + sig_f1  # i, n+1
			# Adjacent nodes
			A[ixc1, ixl0] = Dl0  # i-1, n
			A[ixc1, ixr0] = Dr0  # i+1, n
			A[ixc1, ixl1] = Dl1  # i-1, n+1
			A[ixc1, ixr1] = Dr1  # i+1, n+1
			#
			gen_precursors(i_x=i)
		
		# Boundary conditions
		# Forgive the copy-paste
		
		# West boundary condition (hardcoded dirichlet 0 for now)
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
		# FIXME: not exactly right but good enough for testing
		A[iwc1, iwc0] = +invdt2 - 2*Dr0 - sig_a0 + sig_f0  # 0, n
		A[iwc1, iwc1] = -invdt2 - 2*Dr1 - sig_a1 + sig_f1  # 0, n+1
		# Node 1 (right)
		A[iwc1, iwr0] = Dr0
		A[iwc1, iwr1] = Dr1
		#
		gen_precursors(i_x=0)
		
		# East boundary condition (hardcoded direchlet 0 for now)
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
		# FIXME: not exactly right but good enough for testing
		A[iec1, iec0] = +invdt2 - 2*Dl0 - sig_a0 + sig_f0  # X, n
		A[iec1, iec1] = -invdt2 - 2*Dl1 - sig_a1 + sig_f1  # X, n+1
		# Node X-1 (left)
		A[iec1, iel0] = Dl0
		A[iec1, iel1] = Dl1
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


def implicit_euler(
		n: int,
		rho_vec: T_arr,
		dt: float,
		betas: T_arr,
		lams: T_arr,
		L: float,
		P0: float=1,
) -> typing.Tuple[T_arr, T_arr]:
	"""Build A and B matrices using Implicit Euler.
	
	Example for 1 delayed group:
	
		[-1] P_n  +  [1 - dt*(rho - beta)/L] P_{n+1}  +                    [-dt*lambda_k] C_{k,n+1} = 0
		[+1] P_0                                                                                    = P0
		             [-dt*beta_k/Lambda]     P_{n+1}  +  [-1] C_{k,n} + [1 + dt*lambda_k] C_{k,n+1} = 0
		                                                 [+1] C_{k,n}                               = C0_k
	
	
	Parameters:
	-----------
	n: int
		Number of timesteps
	
	rho_vec: np.ndarray(float)
		Array of reactivities at each timestep ($).
	
	dt: float
		Timestep size (s).
	
	betas: np.ndarray(float)
		Array of delayed neutron precursor fission yields.
	
	lams: np.ndarray(float)
		Array of delayed neutron precursor decay constants (s^-1).
	
	L: float
		Prompt neutron lifetime (s).
		
	P0: float, optional.
		Starting power.
		[Default: 1]
	
	Returns:
	--------
	A: np.ndarray
		Square [NxN] array, for LHS of matrix solution.
	
	B: np.ndarray
		Vector [Nx1] array, for RHS of matrix solution.
	"""
	__check_inputs(n, rho_vec, betas, lams)
	ndg = len(betas)    # number of delayed groups
	beff = sum(betas)   # beta effective
	rho_vec *= beff     # convert from $
	size = (1 + ndg)*n
	A = np.zeros((size, size))
	B = np.zeros(size)
	C0s = (P0*betas)/(lams*L)  # Initial precursor concentrations
	for ip in range(n-1):
		rho1 = rho_vec[ip+1]
		dtrbl = dt*(rho1 - beff)/L
		# P, normal nodes
		A[ip, ip] = -1            # P_n
		A[ip, ip+1] =  1 - dtrbl  # P_{n+1}
		for k in range(ndg):
			ic = ip + n*(k+1)
			A[ip, ic+1] = -dt*lams[k]       # C_{k,n+1}
			# C, normal nodes
			A[ic, ip+1] = -dt*betas[k]/L    # P_{n+1}
			A[ic, ic] = -1                  # C_{n,k}
			A[ic, ic+1] = 1 + dt*lams[k]    # C_{n,k+1}
	# Boundary Conditions
	# Initial Condition: P
	A[n-1, 0] = 1
	B[n-1] = P0
	# Initial Condition: C
	for k in range(ndg):
		A[n*(k+2)-1, n*(k+1)] = 1
		B[n*(k+2)-1] = C0s[k]
	return A, B


def explicit_euler(
		n: int,
		rho_vec: T_arr,
		dt: float,
		betas: T_arr,
		lams: T_arr,
		L: float,
		P0: float = 1,
) -> typing.Tuple[T_arr, T_arr]:
	"""Build A and B matrices using Explicit Euler.

	Example for 1 delayed group:

		[-1 - dt*(rho - beta)/L] P_n  +  [+1] P_{n+1}  + [-dt*lambda_k] C_{k,n}                    = 0
		[+1]                     P_0                                                               = P0
		[-dt*beta_k/Lambda]      P_n  +              [-1 + dt*lambda_k] C_{k,n}  +  [+1] C_{k,n+1} = 0
		                                                           [+1] C_{k,n}                    = C0_k


	Parameters:
	-----------
	n: int
		Number of timesteps

	rho_vec: np.ndarray(float)
		Array of reactivities at each timestep ($).

	dt: float
		Timestep size (s).

	betas: np.ndarray(float)
		Array of delayed neutron precursor fission yields.

	lams: np.ndarray(float)
		Array of delayed neutron precursor decay constants (s^-1).

	L: float
		Prompt neutron lifetime (s).

	P0: float, optional.
		Starting power.
		[Default: 1]

	Returns:
	--------
	A: np.ndarray
		Square [NxN] array, for LHS of matrix solution.

	B: np.ndarray
		Vector [Nx1] array, for RHS of matrix solution.
	"""
	__check_inputs(n, rho_vec, betas, lams)
	ndg = len(betas)    # number of delayed groups
	beff = sum(betas)   # beta effective
	rho_vec *= beff     # convert from $
	size = (1 + ndg)*n
	A = np.zeros((size, size))
	B = np.zeros(size)
	C0s = (P0*betas)/(lams*L)  # Initial precursor concentrations
	for ip in range(n - 1):
		rho0 = rho_vec[ip]
		dtrbl = dt*(rho0 - beff)/L
		# P, normal nodes
		A[ip, ip] = -1 - dtrbl  # P_n
		A[ip, ip+1] = 1         # P_{n+1}
		for k in range(ndg):
			ic = ip + n*(k + 1)
			A[ip, ic] = -dt*lams[k]     # C_{k,n}
			# C, normal nodes
			A[ic, ip] = -dt*betas[k]/L  # P_{n}
			A[ic, ic] = -1 + dt*lams[k] # C_{k,n}
			A[ic, ic+1] = 1             # next C
	# Boundary Conditions
	# Initial Condition: P
	A[n-1, 0] = 1
	B[n-1] = P0
	# Initial Condition: C
	for k in range(ndg):
		A[n*(k+2)-1, n*(k+1)] = 1
		B[n*(k+2)-1] = C0s[k]
	return A, B


METHODS = {
	key: implicit_euler for key in keys.IMPLICIT_NAMES
} | {
	key: explicit_euler for key in keys.EXPLICIT_NAMES
}

