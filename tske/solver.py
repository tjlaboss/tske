"""
Solver

Solve the system of equations

Right now, we only use scipy.linalg.solve().
It'd probably be more efficient to use scipy's sparse matrix methods
or to write my own Gauss-Seidel iterator, but that's not a priority.
"""

import scipy.linalg as la
import scipy.sparse.linalg as spla
from tske.tping import T_arr


def __split_results(vecX: T_arr, nx: int, nt: int):
	"""Split power and precusor concentration results
	
	Parameters:
	----------
	vecX: np.ndarray
		1D array of the solution results
		
	nx: int
		Number of spatial nodes
	
	nt: int
		Number of timesteps
	"""
	n = nx*nt
	ndg = len(vecX)//n - 1
	P = vecX[:n].reshape((nt, nx)).T
	C = vecX[n:].reshape((ndg, nt, nx))
	Clist = [C[k, ...].T for k in range(ndg)]
	return P, Clist


def sparse_linalg(matA, matB, nx, nt):
	"""Solve using scipy sparse matrix methods"""
	vecX = spla.spsolve(matA.tocsr(), matB.tocsr())
	return __split_results(vecX, nx, nt)


def linalg(matA: T_arr, vecB: T_arr, nx: int, nt: int):
	"""Solve using scipy
	
	Let M be the size of the matrix,
	    nt be the number of timesteps,
	    nx be the number of spatial nodes, and
	    ndg be the number of delayed groups
	
	Paramters:
	----------
	matA: np.ndarray
		[M x M] square array of RHS
		
	vecB: np.ndarray
		[1 x M] vector of LHS
	
	nx: int
		Number of spatial nodes
	
	nt: int
		Number of timesteps
	
	Returns:
	--------
	P: np.ndarray
		[nx x nt] vector of powers
	
	C: np.ndarray
		[ndg x n] array of precursor group concentrations
	"""
	vecX = la.solve(matA, vecB)
	return __split_results(vecX, nx, nt)


def inversion(matA: T_arr, matB: T_arr, nx: int, nt: int):
	"""Solve by matrix inversion.
	
	Let M be the size of the matrix,
	    nt be the number of timesteps,
	    nx be the number of nodes, and
	    ndg be the number of delayed groups
	
	Paramters:
	----------
	matA: np.ndarray
		[M x M] square array of RHS
		
	vecB: np.ndarray
		[1 x M] vector of LHS
	
	nx: int
		Number of spatial nodes
	
	nt: int
		Number of timesteps
	
	Returns:
	--------
	P: np.ndarray
		[1 x ndg] vector of powers
	
	C: np.ndarray
		[ndg x n] array of precursor group concentrations
	"""
	invA = la.inv(matA, overwrite_a=False)
	vecX = invA.dot(matB)
	return __split_results(vecX, nx, nt)
