"""
Cranky Reactivity

Testing script for developing Crank-Nicolson solver for reactivity
"""

import tske
from tske.matrices import crank_nicolson

import numpy as np
np.set_printoptions(linewidth=240)

NT = 40
NX = 4

LENGTH = 10  # cm
DX=LENGTH/NX

V1 = 2e-6

EX_STAB_TIME = V1 * DX**2 / 2
print(f"Stability criterion for explicit: {EX_STAB_TIME:.3e} s")

mat1 = tske.Material("test")
mat1.d = 1.0
mat1.sigma_a = 0.10
# mat1.nu_sigma_f = 0.1986821352878128
mat1.nu_sigma_f = mat1.sigma_a
tnode = tske.node.Node1D(1, mat1, dx=LENGTH/NX)


rho_array = np.full(
	shape=(NX, NT),
	fill_value=5e-5 / 0.007,
	# fill_value=0.05,
)

bcs = 2*[tske.analytic.BoundaryConditions.reflective]

a, b = crank_nicolson(
	dt=5e-2,
	betas=np.array([700e-5]),
	lams=np.array([0.1]),
	rhos=rho_array,
	# L=2e-5,
	L=V1/mat1.nu_sigma_f,
	node=tnode,
	v1=V1,
	bcs=bcs,
	method=2
)


# from pylab import *; spy(a); show()
from scipy.linalg import solve, inv
# x = inv(a).dot(b)
x = solve(a,b)
# x = solve(a[:NX*NT,:NX*NT], b[:NX*NT])

phi = x[:NX*NT].reshape(NT, NX) #phi = x[:NX*NT].reshape(NX, NT).T
print(phi)
print()

ccc = x[NX*NT:].reshape(NT,NX) #ccc = x[NX*NT:].reshape(NX, NT).T
print(ccc)

