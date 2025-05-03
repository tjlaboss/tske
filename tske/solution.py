"""
Solution

Module for the solution mode and its helpers
"""

import os
import copy
import time
import typing
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tske
import tske.keys as K


def _get_node(input_dict: typing.Mapping) -> tske.Node1D:
	gdict = input_dict[K.GEOM]
	m_id = gdict[K.NODE_MATERIAL]
	mat = tske.Material.from_dict(input_dict[K.DATA][K.MATERIALS][m_id])
	delta_x = gdict[K.GEOM_TOTAL] / gdict[K.GEOM_NX]
	node = tske.Node1D(ngroups=1, fill=mat, dx=delta_x)
	return node


def _build_node_array(nodelist, materials, times, dx):
	nx = len(nodelist)
	nt = len(times)
	nodearr = np.empty((nx, nt), dtype=tske.Node1D)
	for i, ndict in enumerate(nodelist):
		imat = materials[ndict[K.NODE_MATERIAL]]
		node = tske.Node1D(1, imat, dx)
		swaps = ndict.get(K.NODE_SWAPS)
		nodearr[i, :] = node
		if not swaps:
			continue
		for swaptime, swapmat in sorted(swaps.items()):
			newnode = copy.copy(node)  # or deepcopy()?
			newnode.fill = materials[swapmat]
			for n, t in enumerate(times):
				if t > swaptime:
					nodearr[i, n:] = newnode
					break
	return nodearr


def _get_reactivities(node_arr, beta_arr, lambda_arr, bcs, nx, dx):
	if bcs[0] == bcs[1] == tske.analytic.BoundaryConditions.reflective:
		buck = 0
	elif bcs[0] == bcs[1] == tske.analytic.BoundaryConditions.absorptive:
		xx = (nx+2)*dx
		buck = (np.pi/xx)**2
	else:
		xx = 2*(nx+1)*dx
		buck = (np.pi/xx)**2
	
	bl = (beta_arr*lambda_arr).sum()
	rhof = np.vectorize(lambda node: node.get_rho(bg2=buck, lambda_beta=bl))
	reactivities =  rhof(node_arr) / beta_arr.sum()
	# force reactivities to start at 0
	reactivities -= np.tile(reactivities[: ,0], (reactivities.shape[1], 1)).T
	return reactivities


def solution(input_dict: typing.Mapping, output_dir: tske.tping.PathType):
	"""Solve the Spatial Kinetics Reactor Equations

	Numerically solve the SKRE, write the data to the output directory,
	make the indicated plots, and save plots to the output directory.

	Parameters:
	-----------
	input_dict: dict
		Dictionary of the the parsed input file.

	output_dir: str or PathLike
		Output folder to write results to.
		If it does not exist, it will be created.

	"""
	plots = input_dict.get(K.PLOT, {})
	method = tske.matrices.METHODS[input_dict[K.METH]]
	total = input_dict[K.TIME][K.TIME_TOTAL]
	dt = input_dict[K.TIME][K.TIME_DELTA]
	num_steps = int(np.ceil(total/dt))  # Will raise total if not divisible
	times = np.linspace(0, num_steps*dt, num_steps)
	np.savetxt(os.path.join(output_dir, K.FNAME_TIME), times)
	dx = input_dict[K.GEOM][K.GEOM_DX]
	nodelist = input_dict[K.GEOM][K.NODES]
	nx = len(nodelist)
	materials = [tske.Material.from_dict(m) for m in input_dict[K.DATA][K.MATERIALS]]
	nodes = _build_node_array(nodelist, materials, times, dx)
	bcs = []
	for edge_node in [0, -1]:
		bc = input_dict[K.GEOM][K.NODES][edge_node][K.NODE_BC]
		bcs.append(getattr(tske.analytic.BoundaryConditions, bc))
	beta_arr = np.array(input_dict[K.DATA][K.DATA_B])
	lambda_arr = np.array(input_dict[K.DATA][K.DATA_L])
	reactivities = _get_reactivities(nodes, beta_arr, lambda_arr, bcs, nx, dx)
	np.savetxt(os.path.join(output_dir, K.FNAME_RHO), reactivities.T)
	matA, matB = tske.matrices.crank_nicolson(
		method=method,
		dt=dt,
		dx=dx,
		betas=beta_arr,
		lams=lambda_arr,
		L=input_dict[K.DATA][K.DATA_BIG_L],
		v1=input_dict[K.DATA][K.DATA_IV],
		nodes=nodes,
		bcs=bcs,
		P0=None
	)
	np.savetxt(os.path.join(output_dir, K.FNAME_MATRIX_A), matA)
	np.savetxt(os.path.join(output_dir, K.FNAME_MATRIX_B), matB)
	to_show = plots.get(K.PLOT_SHOW, 0)
	if plots.get(K.PLOT_SPY):
		tske.plotter.plot_matrix(matA)
		plt.savefig(os.path.join(output_dir, K.FNAME_SPY))
		if to_show > 1:
			plt.show()
	tick = time.time()
	print("Solving...")
	power_vals, concentration_val_list = tske.solver.linalg(matA, matB, nx, num_steps)
	tock = time.time()
	print(f"...Completed in {tock - tick:.2f} seconds. Outputs saved to: {output_dir}.")
	np.savetxt(os.path.join(output_dir, K.FNAME_P), power_vals.T)
	for k, concentration_vals in enumerate(concentration_val_list):
		fpath = os.path.join(output_dir, K.FNAME_FMT_C.format( k +1))
		np.savetxt(fpath, concentration_vals.T)
	prplot = plots.get(K.PLOT_PR)
	if prplot == 1:
		tske.plotter.plot_reactivity_and_power(
			times=times,
			powers=power_vals,
			reacts=reactivities,
			dx=dx,
			output_dir=output_dir
		)
	elif prplot == 2:
		# Plot them separately
		warnings.warn("Not implemented yet: separate power and reactivity plots", FutureWarning)
	if to_show > 1:
		plt.show()
	
	# keep at end
	if to_show:
		plt.show()
