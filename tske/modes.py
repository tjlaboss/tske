"""
Modes

Run modes for TSKE
"""
import os
import sys
import copy
import typing
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tske
import tske.keys as K


def plot_only(output_dir: tske.tping.PathType):
	"""Only plot the existing results
	
	Parameters:
	-----------
	output_dir: str or PathLike
		Output folder to read existing results from.
	
	Returns:
	--------
	le: int
		Error status.
		le == 0 is OK, le != 0 indicates errors.
	"""
	errs = []
	# Spy plot of Matrix A
	afpath = os.path.join(output_dir, K.FNAME_MATRIX_A)
	if not os.path.exists(afpath):
		errs.append(f"Matrix A could not be found at: {afpath}")
	else:
		try:
			matA = np.loadtxt(afpath)
			tske.plotter.plot_matrix(matA)
		except Exception as e:
			errs.append(f"Failed to plot Matrix A: {type(e)}: {e}")
		else:
			plt.savefig(K.FNAME_SPY)
			print("Matrix A spy plot saved to:", K.FNAME_SPY)
	# Power-Reactivity plot
	tfpath = os.path.join(output_dir, K.FNAME_TIME)
	rfpath = os.path.join(output_dir, K.FNAME_RHO)
	pfpath = os.path.join(output_dir, K.FNAME_P)
	if not os.path.exists(tfpath):
		errs.append(f"Times could not be found at: {tfpath}")
	elif not os.path.exists(rfpath):
		errs.append(f"Reactivities could not be found at: {rfpath}")
	elif not os.path.exists(pfpath):
		errs.append(f"Reactor powers could not be found at: {pfpath}")
	else:
		try:
			times = np.loadtxt(tfpath)
			reactivities = np.loadtxt(rfpath).T
			powers = np.loadtxt(pfpath).T
			# if len(times) != len(reactivities) != len(powers)  -> handled in plotting
			tske.plotter.plot_reactivity_and_power(times, reactivities, powers)
		except Exception as e:
			errs.append(f"Failed to plot power and reactivity: {type(e)}: {e}")
		else:
			plt.savefig(K.FNAME_FLUX)
			print("Power and reactivity plot saved to:", K.FNAME_PR)
	le = len(errs)
	if le:
		errstr = f"There were {le} errors:\n\t"
		errstr += "\n\t".join(errs)
		print(errs, file=sys.stderr)
	plt.show()
	return le


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
	matA, matB = tske.matrices.crank_nicolson(
		method=method,
		dt=dt,
		dx=dx,
		betas=input_dict[K.DATA][K.DATA_B],
		lams=input_dict[K.DATA][K.DATA_L],
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
	power_vals, concentration_val_list = tske.solver.linalg(matA, matB, nx, num_steps)
	np.savetxt(os.path.join(output_dir, K.FNAME_P), power_vals.T)
	for k, concentration_vals in enumerate(concentration_val_list):
		fpath = os.path.join(output_dir, K.FNAME_FMT_C.format(k+1))
		np.savetxt(fpath, concentration_vals.T)
	prplot = plots.get(K.PLOT_PR)
	if prplot == 1:
		tske.plotter.plot_3d_power(
			times=times,
			powers=power_vals,
		)
		plt.savefig(os.path.join(output_dir, K.FNAME_PR))
	elif prplot == 2:
		# Plot them separately
		warnings.warn("Not implemented yet: separate power and reactivity plots", FutureWarning)
	if to_show > 1:
		plt.show()
	
	# keep at end
	if to_show:
		plt.show()


def study_timesteps(
		input_dict: typing.Mapping,
		output_dir: tske.tping.PathType,
		dts: typing.Iterable[float]
):
	"""Study the effect of timestep size upon final power.
	
	Parameters:
	-----------
	input_dict: dict
		Dictionary of the the parsed input file.
	
	output_dir: str or PathLike
		Output folder to write results to.
		If it does not exist, it will be created.
		Each result will create a subfolder in 'output_dir'.
	
	dts: iterable of float
		List of timestep sizes (s).
	"""
	dts = sorted(dts)
	errors = []
	ref = np.nan
	for i, dt in enumerate(dts):
		cfg = dict(input_dict)
		cfg[K.TIME][K.TIME_DELTA] = dt
		out_fpath = os.path.join(output_dir, str(i))
		os.makedirs(out_fpath, exist_ok=True)
		with open(os.path.join(out_fpath, K.FNAME_DT), 'w') as f:
			f.write(str(dt))
		solution(cfg, out_fpath)
		power = _load_solution(out_fpath)
		report = f"\tP(dt={dt:.2e} s): {power:.4f}"
		# Calculate the relative error vs. the reference solution.
		if i == 0:
			ref = power
			error = 0
		else:
			error = (power - ref)/ref
			report += f" | Error: {error:+8.4%}"
		print(report)
		errors.append(error)
	with open(os.path.join(output_dir, K.FNAME_REPORT), 'w') as f:
		f.write(report)
	plot_dts = np.array(dts)[1:]
	plot_err = np.array(errors)[1:]*100
	tske.plotter.plot_convergence(plot_dts, plot_err, in_percent=True)
	fpath_plot = os.path.join(output_dir, K.FNAME_CONVERGE)
	plt.savefig(fpath_plot)
	print("Results plotted to:", fpath_plot)
	plt.show()


def _load_solution(study_dir: tske.tping.PathType) -> float:
	"""Load the last power from a transient."""
	fpath = os.path.join(study_dir, K.FNAME_P)
	try:
		powers = np.loadtxt(fpath)
		endpow = powers.flatten()[-1]
		return float(endpow)
	except Exception as e:
		warnings.warn(f"Failed to load {fpath}: {e}", Warning)
	return np.nan
