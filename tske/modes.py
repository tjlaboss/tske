"""
Modes

Run modes for TSKE
"""
import os
import sys
import typing
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tske
import tske.keys as K
from tske.solution import solution


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
	# # Spy plot of Matrix A
	# afpath = os.path.join(output_dir, K.FNAME_MATRIX_A)
	# if not os.path.exists(afpath):
	# 	errs.append(f"Matrix A could not be found at: {afpath}")
	# else:
	# 	try:
	# 		matA = np.loadtxt(afpath)
	# 		tske.plotter.plot_matrix(matA)
	# 	except Exception as e:
	# 		errs.append(f"Failed to plot Matrix A: {type(e)}: {e}")
	# 	else:
	# 		plt.savefig(K.FNAME_SPY)
	# 		print("Matrix A spy plot saved to:", K.FNAME_SPY)
	# Power-Reactivity plot
	tfpath = os.path.join(output_dir, K.FNAME_TIME)
	pfpath = os.path.join(output_dir, K.FNAME_P)
	rfpath = os.path.join(output_dir, K.FNAME_RHO)
	cfgyml = os.path.join(output_dir, K.FNAME_CFG)
	if not os.path.exists(tfpath):
		errs.append(f"Times could not be found at: {tfpath}")
	elif not os.path.exists(pfpath):
		errs.append(f"Reactor powers could not be found at: {pfpath}")
	else:
		try:
			times = np.loadtxt(tfpath)
			powers = np.loadtxt(pfpath).T
			reactivities = np.loadtxt(rfpath).T
			config = tske.yamlin.load_input_file(cfgyml)
			dx = config[K.GEOM][K.GEOM_DX]
			tske.plotter.plot_reactivity_and_power(
				times, reactivities, powers, dx, output_dir
			)
		except Exception as e:
			errs.append(f"Failed to plot power and reactivity: {type(e)}: {e}")
		else:
			print("Power and reactivity plot saved to:", K.FNAME_PR)
	le = len(errs)
	if le:
		errstr = f"There were {le} errors:\n\t"
		errstr += "\n\t".join(errs)
		print(errs, file=sys.stderr)
	plt.show()
	return le




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
	header = "Timestep Study:"
	report = header + "\n" + '='*len(header)
	for i, dt in enumerate(dts):
		cfg = dict(input_dict)
		cfg[K.TIME][K.TIME_DELTA] = dt
		out_fpath = os.path.join(output_dir, str(i))
		os.makedirs(out_fpath, exist_ok=True)
		with open(os.path.join(out_fpath, K.FNAME_DT), 'w') as f:
			f.write(str(dt))
		solution(cfg, out_fpath)
		power = _load_solution(out_fpath)
		report += f"\n\tP(dt={dt:.2e} s): {power:.4f}"
		# Calculate the relative error vs. the reference solution.
		if i == 0:
			ref = power
			error = 0
		else:
			error = (power - ref)/ref
			report += f" | Error: {error:+8.4%}"
		# print(report)
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
		endpow = powers[-1, :].max()
		return float(endpow)
	except Exception as e:
		warnings.warn(f"Failed to load {fpath}: {e}", Warning)
	return np.nan
