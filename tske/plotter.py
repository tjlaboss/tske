"""
Plotter

Plotting thingies
"""
import tske.keys as K
from matplotlib import rcParams
import matplotlib.pyplot as plt
from tske.tping import T_arr

# This will make the y-labels not be so stupid.
rcParams['axes.formatter.useoffset'] = False

COLOR_P = "forestgreen"
COLOR_R = "firebrick"


def plot_reactivity_and_power(
		times: T_arr,
		reacts: T_arr,
		powers: T_arr,
		plot_type=K.PLOT_LOG,
		power_units=None,
		title_text=""
):
	f"""Plot the reactor power and reactivity vs. time
	
	Parameters:
	-----------
	times: collection of float
		List of times (s)
		
	reacts: ndarray of float
		2D array (nx, nt) of reactivities ($)
	
	powers: ndarray of float
		2D array (nx, nt) of powers/fluxes (power_units).
	
	plot_type: str, optional
		Type of plot to make for power:
			{K.PLOT_LINEAR}: linear-linear
			{K.PLOT_SEMLOG}: log-linear (semilog-y)
			{K.PLOT_LOGLOG}: log-log
		[Default: {K.PLOT_LOGLOG}.]
	
	power_units: str, optional
		Units to show for the y-axis for power.
		[Default: None --> relative power]
	
	title_text: str, optional
		Title for the plot.
		[Default: None]
	"""
	n = len(times)
	assert (powers.shape == reacts.shape,
		f"The array of powers ({powers.shape}) must be "
	    f"the same shape as the array of reactivities ({reacts.shape}")
	nx, nt = powers.shape
	assert nt == n, \
		f"The number of times ({n}), powers ({nt}), and reactivities ({nt}) must be equal."
	if power_units is None:
		power_units = "Relative"
	
	# Plot power
	fig, pax = plt.subplots()
	plot_functions = {
		K.PLOT_LINEAR: pax.plot,
		K.PLOT_SEMLOG: pax.semilogy,
		K.PLOT_LOGLOG: pax.loglog
	}
	plot_f = plot_functions.get(plot_type, pax.loglog)
	plines = plot_f(times, powers, "-", color=COLOR_P, label=r"$P(t)$")
	pax.tick_params(axis="y", which="both", labelcolor=COLOR_P)
	pax.set_ylabel(f"Power ({power_units})", color=COLOR_P)
	
	# Plot reactivity
	rax = pax.twinx()
	rlines = rax.plot(times, reacts, "--", color=COLOR_R, label=r"$\rho(t)$")
	rax.tick_params(axis="y", labelcolor=COLOR_R)
	rax.set_ylabel("Reactivity (\$)", color=COLOR_R)
	
	lines = plines + rlines
	pax.legend(lines, [l.get_label() for l in lines], loc=0)
	pax.set_xlim([0, max(times)])
	pax.set_xlabel("Time (s)")
	
	# continue...
	if title_text:
		plt.suptitle(title_text)
	plt.tight_layout()


def plot_convergence(dts: T_arr, errors: T_arr, in_percent=False):
	"""Plot the convergence of a solution as a function of timestep size
	
	Parameters:
	-----------
	dts: collection of float
		List of 'dt' values.
	
	errors: collection of float
		List of the relative errors for each 'dt'.
	
	in_percent: bool, optional
		Whether the provided errors are in percent.
		[Default: False]
	"""
	lendts = len(dts)
	lenerr = len(errors)
	assert lendts == lenerr, \
		f"The number of dt ({lendts}) and errors ({lenerr}) must be equal."
	ax = plt.figure().add_subplot()
	ax.plot(dts, errors, 'rx')
	ax.set_xlabel(r"$\Delta t$ (s)")
	if in_percent:
		ax.set_ylabel(r"% Error in Power")
	else:
		ax.set_ylabel(r"Relative Error in Power")
	# Make sure the markers are visible of the plot
	xmin = min(dts)*0.9
	xmax = max(dts)*1.1
	ymin = min(min(errors)*1.1, -0.01)
	ymax = max(max(errors)*1.1, +0.01)
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])
	# highlight zero
	ax.plot([xmin, xmax], [0, 0], 'k-', lw=2)
	ax.grid()
	plt.tight_layout()


def plot_matrix(matA):
	"""Spy plot of the generated matrix
	
	Parameters:
	-----------
	matA: np.ndarray
		Square matrix, LHS of the equation, to plot.
	"""
	axA = plt.figure().add_subplot()
	axA.spy(matA)
	# axA.set_title(r"$\overline{\overline{A}}$")
	plt.tight_layout()
	return axA
