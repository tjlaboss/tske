"""
Plotter

Plotting thingies
"""
from tske.tping import T_arr
from matplotlib import rcParams
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


# This will make the y-labels not be so stupid.
# rcParams['axes.formatter.useoffset'] = False

COLOR_P = "forestgreen"
COLOR_R = "firebrick"


def plot_reactivity_and_power(
		times: T_arr,
		reacts: T_arr,
		powers: T_arr,
		title_text=""
):
	"""Plot the reactor power and reactivity vs. time
	
	Parameters:
	-----------
	times: collection of float
		List of times (s)
		
	reacts: ndarray of float
		2D array (nx, nt) of reactivities ($)
	
	powers: ndarray of float
		2D array (nx, nt) of powers/fluxes (power_units).
	
	title_text: str, optional
		Title for the plot.
		[Default: None]
	"""
	assert powers.shape == reacts.shape, \
		f"The array of powers ({powers.shape}) must be " \
	    f"the same shape as the array of reactivities ({reacts.shape}"
	nx, nt = powers.shape
	assert nt == len(times), \
		f"The number of times ({len(times)}), powers ({nt}), and reactivities ({nt}) must be equal."
	xvals = np.arange(nx)
	pzlims = (min(0.9, 0.9*powers.min()), max(1.1, 1.1*powers.max()))
	rmin = reacts.min(); rmax = reacts.max()
	rzlims = (min(1.1*rmin, 0.9*rmin), max(0.9*rmax, 1.1*rmax))
	# Plot power
	fig = plt.figure(1, figsize=[11,5])
	pax = fig.add_subplot(121, projection='3d')
	rax = fig.add_subplot(122, projection='3d')
	X, Y = np.meshgrid(xvals, times)
	P = powers.T
	pax.plot_surface(
		X, Y, P, edgecolor=COLOR_P, color=COLOR_P,
		alpha=0.3
	)
	pax.set(
		xlim=(0, nx-1),         xlabel="x (cm)",
		ylim=(0, max(times)),   ylabel="time (s)",
		zlim=pzlims,            zlabel="Power",
    )
	
	# Plot reactivity
	rax.plot_surface(
		X, Y, reacts.T, edgecolor=COLOR_R, color=COLOR_R,
		alpha=0.3
	)
	rax.set(
		xlim=(0, nx-1),         xlabel="x (cm)",
		ylim=(0, max(times)),   ylabel="time (s)",
		zlim=rzlims,            zlabel=r"Reactivity (\$)",
	)
	
	# Finish up.
	fig.tight_layout()
	if title_text:
		plt.suptitle(title_text)
	# plt.tight_layout()
	
	# Make 2D plots too
	fig2 = plt.figure(2, figsize=[11, 5])
	time_indices = [0]
	for t in np.ceil( np.logspace(0, np.log(nt-1), 5, base=np.e) ):
		print(t)
		time_indices.append(int(t))
	# times2 = times[time_indices]

	pax2 = fig2.add_subplot(121)
	rax2 = fig2.add_subplot(122)
	for t in time_indices:
		# lbl = fr"t = {times[t]*1e6:.0f} $\mu$s"
		lbl = fr"t = {times[t]*1e3:.0f} ms"
		pax2.plot(xvals, powers[:, t], "-",  label=lbl)
		pax2.set_ylabel(r"$\phi(x)$")
		pax2.set_ylim(pzlims)
		
		rax2.plot(xvals, reacts[:, t], "--", label=lbl)
		rax2.set_ylabel(r"$\rho(x)$ (\$)")
		rax2.set_ylim(rzlims)
		rax2.yaxis.set_label_position("right")
		rax2.yaxis.tick_right()
	for ax in (pax2, rax2):
		ax.legend(loc=0)
		ax.set_xlim([0, max(times)])
		ax.set_xlabel("$x$ (cm)")
	plt.tight_layout()


def plot_3d_power(
		times: T_arr,
		powers: T_arr,
):
	"""Plot the reactor power and reactivity vs. time
	
	Parameters:
	-----------
	times: collection of float
		List of times (s)
		
	powers: ndarray of float
		2D array (nx, nt) of powers/fluxes (power_units).
	
	"""
	nx, nt = powers.shape
	assert nt == len(times), \
		f"The number of times ({len(times)}), powers ({nt}), and reactivities ({nt}) must be equal."
	times *= 1e3
	
	# Plot power
	fig = plt.figure()
	pax = fig.add_subplot(111, projection='3d')
	X, Y = np.meshgrid(np.arange(nx), times)
	P = powers.T
	pax.plot_surface(
		X, Y, P, edgecolor=COLOR_P, color=COLOR_P,
		alpha=0.3
	)
	pzlims = (0.9*powers.min(), 1.1*powers.max())
	pax.set(
		xlim=(0, nx-1),         xlabel="x-node",
		ylim=(0, max(times)),   ylabel="time (ms)",
		zlim=pzlims,            zlabel="Power"
    )
	
	# Finish up.
	fig.tight_layout()


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
