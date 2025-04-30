"""
Plotter

Plotting thingies
"""
import tske.keys as K
from tske.tping import T_arr
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import os


COLOR_P = "forestgreen"
COLOR_R = "firebrick"
FS = (5.0, 4.0)

def plot_reactivity_and_power(
		times: T_arr,
		reacts: T_arr,
		powers: T_arr,
		dx: float,
		output_dir: str,
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
	
	dx: float
		Delta x (cm)
	
	output_dir: str
		Output directory to save plots to
	
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
	xvals = np.arange(nx)*dx
	xlims = (xvals[0], xvals[-1])
	ylims = (times[0], times[-1])
	pzlims = (min(0.9, 0.9*powers.min()), max(1.1, 1.1*powers.max()))
	rmin = reacts.min(); rmax = reacts.max()
	rzlims = (min(1.1*rmin, 0.9*rmin), max(0.9*rmax, 1.1*rmax))
	# Plot power
	fig1 = plt.figure(1, figsize=FS)
	fig2 = plt.figure(2, figsize=FS)
	pax = fig1.add_subplot(projection='3d')
	rax = fig2.add_subplot(projection='3d')
	X, Y = np.meshgrid(xvals, times)
	P = powers.T
	pax.plot_surface(
		X, Y, P,
		edgecolor=COLOR_P,
		#color=COLOR_P,
		alpha=0.3,
		cmap=cm.coolwarm,
	)
	pax.set(
		xlim=xlims,  xlabel="x (cm)",
		ylim=ylims,  ylabel="time (s)",
		zlim=pzlims, zlabel=r"$\phi(x,t)$",
	)
	
	# Plot reactivity
	rax.plot_surface(
		X, Y, reacts.T,
		edgecolor="gray",
		#edgecolor=COLOR_R,
		alpha=0.3,
		cmap=cm.bwr,
	)
	rax.set(
		xlim=xlims,  xlabel="x (cm)",
		ylim=ylims,  ylabel="time (s)",
		zlim=rzlims, zlabel=r"$\rho(x,t)$ (\$)",
	)
	
	# Finish up.
	fig1.tight_layout()
	fig1.savefig(os.path.join(output_dir, K.FNAME_FLUX3))
	fig2.tight_layout()
	fig2.savefig(os.path.join(output_dir, K.FNAME_REACT3))
	
	# Make 2D plots too
	# fig2 = plt.figure(2, figsize=[11, 5])
	fig3 = plt.figure(3, figsize=FS)
	fig4 = plt.figure(4, figsize=FS)
	time_indices = {0}
	time_indices |= set(np.argmax(powers, axis=1))
	for t in np.ceil( np.logspace(0, np.log(nt-1), 5, base=np.e) ):
		time_indices.add(int(t))
	time_indices = list(sorted(time_indices))

	pax2 = fig3.add_subplot()
	rax2 = fig4.add_subplot()
	for t in time_indices:
		lbl = fr"t = {times[t]*1e3:.1f} ms"
		pax2.plot(xvals, powers[:, t], "x-",  label=lbl)
		pax2.set_ylabel(r"$\phi(x)$")
		pax2.set_ylim(pzlims)
		pax2.grid(which='both')
		pax2.yaxis.set_ticks(np.arange(0, powers.max()+5, 5))
		
		rax2.plot(xvals, reacts[:, t], "o--", label=lbl)
		rax2.set_ylabel(r"$\rho(x)$ (\$)")
		rax2.set_ylim(rzlims)
		rax2.yaxis.set_label_position("right")
		rax2.yaxis.tick_right()
		rax2.grid()
	for ax in (pax2, rax2):
		ax.legend(loc=0)
		ax.set_xlim(xlims)
		ax.set_xlabel("$x$ (cm)")
	fig3.tight_layout()
	fig3.savefig(os.path.join(output_dir, K.FNAME_FLUX2))
	fig4.tight_layout()
	fig3.savefig(os.path.join(output_dir, K.FNAME_REACT2))


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
