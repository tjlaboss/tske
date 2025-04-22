"""
Material

Module for multigroup diffusion material
"""

import numpy as np


class Material:
	"""A one-group material for neutron diffusion
	
	Parameters:
	-----------
	ngroups: int
		number of energy groups to use
	
	name: str, optional
		Name of the material
	
	Attributes:
	-----------
	d: np.ndarray(shape=[ngroups, 1], dtype=float), cm
		Diffusion coefficient
	
	sigma_a: np.ndarray(shape=[ngroups, 1], dtype=float), cm^-1
		Macroscopic absorption xs
	
	nu_sigma_f: np.ndarray(shape=[ngroups, 1], dtype=float), cm^-1
		Macroscopic nu-fission xs
	"""
	def __init__(self, name=""):
		self.ngroups = 1
		self.name = name
		self._d = 0
		self._sigma_a = 0
		self._nu_sigma_f = 0
	
	@classmethod
	def from_dict(cls, mdict):
		m = cls()
		m.sigma_a = mdict["sigma_a"]
		m.nu_sigma_f = mdict["prompt_nu_sigma_f"]
		m.d = mdict["diffusion_coefficient"]
		return m
	
	@property
	def d(self):
		return self._d

	@property
	def sigma_a(self):
		return self._sigma_a

	@property
	def nu_sigma_f(self):
		return self._nu_sigma_f

	@d.setter
	def d(self, d):
		self._d = d

	@sigma_a.setter
	def sigma_a(self, sigma_a):
		self._sigma_a = sigma_a

	@nu_sigma_f.setter
	def nu_sigma_f(self, nu_sigma_f):
		self._nu_sigma_f = nu_sigma_f
	
	def check_cross_sections(self):
		assert self.d, "Diffusion coefficient must be set."
		assert self.sigma_a, "Absorption XS must be set."

	def get_kinf(self):
		"""Wrapper for get_keff() for an infinite medium"""
		return self.get_keff(bg2=0.0)

	def get_keff(self, bg2):
		"""Find the eigenvalue in a homogeneous material

		Parameter:
		----------
		bg2: float; optional
			Geometric buckling

		Returns:
		--------
		float
			The multiplication factor/eigenvalue
		"""
		return np.float64(self.nu_sigma_f/(self.sigma_a + self.d*bg2))
