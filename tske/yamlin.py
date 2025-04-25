"""
YAML in

YAML reading and validation.
"""

import typing
import yamale
import numpy as np
from tske.matrices import METHODS
from tske.tping import PathType
from tske.keys import *

try:
	from ruamel import yaml
	PARSER = "ruamel"
except ModuleNotFoundError:
	import yaml
	PARSER = "PyYaml"


def _enum(iterable: typing.Iterable, **kwargs) -> str:
	"""Turn an interable into a yamale-compatible enum string"""
	string = 'enum(' + ','.join([repr(s) for s in iterable])
	for k, v in kwargs.items():
		string += f', {k}={v}'
	string += ')'
	return string


SCHEMA = f"""\
{TIME}: include('time_type')
{DATA}: include('data_type')
{GEOM}: include('geometry_type')
{PLOT}: include('plot_type', required=False)
{METH}: {_enum(METHODS.keys(), ignore_case=True)}
---
time_type:
  {TIME_TOTAL}: num(min=0)
  {TIME_DELTA}: num(min=0)
---
data_type:
  {DATA_B}: list(num(min=0))
  {DATA_L}: list(num(min=0))
  {DATA_BIG_L}: num(min=0)
  {DATA_IV}: num(min=0)
  {MATERIALS}: list(include('material_type'))
---
material_type:
  {XS_SIGMA_A}: num(min=0)
  {XS_NUSIGMA_FP}: num(min=0)
  {XS_D}: num(min=0)
---
geometry_type:
  {GEOM_DX}: num(min=0)
  {NODES}: list('node_type')
---
node_type:
  {NODE_MATERIAL}: int(min=0)
  {NODE_SWAPS}: map(int(), key=num(min=0))
  {NODE_BC}: {_enum(BC_TYPES, ignore_case=True, required=False)}
---
plot_type:
  {PLOT_SHOW}: int(min=0, max=2, required=False)
  {PLOT_SPY}: int(min=0, max=1, required=False)
  {PLOT_PR}: int(min=0, max=2, required=False)
  {PLOT_LOG}: {_enum(PLOT_TYPES, ignore_case=True, required=False)}
"""

yamale_schema = yamale.make_schema(content=SCHEMA, parser=PARSER)


def load_input_file(fpath: PathType) -> typing.MutableMapping:
	"""Load and check a YAML input file using the best available data.
	
	This function also does some type enforcement.
	This isn't where I want to do that. Move eventually...
	
	Parameters:
	-----------
	fpath: str or PathLike
		Path to the input YAML file to read
	
	Returns:
	--------
	ydict: dict
		Dictionary of the input parameters.
	"""
	data = yamale.make_data(fpath, parser=PARSER)
	yamale.validate(yamale_schema, data)
	ydict = data[0][0]
	check_input(ydict)
	# Let's make these arrays for later.
	ydict[DATA][DATA_B] = np.array(ydict[DATA][DATA_B])*1e-5
	ydict[DATA][DATA_L] = np.array(ydict[DATA][DATA_L])
	return ydict


def check_input(config: typing.Mapping):
	"""Check the input dictionary and raise an error if appropriate"""
	errs = []
	if len(config[DATA][DATA_B]) != len(config[DATA][DATA_L]):
		errs.append("Number of delayed fractions does not match number of decay constants.")
	if config[TIME][TIME_TOTAL] < config[TIME][TIME_DELTA]:
		errs.append("Total time is less than timestep size.")
	max_mat = len(config[DATA][MATERIALS]) - 1
	num_nodes = len(config[GEOM][NODES])
	if num_nodes < 3:
		errs.append("The minimum number of spatial nodes is 3.")
	boundary_nodes = (0, len(config[GEOM][NODES]) - 1)
	found_mats = set()
	for i, node in enumerate(config[GEOM][NODES]):
		found_mats.add(node[NODE_MATERIAL])
		if NODE_SWAPS in node:
			found_mats |= set(node[NODE_SWAPS].values())
		if i in boundary_nodes:
			if not node.get(NODE_BC):
				errs.append("First and last nodes must have boundary conditions.")
		else:
			if node.get(NODE_BC):
				errs.append("Interior nodes must not have boundary conditions.")
			node[NODE_BC] = None
	if max(found_mats) > max_mat:
		errs.append(f"Invalid material number: maximum is {max_mat}")
	if errs:
		errstr = f"There were {len(errs)} errors:\n\t"
		errstr += "\n\t".join(errs)
		raise ValueError(errstr)
		

def _ruamel_load_input_file(stream: typing.TextIO) -> typing.Mapping:
	y = yaml.YAML(typ="safe")
	return y.load(stream)


def _pyyaml_load_input_file(stream: typing.TextIO) -> typing.Mapping:
	return yaml.safe_load(stream)


def _ruamel_dump_input_file(data: typing.Mapping, stream: typing.TextIO):
	y = yaml.YAML(typ="safe")
	return y.dump(data, stream)


def _pyyaml_dump_input_file(data: typing.Mapping, stream: typing.TextIO):
	return yaml.safe_dump(data, stream)


def dump(fpath: PathType, data: typing.Mapping):
	"""Dump a dictionary to a file.
	
	Parameters:
	-----------
	fpath: str or PathLike
		File to dump the YAML to.
	
	data: dict
		Dictionary of data to dump.
	"""
	with open(fpath, 'w') as fy:
		if PARSER == "ruamel":
			return _ruamel_dump_input_file(data, fy)
		else:
			return _pyyaml_dump_input_file(data, fy)

