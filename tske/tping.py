"""
tping

custom typing
"""

import os as _os
import pathlib as _pl
import numpy as _np
import numpy.typing as _npt
import typing as _typing
import tske.node

T_arr = _npt.NDArray[_np.floating]
T_nodearr = _typing.Mapping[_typing.Tuple[int, int], tske.node.Node1D]
T_nodearr = _npt.NDArray[tske.node.Node1D]

PathType = (str, _os.PathLike, _pl.Path)
