"""Helper module containing type aliases."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, Union

if TYPE_CHECKING:
    from pathlib import Path

    from MDAnalysis.coordinates.base import (
        FrameIteratorIndices,
        FrameIteratorSliced,
        ProtoReader,
    )
    from MDAnalysis.coordinates.timestep import Timestep
    from MDAnalysis.core.groups import AtomGroup
    from MDAnalysis.core.universe import Universe

    from prolif.ifp import IFP
    from prolif.plotting.complex3d.base import Complex3DBaseBackendSettings
    from prolif.plotting.complex3d.py3dmol_backend import Py3DmolSettings
    from prolif.plotting.complex3d.pymol_backend import PyMOLSettings
    from prolif.residue import ResidueId

# utils
PathLike: TypeAlias = Union[str, "Path"]

# residues
ResidueKey: TypeAlias = Union["ResidueId", int, str]
ResidueSelection: TypeAlias = Literal["all"] | Sequence[ResidueKey] | None

# IFP
IFPResults: TypeAlias = dict[int, "IFP"]
"""A dictionary holding the interaction fingerprint data for each frame/pose."""

InteractionMetadata: TypeAlias = dict[str, Any]
IFPData: TypeAlias = dict[str, Sequence[InteractionMetadata]]
Component: TypeAlias = Literal["ligand", "protein", "water"]

# MDAnalysis
Trajectory: TypeAlias = Union[  # type: ignore[no-any-unimported]
    "ProtoReader", "FrameIteratorSliced", "FrameIteratorIndices", "Timestep"
]
"""Trajectory-like object from MDAnalysis."""

MDAObject: TypeAlias = Union["Universe", "AtomGroup"]  # type: ignore[no-any-unimported]
"""An MDAnalysis Universe or Atomgroup."""

# Interaction parameters
Angles: TypeAlias = tuple[float, float]


# complex3D
class PyMOLRPCServer(Protocol):
    def ping(self) -> None: ...
    def do(self, command: str) -> None: ...
    def __close(self) -> None: ...


Complex3DBackendSettings: TypeAlias = (
    "Literal['py3dmol', 'pymol']"
    " | Py3DmolSettings"
    " | PyMOLSettings"
    " | str"
    " | Complex3DBaseBackendSettings"
)
