"""Helper module containing type aliases."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

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

# MDAnalysis
Trajectory: TypeAlias = Union[  # type: ignore[no-any-unimported]
    "ProtoReader", "FrameIteratorSliced", "FrameIteratorIndices", "Timestep"
]
"""Trajectory-like object from MDAnalysis."""

MDAObject: TypeAlias = Union["Universe", "AtomGroup"]  # type: ignore[no-any-unimported]
"""An MDAnalysis Universe or Atomgroup."""

# Interaction parameters
Angles: TypeAlias = tuple[float, float]
