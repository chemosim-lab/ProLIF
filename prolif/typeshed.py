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

PathLike: TypeAlias = Union[str, "Path"]
ResidueKey: TypeAlias = Union["ResidueId", int, str]
ResidueSelection: TypeAlias = Literal["all"] | Sequence[ResidueKey] | None
IFPResults: TypeAlias = dict[int, "IFP"]
InteractionMetadata: TypeAlias = dict[str, Any]
IFPData: TypeAlias = dict[str, tuple[InteractionMetadata, ...]]
Trajectory: TypeAlias = Union[  # type: ignore[no-any-unimported]
    "ProtoReader", "FrameIteratorSliced", "FrameIteratorIndices", "Timestep"
]
MDAObject: TypeAlias = Union["Universe", "AtomGroup"]  # type: ignore[no-any-unimported]
