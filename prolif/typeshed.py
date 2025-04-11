from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

if TYPE_CHECKING:
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

ResidueKey: TypeAlias = Union["ResidueId", int, str]
ResidueSelection: TypeAlias = Literal["all"] | Sequence[ResidueKey] | None
IFPResults: TypeAlias = dict[int, "IFP"]
Trajectory: TypeAlias = Union[  # type: ignore[no-any-unimported]
    "ProtoReader", "FrameIteratorSliced", "FrameIteratorIndices", "Timestep"
]
MDAObject: TypeAlias = Union["Universe", "AtomGroup"]  # type: ignore[no-any-unimported]
