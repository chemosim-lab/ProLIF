"""Helper module containing type aliases."""

from typing import TypeAlias, Union

from MDAnalysis.coordinates.base import FrameIteratorSliced, ProtoReader

Trajectory: TypeAlias = Union[FrameIteratorSliced, ProtoReader]
