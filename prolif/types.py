from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, TypeGuard, Union

Indices: TypeAlias = tuple[int, ...]
Angles: TypeAlias = tuple[float, float]


class Specs:
    pass


@dataclass
class Pattern(Specs):
    ligand: Union[bool, str] = False
    protein: Union[bool, str] = False


@dataclass
class Geometry(Specs):
    type: Literal["distance", "angles"]
    attr: Optional[str] = None


def is_pattern(spec: Specs) -> TypeGuard[Pattern]:
    return isinstance(spec, Pattern)


def is_geometry(spec: Specs) -> TypeGuard[Geometry]:
    return isinstance(spec, Geometry)
