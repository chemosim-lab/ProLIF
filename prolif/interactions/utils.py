"""
Utility functions for interactions --- :mod:`prolif.interactions.utils`
=======================================================================

This module contains some utilities used by the interaction classes.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from rdkit.Geometry import Point3D

    from prolif.residue import Residue

    DistFunc3Args: TypeAlias = Callable[[Point3D, Point3D, Point3D], float]
    DistFunc4Args: TypeAlias = Callable[[Point3D, Point3D, Point3D, Point3D], float]


def get_mapindex(res: "Residue", index: int) -> int:
    """Get the index of the atom in the original molecule

    Parameters
    ----------
    res : prolif.residue.Residue
        The residue in the protein or ligand
    index : int
        The index of the atom in the :class:`~prolif.residue.Residue`

    Returns
    -------
    mapindex : int
        The index of the atom in the :class:`~prolif.molecule.Molecule`
    """
    return res.GetAtomWithIdx(index).GetUnsignedProp("mapindex")


def _distance_3args_l1_p1(
    l1: "Point3D",
    p1: "Point3D",
    p2: "Point3D",  # noqa: ARG001
) -> float:
    return l1.Distance(p1)


def _distance_3args_l1_p2(
    l1: "Point3D",
    p1: "Point3D",  # noqa: ARG001
    p2: "Point3D",
) -> float:
    return l1.Distance(p2)


def _distance_4args_l1_p1(
    l1: "Point3D",
    l2: "Point3D",  # noqa: ARG001
    p1: "Point3D",
    p2: "Point3D",  # noqa: ARG001
) -> float:
    return l1.Distance(p1)


def _distance_4args_l1_p2(
    l1: "Point3D",
    l2: "Point3D",  # noqa: ARG001
    p1: "Point3D",  # noqa: ARG001
    p2: "Point3D",
) -> float:
    return l1.Distance(p2)


def _distance_4args_l2_p1(
    l1: "Point3D",  # noqa: ARG001
    l2: "Point3D",
    p1: "Point3D",
    p2: "Point3D",  # noqa: ARG001
) -> float:
    return l2.Distance(p1)


def _distance_4args_l2_p2(
    l1: "Point3D",  # noqa: ARG001
    l2: "Point3D",
    p1: "Point3D",  # noqa: ARG001
    p2: "Point3D",
) -> float:
    return l2.Distance(p2)


DISTANCE_FUNCTIONS_3ARGS: dict[str, "DistFunc3Args"] = {
    "P1": _distance_3args_l1_p1,
    "P2": _distance_3args_l1_p2,
}
DISTANCE_FUNCTIONS_4ARGS: dict[
    tuple[Literal["L1", "L2"], Literal["P1", "P2"]], "DistFunc4Args"
] = {
    ("L1", "P1"): _distance_4args_l1_p1,
    ("L1", "P2"): _distance_4args_l1_p2,
    ("L2", "P1"): _distance_4args_l2_p1,
    ("L2", "P2"): _distance_4args_l2_p2,
}
