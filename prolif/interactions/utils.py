"""
Utility functions for interactions --- :mod:`prolif.interactions.utils`
=======================================================================

This module contains some utilities used by the interaction classes.
"""


def get_mapindex(res, index):
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


def _distance_3args_l1_p1(l1, p1, p2):
    return l1.Distance(p1)


def _distance_3args_l1_p2(l1, p1, p2):
    return l1.Distance(p2)


def _distance_4args_l1_p1(l1, l2, p1, p2):
    return l1.Distance(p1)


def _distance_4args_l1_p2(l1, l2, p1, p2):
    return l1.Distance(p2)


def _distance_4args_l2_p1(l1, l2, p1, p2):
    return l2.Distance(p1)


def _distance_4args_l2_p2(l1, l2, p1, p2):
    return l2.Distance(p2)


DISTANCE_FUNCTIONS = {
    "P1": _distance_3args_l1_p1,
    "P2": _distance_3args_l1_p2,
    ("L1", "P1"): _distance_4args_l1_p1,
    ("L1", "P2"): _distance_4args_l1_p2,
    ("L2", "P1"): _distance_4args_l2_p1,
    ("L2", "P2"): _distance_4args_l2_p2,
}
