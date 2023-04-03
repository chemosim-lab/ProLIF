"""
Helper functions --- :mod:`prolif.utils`
========================================
"""
import warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from importlib.util import find_spec
from math import pi

import numpy as np
import pandas as pd
from rdkit import rdBase
from rdkit.Chem import FragmentOnBonds, GetMolFrags, SplitMolByPDBResidues
from rdkit.DataStructs import ExplicitBitVect
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

from .residue import ResidueId

_90_deg_to_rad = pi / 2


def requires(module):  # pragma: no cover
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if find_spec(module):
                return func(*args, **kwargs)
            raise ModuleNotFoundError(
                f"The module {module!r} is required to use {func.__name__!r} "
                "but it is not installed!"
            )

        return wrapper

    return inner


@contextmanager
def catch_rdkit_logs():
    log_status = rdBase.LogStatus()
    rdBase.DisableLog("rdApp.*")
    yield
    log_status = {st.split(":")[0]: st.split(":")[1] for st in log_status.split("\n")}
    log_status = {k: True if v == "enabled" else False for k, v in log_status.items()}
    for k, v in log_status.items():
        if v is True:
            rdBase.EnableLog(k)
        else:
            rdBase.DisableLog(k)


@contextmanager
def catch_warning(**kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", **kwargs)
        yield


def get_centroid(coordinates):
    """Centroid for an array of XYZ coordinates"""
    return np.mean(coordinates, axis=0)


def get_ring_normal_vector(centroid, coordinates):
    """Returns a vector that is normal to the ring plane"""
    # A & B are two edges of the ring
    a = Point3D(*coordinates[0])
    b = Point3D(*coordinates[1])
    # vectors between centroid and these edges
    ca = centroid.DirectionVector(a)
    cb = centroid.DirectionVector(b)
    # cross product between these two vectors
    normal = ca.CrossProduct(cb)
    # cb.CrossProduct(ca) is the normal vector in the opposite direction
    return normal


def angle_between_limits(angle, min_angle, max_angle, ring=False):
    """Checks if an angle value is between min and max angles in radian

    Parameters
    ----------
    angle : float
        Angle to check, in radians
    min_angle : float
        Lower bound angle, in radians
    max_angle : float
        Upper bound angle, in radians
    ring : bool
        Wether the angle being checked involves a ring or not

    Notes
    -----
    When ``ring=True``, the angle is capped between 0 and 90, and so should be
    the min and max angles. This is useful for angles involving a ring's plane
    normal vector.
    """
    if ring:
        if angle >= pi:
            angle %= _90_deg_to_rad
        elif angle > _90_deg_to_rad:
            angle = _90_deg_to_rad - (angle % _90_deg_to_rad)
    return min_angle <= angle <= max_angle


def get_residues_near_ligand(lig, prot, cutoff=6.0):
    """Detects residues close to a reference ligand

    Parameters
    ----------
    lig : prolif.molecule.Molecule
        Select residues that are near this ligand
    prot : prolif.molecule.Molecule
        Protein containing the residues
    cutoff : float
        If any interatomic distance between the ligand reference points and a
        residue is below or equal to this cutoff, the residue will be selected

    Returns
    -------
    residues : list
        A list of unique :class:`~prolif.residue.ResidueId` that are close to
        the ligand
    """
    tree = cKDTree(prot.xyz)
    ix = tree.query_ball_point(lig.xyz, cutoff)
    ix = set([i for lst in ix for i in lst])
    resids = [ResidueId.from_atom(prot.GetAtomWithIdx(i)) for i in ix]
    return list(set(resids))


def split_mol_by_residues(mol):
    """Splits a molecule in multiple fragments based on residues

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to fragment

    Returns
    -------
    residues : list
        A list of :class:`rdkit.Chem.rdchem.Mol`

    Notes
    -----
    Code adapted from Maciek Wójcikowski on the RDKit discussion list
    """
    residues = []
    for res in SplitMolByPDBResidues(mol).values():
        for frag in GetMolFrags(res, asMols=True, sanitizeFrags=False):
            # count number of unique residues in the fragment
            resids = {a.GetIdx(): ResidueId.from_atom(a) for a in frag.GetAtoms()}
            if len(set(resids.values())) > 1:
                # split on peptide bonds
                bonds = [
                    b.GetIdx() for b in frag.GetBonds() if is_peptide_bond(b, resids)
                ]
                mols = FragmentOnBonds(frag, bonds, addDummies=False)
                mols = GetMolFrags(mols, asMols=True, sanitizeFrags=False)
                residues.extend(mols)
            else:
                residues.append(frag)
    return residues


def is_peptide_bond(bond, resids):
    """Checks if a bond is a peptide bond based on the ResidueId of the atoms
    on each part of the bond. Also works for disulfide bridges or any bond that
    links two residues in biopolymers.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        The bond to check
    resids : dict
        A dictionnary of ResidueId indexed by atom index
    """
    return resids[bond.GetBeginAtomIdx()] != resids[bond.GetEndAtomIdx()]


def to_dataframe(
    ifp,
    interactions,
    index_col="Frame",
    dtype=None,
    drop_empty=True,
    return_atoms=False,
):
    """Converts IFPs to a pandas DataFrame

    Parameters
    ----------
    ifp : list
        A list of dict in the format {key: bitvector}. "key" is a tuple of
        ligand and protein ResidueId. "bitvector" is either a numpy.ndarray
        of bits, or a list of bitarray, ligand atom indices, and protein atom
        indices. Each dictionnary must also contain an entry that will be used
        as an index, typically a frame number.
    interactions : list
        A list of interactions, in the same order as the bitvector.
    index_col : str
        The dictionnary key that will be used as an index in the DataFrame
    dtype : object or None
        Cast the input of each bit in the bitvector to this type. If None, keep
        the data as is. Not compatible with ``return_atoms=True``
    drop_empty : bool
        Drop columns with only empty values
    return_atoms : bool
        For each residue pair and interaction, return indices of atoms
        responsible for the interaction instead of bits

    Returns
    -------
    df : pandas.DataFrame
        A 3-levels DataFrame where each ligand residue, protein residue, and
        interaction type are in separate columns

    Example
    -------
    ::

        >>> df = prolif.to_dataframe(results, fp.interactions.keys(), dtype=int)
        >>> print(df)
        ligand             LIG1.G
        protein             ILE59                  ILE55       TYR93
        interaction   Hydrophobic HBAcceptor Hydrophobic Hydrophobic PiStacking
        Frame
        0                       0          1           0           0          0
        ...

    .. versionchanged:: 0.3.2
        Moved the ``return_atoms`` parameter from the ``run`` methods to the
        dataframe conversion code
    """
    if dtype and return_atoms:
        raise ValueError("`dtype` cannot be used with `return_atoms=True`")
    ifp = deepcopy(ifp)
    n_interactions = len(interactions)
    empty_value = dtype(False) if dtype else False
    # residue pairs
    keys = sorted(set([k for d in ifp for k in d.keys() if k != index_col]))
    # check if each interaction value is a list of atom indices or smthg else
    has_atom_indices = False
    for d in ifp:
        for key, value in d.items():
            if key != index_col:
                has_atom_indices = isinstance(value[0], Iterable)
                break
    if return_atoms and not has_atom_indices:
        raise ValueError(
            "The IFP either doesn't contain atom indices or is formatted incorrectly"
        )
    # create empty array for each residue pair interaction that doesn't exist
    # in a particular frame
    if has_atom_indices and return_atoms:
        empty_arr = [[None, None]] * n_interactions
    else:
        empty_arr = np.array([empty_value] * n_interactions)
    # sparse to dense
    data = defaultdict(list)
    index = []
    for d in ifp:
        index.append(d.pop(index_col))
        for key in keys:
            try:
                arr = d[key]
            except KeyError:
                data[key].append(empty_arr)
            else:
                if has_atom_indices and return_atoms:
                    arr = list(zip(*arr[1:]))
                elif has_atom_indices:
                    arr = arr[0]
                data[key].append(arr)
    index = pd.Series(index, name=index_col)
    # create dataframes
    if not data:
        warnings.warn("No interaction detected")
        return pd.DataFrame([], index=index)
    values = np.array(
        [np.hstack([np.ravel(a[i]) for a in data.values()]) for i in range(len(index))]
    )
    if has_atom_indices and return_atoms:
        columns = pd.MultiIndex.from_tuples(
            [
                (str(k[0]), str(k[1]), i, a)
                for k in keys
                for i in interactions
                for a in ["ligand", "protein"]
            ],
            names=["ligand", "protein", "interaction", "atom"],
        )
    else:
        columns = pd.MultiIndex.from_tuples(
            [(str(k[0]), str(k[1]), i) for k in keys for i in interactions],
            names=["ligand", "protein", "interaction"],
        )
    df = pd.DataFrame(values, columns=columns, index=index)
    if has_atom_indices and return_atoms:
        
        # check which cols only contain None
        isColNotAllNone = ~df.applymap(lambda x: x is None).all()
        
        # ensure the cols left are still always in pairs of two on the last column level axis
        assert df[df.columns[isColNotAllNone]].groupby(axis=1, \
                                             level=["ligand", "protein", "interaction"])\
                                            .apply(lambda g: len(g.columns))\
                                            .eq(2)\
                                            .all()
        # remove the empty columns before grouping
        df = df[df.columns[isColNotAllNone]]
        df = df.groupby(axis=1, level=["ligand", "protein", "interaction"]).agg(tuple)
    if dtype:
        df = df.astype(dtype)
    if drop_empty:
        if has_atom_indices and return_atoms:
            mask = df.apply(lambda s: ~(s.isin([(None, None)]).all()), axis=0)
        else:
            mask = (df != empty_value).any(axis=0)
        df = df.loc[:, mask]
    return df


def pandas_series_to_bv(s):
    bv = ExplicitBitVect(len(s))
    on_bits = np.where(s >= True)[0].tolist()
    bv.SetBitsFromList(on_bits)
    return bv


def to_bitvectors(df):
    """Converts an interaction DataFrame to a list of RDKit ExplicitBitVector

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where each column corresponds to an interaction between two
        residues

    Returns
    -------
    bv : list
        A list of :class:`~rdkit.DataStructs.cDataStructs.ExplicitBitVect`
        for each frame

    Example
    -------
    ::

        >>> from rdkit.DataStructs import TanimotoSimilarity
        >>> bv = prolif.to_bitvectors(df)
        >>> TanimotoSimilarity(bv[0], bv[1])
        0.42

    """
    return df.apply(pandas_series_to_bv, axis=1).tolist()
