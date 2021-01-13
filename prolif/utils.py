"""
Helper functions --- :mod:`prolif.utils`
========================================
"""
from math import pi
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from rdkit.Chem import (SplitMolByPDBResidues,
                        GetMolFrags,
                        FragmentOnBonds)
from rdkit.Geometry import Point3D
from rdkit.DataStructs import ExplicitBitVect
from .residue import ResidueId


_90_deg_to_rad = pi/2


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
    """Checks if an angle value is between min and max angles in radian.
    If the angle to check involves a ring, include the angle that would be
    obtained if we had used the other normal vector (same axis but opposite
    direction)
    """
    if ring and (angle > _90_deg_to_rad):
        mirror_angle = _90_deg_to_rad - (angle % _90_deg_to_rad)
        return (min_angle <= angle <= max_angle) or (
                min_angle <= mirror_angle <= max_angle)
    return (min_angle <= angle <= max_angle)


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
            resids = {a.GetIdx(): ResidueId.from_atom(a)
                      for a in frag.GetAtoms()}
            if len(set(resids.values())) > 1:
                # split on peptide bonds
                bonds = [b.GetIdx() for b in frag.GetBonds()
                         if is_peptide_bond(b, resids)]
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
    if resids[bond.GetBeginAtomIdx()] == resids[bond.GetEndAtomIdx()]:
        return False
    return True


def to_dataframe(ifp, interactions, index_col="Frame", dtype=None,
                 drop_empty=True):
    """Converts IFPs to a pandas DataFrame

    Parameters
    ----------
    ifp : list
        A list of dict in the format {key: bitvector} where
        "bitvector" is a numpy.ndarray obtained by running the
        :meth:`~prolif.fingerprint.Fingerprint.bitvector` method of a
        :class:`~prolif.fingerprint.Fingerprint`, and "key" is a tuple of
        ligand and protein ResidueId. Each dictionnary must also contain an
        entry that will be used as an index, typically a frame number.
    interactions : list
        A list of interactions, in the same order as the bitvector.
    index_col : str
        The dictionnary key that will be used as an index in the DataFrame
    dtype : object or None
        Cast the input of each bit in the bitvector to this type. If None, keep
        the data as is.
    drop_empty : bool
        Drop columns with only empty values

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

    """
    ifp = deepcopy(ifp)
    n_interactions = len(interactions)
    empty_value = dtype(False) if dtype else False
    # residue pairs
    keys = sorted(set([k for d in ifp for k in d.keys() if k != index_col]))
    # check if each interaction value is a list of atom indices or smthg else
    for k in keys:
        if k in ifp[0].keys():
            break
    is_atompair = isinstance(ifp[0][k][0], Iterable)
    # create empty array for each residue pair interaction that doesn't exist
    # in a particular frame
    if is_atompair:
        empty_arr =  [[None, None]] * n_interactions
    else:
        empty_arr = np.array([empty_value] * n_interactions)
    # sparse to dense
    data = defaultdict(list)
    index = []
    for d in ifp:
        index.append(d.pop(index_col))
        for key in keys:
            try:
                data[key].append(d[key])
            except KeyError:
                data[key].append(empty_arr)
    # create dataframes
    values = np.array([np.hstack([np.ravel(a[i]) for a in data.values()])
                    for i in range(len(index))])
    if is_atompair:
        columns = pd.MultiIndex.from_tuples([(str(k[0]), str(k[1]), i, a) for k in keys
                                        for i in interactions for a in ["ligand", "protein"]],
                                        names=["ligand", "protein", "interaction", "atom"])
    else:
        columns = pd.MultiIndex.from_tuples([(str(k[0]), str(k[1]), i) for k in keys
                                        for i in interactions],
                                        names=["ligand", "protein", "interaction"])
    index = pd.Series(index, name=index_col)
    df = pd.DataFrame(values, columns=columns, index=index)
    if is_atompair:
        df = df.groupby(axis=1, level=["ligand", "protein", "interaction"]).agg(tuple)
    if dtype:
        df = df.astype(dtype)
    if drop_empty:
        if is_atompair:
            mask = df.apply(lambda s:
                            ~(s.isin([(None, None)]).all()), axis=0)
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