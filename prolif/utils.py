"""
Helper functions --- :mod:`prolif.utils`
========================================
"""
from math import pi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from rdkit.Chem import (rdmolops,
                        SplitMolByPDBResidues,
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
    """Check if an angle value is between min and max angles in radian.
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
    """Detect residues close to a reference ligand
    
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
    ix = list(set([i for lst in ix for i in lst]))
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


def to_dataframe(ifp, fingerprint):
    """Convert IFPs to a pandas DataFrame

    Parameters
    ----------
    ifp : list
        A list of dict in the format {ResidueId: fingerprint} where
        fingerprint is a numpy.ndarray obtained by running the
        :meth:`~prolif.fingerprint.Fingerprintrun` method of a
        :class:`~prolif.fingerprint.Fingerprint`. Each dictionnary can
        contain other (key, value) pairs, such as frame numbers...etc. as long
        as the values are not numpy arrays or np.NaN
    fingerprint : prolif.fingerprint.Fingerprint
        The fingerprint generator that was used to obtain the fingerprint

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame where each residue and interaction type are in separate
        columns

    Example
    -------
    ::

        >>> df = prolif.to_dataframe(results, fp)
        >>> print(df)
        Frame     ILE59                  ILE55       TYR93
                Hydrophobic HBAcceptor Hydrophobic Hydrophobic PiStacking
        0      0           1          0           0           0          0
        ...

    """
    df = pd.DataFrame(ifp)
    resids = list(set(key for d in ifp
                          for key, value in d.items()
                          if isinstance(value, np.ndarray)))
    resids.sort()
    ids = df.drop(columns=resids).columns.tolist()
    df = df.applymap(lambda x: [False] * fingerprint.n_interactions
                               if x is np.nan else x)
    ifps = pd.DataFrame()
    for res in resids:
        cols = [f"{res}{i}" for i in range(fingerprint.n_interactions)]
        ifps[cols] = df[res].apply(pd.Series)
    ifps.columns = pd.MultiIndex.from_product([[str(r) for r in resids],
                                               fingerprint.interactions],
                                              names=["residue", "interaction"])
    ifps = ifps.astype(np.uint8)
    ifps = ifps.loc[:, (ifps != 0).any(axis=0)]
    temp = df[ids].copy()
    temp.columns = pd.MultiIndex.from_product([ids, [""]])
    return pd.concat([temp, ifps], axis=1)


def _series_to_bv(s):
    bv = ExplicitBitVect(len(s))
    on_bits = np.where(s == 1)[0].tolist()
    bv.SetBitsFromList(on_bits)
    return bv


def to_bitvectors(ifp, fingerprint):
    """Convert IFPs to a list of RDKit BitVector

    Parameters
    ----------
    ifp : list
        A list of dict in the format {ResidueId: fingerprint} where
        fingerprint is a numpy.ndarray obtained by running the
        :meth:`~prolif.fingerprint.Fingerprint.run` method of a
        :class:`~prolif.fingerprint.Fingerprint`. Each dictionnary can
        contain other (key, value) pairs, such as frame numbers...etc. as long
        as these values are not numpy arrays or np.NaN
    fingerprint : prolif.fingerprint.Fingerprint
        The fingerprint that was used to generate the fingerprint

    Returns
    -------
    bv : list
        A list of :class:`~rdkit.DataStructs.cDataStructs.ExplicitBitVect`
        for each frame

    Example
    -------
    ::

        >>> from rdkit.DataStructs import TanimotoSimilarity
        >>> bv = prolif.to_bitvectors(results, fp)
        >>> TanimotoSimilarity(bv[0], bv[1])
        0.42

    """
    df = to_dataframe(ifp, fingerprint)
    resids = list(set(str(key) for d in ifp for key, value in d.items()
                      if (isinstance(value, np.ndarray) and value.sum() > 0)))
    if not resids:
        raise ValueError("The input IFP only contains off bits")
    return df[resids].apply(_series_to_bv, axis=1).tolist()