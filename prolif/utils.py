"""
Helper functions --- :mod:`prolif.utils`
========================================
"""
from math import pi
from operator import attrgetter
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Geometry import Point3D
from rdkit.DataStructs import ExplicitBitVect
from .residue import ResidueId


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
    """Check if an angle value is between min and max angles in degrees
    If the angle to check involves a ring, include the angle that would be
    obtained if we had used the other normal vector (same axis but opposite
    direction)
    """
    if ring and (angle > pi/2):
        mirror_angle = (pi/2) - (angle % (pi/2))
        return (min_angle <= angle <= max_angle) or (
                min_angle <= mirror_angle <= max_angle)
    return (min_angle <= angle <= max_angle)


def get_reference_points(mol):
    """Returns 4 rdkit Point3D objects similar to those used in the USR method:

    - centroid (ctd)
    - closest to ctd (cst)
    - farthest from cst (fct)
    - farthest from fct (ftf)
    """
    matrix = rdmolops.Get3DDistanceMatrix(mol)
    conf = mol.GetConformer()
    coords = conf.GetPositions()

    # centroid
    ctd = mol.centroid
    # closest to centroid
    min_dist = 100
    for atom in mol.GetAtoms():
        point = Point3D(*coords[atom.GetIdx()])
        dist = ctd.Distance(point)
        if dist < min_dist:
            min_dist = dist
            cst = point
            cst_idx = atom.GetIdx()
    # furthest from cst
    fct_idx = np.argmax(matrix[cst_idx])
    fct = Point3D(*coords[fct_idx])
    # furthest from fct
    ftf_idx = np.argmax(matrix[fct_idx])
    ftf = Point3D(*coords[ftf_idx])
    return ctd, cst, fct, ftf


def get_pocket_residues(lig, prot, cutoff=6.0):
    """Detect residues close to a reference ligand

    Based on the distance between the protein and a few reference points in the
    ligand. The reference points are chosen similarly to the "Ultrafast Shape
    Recognition" technique: centroid, closest to centroid (cst), furthest from
    cst (fct), furthest from fct.
    
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
        Sorted list of :class:`ResidueId` that are close to the ligand
    """
    ref_points = get_reference_points(lig)
    pocket_residues = []
    for point in ref_points:
        distances = np.linalg.norm(point - prot.xyz, axis=1)
        indices = np.where(distances <= cutoff)[0]
        pocket_residues.extend([ResidueId.from_atom(
            prot.GetAtomWithIdx(int(i))) for i in indices])
    return sorted(set(pocket_residues), key=attrgetter("chain", "number"))


def split_mol_in_residues(protein):
    """Splits an RDKit Mol in multiple residues
    Code adapted from Maciek Wójcikowski on the discussion list
    """
    residues = []
    peptide_bond = Chem.MolFromSmarts('N-C-C(=O)-N')
    disulfide_bridge = Chem.MolFromSmarts('S-S')
    for res in Chem.SplitMolByPDBResidues(protein).values():
        for frag in Chem.GetMolFrags(res, asMols=True, sanitizeFrags=False):
            # split on peptide bond
            frags = _split_on_pattern(frag, peptide_bond, (2, 4))
            for f in frags:
                # split on disulfide bridge
                mols = _split_on_pattern(f, disulfide_bridge, (0, 1))
                residues.extend(mols)
    return residues


def _split_on_pattern(mol, smarts, smarts_atom_indices):
    """Splits a molecule in fragments on a bond, given a SMARTS pattern and the
    indices of atoms in the SMARTS pattern between which the split will be done
    """
    bonds = [mol.GetBondBetweenAtoms(match[smarts_atom_indices[0]],
             match[smarts_atom_indices[1]]).GetIdx() for match in
             mol.GetSubstructMatches(smarts)]
    if bonds:
        disconnected_aa = Chem.FragmentOnBonds(mol, bonds, addDummies=False)
        return Chem.GetMolFrags(disconnected_aa, asMols=True,
                                sanitizeFrags=False)
    return (mol,)


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
        The fingerprint that was used to generate the fingerprint

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
    resids = list(set(key for d in ifp for key, value in d.items() if isinstance(value, np.ndarray)))
    resids = sorted(resids, key=attrgetter("chain", "number"))
    ids = df.drop(columns=resids).columns.tolist()
    df = df.applymap(lambda x: [False]*fingerprint.n_interactions if x is np.nan else x)
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


def _series_to_bv(s, n_bits):
    bv = ExplicitBitVect(n_bits)
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
        :meth:`~prolif.fingerprint.Fingerprintrun` method of a
        :class:`~prolif.fingerprint.Fingerprint`. Each dictionnary can
        contain other (key, value) pairs, such as frame numbers...etc. as long
        as the values are not numpy arrays or np.NaN
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
    resids = list(set(str(key) for d in ifp for key, value in d.items() if isinstance(value, np.ndarray)))
    ids = df.drop(columns=resids).columns.tolist()
    n_bits = len(df[resids].columns)
    return df[resids].apply(_series_to_bv, n_bits=n_bits, axis=1).tolist()