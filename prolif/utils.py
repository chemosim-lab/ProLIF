import re
from math import pi
from functools import wraps
from collections import namedtuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Geometry import Point3D
try:
    import sklearn
except ImportError:
    _has_sklearn = False
else:
    _has_sklearn = True
try:
    import seaborn
except ImportError:
    _has_seaborn = False
else:
    _has_seaborn = True


ResidueId = namedtuple("Residue", ["name", "number", "chain"])


def requires_sklearn(func):
    """Decorator for when sklearn is required in a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_sklearn:
            raise ImportError("sklearn is required for this function, but isn't installed")
        return func(*args, **kwargs)
    return wrapper


def requires_seaborn(func):
    """Decorator for when seaborn is required in a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_seaborn:
            raise ImportError("seaborn is required for this function, but isn't installed")
        return func(*args, **kwargs)
    return wrapper


def requires_config(func):
    """Check if the dataframe has been configured with a FingerprintFactory"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self._configured, "The Dataframe needs to be configured with df.configure()"
        return func(self, *args, **kwargs)
    return wrapper


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
    # note that cb.CrossProduct(ca) will the normal vector in the opposite direction
    return normal


def angle_between_limits(angle, min_angle, max_angle, ring=False):
    """
    Check if an angle value is between min and max angles in degrees.
    If the angle to check involves a ring, include the angle that would be obtained
    if we had used the other normal vector (same axis but opposite direction).
    """
    if ring and (angle > pi/2):
        mirror_angle = (pi/2) - (angle % (pi/2))
        return (min_angle <= angle <= max_angle) or (min_angle <= mirror_angle <= max_angle)
    return (min_angle <= angle <= max_angle)


def get_residue_components(resid_str):
    matches = re.search(r'(\w{3})(\d+)\.?(\w)?', resid_str)
    resname, resnumber, chain = matches.groups()
    return (resname, int(resnumber), chain if chain else "")


def get_resid(atom):
    mi = atom.GetMonomerInfo()
    if mi:
        resname = mi.GetResidueName() if mi.GetResidueName() else "UNK"
        return ResidueId(resname, mi.GetResidueNumber(), 
                         mi.GetChainId())
    return ResidueId("UNK", 1, "")


def get_reference_points(mol):
    """Returns 4 rdkit Point3D objects similar to those used in the USR method:
    - centroid (ctd)
    - closest to ctd (cst)
    - farthest from cst (fct)
    - farthest from fct (ftf)"""
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
    # farthest from cst
    fct_idx = argmax(matrix[cst_idx])
    fct = Point3D(*coords[fct_idx])
    # farthest from fct
    ftf_idx = argmax(matrix[fct_idx])
    ftf = Point3D(*coords[ftf_idx])
    logger.debug('centroid (ctd) = {}'.format(list(ctd)))
    logger.debug('closest to ctd (cst) = {}'.format(list(cst)))
    logger.debug('farthest from cst (fct) = {}'.format(list(fct)))
    logger.debug('farthest from fct (ftf) = {}'.format(list(ftf)))
    return ctd, cst, fct, ftf


def detect_pocket_residues(prot, lig, cutoff=6.0):
    """Detect residues close to a reference ligand"""
    pocket_residues = []
    ref_points = get_reference_points(lig)
    for residue in prot:
        # skip residues already inside the list
        if residue.resname in pocket_residues:
            continue
        # skip residues with centroid far from ligand centroid
        if residue.centroid.Distance(lig.centroid) > 3*cutoff:
            continue
        # iterate over each reference point
        for ref_point in ref_points:
            for atom_crd in residue.xyz:
                resid_point = Point3D(*atom_crd)
                dist = ref_point.Distance(resid_point)
                if dist <= cutoff:
                    pocket_residues.append(residue.resname)
                    break
            if residue.resname in pocket_residues:
                break
    logger.info('Detected {} residues inside the binding pocket'.format(len(pocket_residues)))
    return pocket_residues