import re
from math import pi
from functools import wraps
from operator import attrgetter
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Geometry import Point3D
from .residue import ResidueId
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
    fct_idx = np.argmax(matrix[cst_idx])
    fct = Point3D(*coords[fct_idx])
    # farthest from fct
    ftf_idx = np.argmax(matrix[fct_idx])
    ftf = Point3D(*coords[ftf_idx])
    return np.array((ctd, cst, fct, ftf))


def detect_pocket_residues(prot, lig, cutoff=6.0):
    """Detect residues close to a reference ligand"""
    ref_points = get_reference_points(lig)
    pocket_residues = []
    for point in ref_points:
        distances = np.linalg.norm(point - prot.xyz, axis=1)
        indices = np.where(distances <= cutoff)[0]
        pocket_residues.extend([ResidueId.from_atom(
            prot.GetAtomWithIdx(int(i))) for i in indices])
    return sorted(set(pocket_residues), key=attrgetter("chain", "number"))
