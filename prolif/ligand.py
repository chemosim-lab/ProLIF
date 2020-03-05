import os.path
import logging
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolTransforms, rdmolops
from rdkit import Geometry as rdGeometry
from numpy import argmax
from .trajectory import Trajectory

logger = logging.getLogger("prolif")

class Ligand(Trajectory):
    """The Ligand class"""

    def __init__(self, topology, coordinates, name="ligand"):
        super().__init__(topology, coordinates, name=name)

    def get_USRlike_atoms(self, frame=0):
        """Returns 4 rdkit Point3D objects similar to those used in the USR method:
        - centroid (ctd)
        - closest to ctd (cst)
        - farthest from cst (fct)
        - farthest from fct (ftf)"""
        frame = self.get_frame(frame)
        matrix = rdmolops.Get3DDistanceMatrix(frame)
        conf = frame.GetConformer()
        coords = conf.GetPositions()

        # centroid
        ctd = frame.centroid
        # closest to centroid
        min_dist = 100
        for atom in self.GetAtoms():
            point = rdGeometry.Point3D(*coords[atom.GetIdx()])
            dist = ctd.Distance(point)
            if dist < min_dist:
                min_dist = dist
                cst = point
                cst_idx = atom.GetIdx()
        # farthest from cst
        fct_idx = argmax(matrix[cst_idx])
        fct = rdGeometry.Point3D(*coords[fct_idx])
        # farthest from fct
        ftf_idx = argmax(matrix[fct_idx])
        ftf = rdGeometry.Point3D(*coords[ftf_idx])
        logger.debug('centroid (ctd) = {}'.format(list(ctd)))
        logger.debug('closest to ctd (cst) = {}'.format(list(cst)))
        logger.debug('farthest from cst (fct) = {}'.format(list(fct)))
        logger.debug('farthest from fct (ftf) = {}'.format(list(ftf)))
        return ctd, cst, fct, ftf
