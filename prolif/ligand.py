"""
   Copyright 2017 Cédric BOUYSSET

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os.path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolTransforms, rdmolops
from rdkit import Geometry as rdGeometry
from numpy import argmax
from .logger import logger

class Ligand:
    """Class for a ligand"""
    def __init__(self, inputFile):
        """Initialize the ligand from a file"""
        self.inputFile = inputFile
        fileExtension = os.path.splitext(inputFile)[1]
        if fileExtension.lower() == '.mol2':
            logger.debug('Reading {}'.format(self.inputFile))
            self.mol = Chem.MolFromMol2File(inputFile, sanitize=True, removeHs=False)
        else:
            raise ValueError('{} files are not supported for the ligand.'.format(fileExtension[1:].upper()))
        # Set Centroid
        self.coordinates = self.mol.GetConformer().GetPositions()
        self.centroid = rdMolTransforms.ComputeCentroid(self.mol.GetConformer())
        logger.debug('Ligand centroid was detected as x={:.3f} y={:.3f} z={:.3f}'.format(*[c for c in self.centroid]))


    def __repr__(self):
        return self.inputFile


    def setIFP(self, IFP, vector):
        """Set the IFP for the ligand, as a bitstring and vector"""
        self.IFP = IFP
        self.IFPvector = vector


    def getSimilarity(self, reference, method='tanimoto', alpha=None, beta=None):
        if   method == 'tanimoto':
            return DataStructs.TanimotoSimilarity(reference.IFPvector, self.IFPvector)
        elif method == 'dice':
            return DataStructs.DiceSimilarity(reference.IFPvector, self.IFPvector)
        elif method == 'tversky':
            return DataStructs.TverskySimilarity(reference.IFPvector, self.IFPvector, alpha, beta)


    def setSimilarity(self, score):
        """Set the value for the similarity score between the ligand and a reference"""
        self.score = score


    def get_USRlike_atoms(self):
        """Returns 4 rdkit Point3D objects similar to those used in USR:
        - centroid (ctd)
        - closest to ctd (cst)
        - farthest from cst (fct) (usually ctd but let's avoid computing too many dist matrices)
        - farthest from fct (ftf)"""
        matrix = rdmolops.Get3DDistanceMatrix(self.mol)
        conf = self.mol.GetConformer()
        coords = conf.GetPositions()

        # centroid
        ctd = rdMolTransforms.ComputeCentroid(conf)

        # closest to centroid
        min_dist = 100
        for atom in self.mol.GetAtoms():
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

        logger.debug('ctd={}'.format(list(ctd)))
        logger.debug('cst={}'.format(list(cst)))
        logger.debug('fct={}'.format(list(fct)))
        logger.debug('ftf={}'.format(list(ftf)))
        return ctd, cst, fct, ftf
