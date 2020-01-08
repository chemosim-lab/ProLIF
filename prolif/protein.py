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

from rdkit import Chem
from rdkit import Geometry as rdGeometry
import os.path
from .residue import Residue
from .utils import mol2_reader
from .logger import logger

class Protein:
    """Class for a protein"""

    def __init__(self, inputFile, reference=None, cutoff=10.0, residueList=None):
        """Initialization of the protein, defined by a list of residues"""
        self.residueList = residueList
        self.residues = {}
        self.inputFile = inputFile
        fileExtension = os.path.splitext(inputFile)[1]
        if fileExtension.lower() == '.mol2':
            logger.debug('Reading {}'.format(self.inputFile))
            self.residuesFromMOL2File()
        else:
            raise ValueError('{} files are not supported for the protein.'.format(fileExtension[1:].upper()))
        if not self.residueList:
            logger.info('Detecting residues within {} Å of the reference molecule'.format(cutoff))
            self.residueList = self.detectCloseResidues(reference, cutoff)
        self.cleanResidues()

    def __repr__(self):
        return self.inputFile

    def residuesFromMOL2File(self):
        """Read a MOL2 file and assign each line to an object of class Atom"""
        # Create a list of molecule with RDKIT
        residues_list = mol2_reader(self.inputFile, ignoreH=False)
        # Loop through each RDKIT molecule and create a Residue
        for mol in residues_list:
            resname = mol.GetProp('resname')
            self.residues[resname] = Residue(mol)
        logger.debug('Read {} residues'.format(len(self.residues)))

    def detectCloseResidues(self, reference, cutoff=5.0):
        """Detect residues close to a reference ligand"""
        residueList = []
        for ref_point in reference.get_USRlike_atoms():
            for residue in self.residues:
                if self.residues[residue].centroid.Distance(ref_point) > 14:
                    # skip residues with centroid far from ligand reference point
                    continue
                if self.residues[residue].resname in residueList:
                    # skip residues already inside the list
                    continue
                for atom in self.residues[residue].mol.GetConformer().GetPositions():
                    res_point = rdGeometry.Point3D(*atom)
                    dist = ref_point.Distance(res_point)
                    if dist <= cutoff:
                        residueList.append(self.residues[residue].resname)
                        break
        logger.info('Detected {} residues'.format(len(residueList)))
        return residueList

    def cleanResidues(self):
        """Cleans the residues of the protein to only keep those in self.residueList"""
        residues = {}
        for residue in self.residues:
            if residue in self.residueList:
                residues[residue] = self.residues[residue]
        self.residues = residues
