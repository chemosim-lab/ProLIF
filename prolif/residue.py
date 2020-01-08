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
from rdkit.Chem import rdMolTransforms
from .logger import logger

class Residue:
    """Class for a residue in a protein"""

    def __init__(self, mol):
        self.mol      = mol  # RDkit molecule
        self.resname  = self.mol.GetProp('resname')  # unique identifier for the residue
        self.coordinates = self.mol.GetConformer().GetPositions() # atomic coordinates of the residue
        self.centroid    = rdMolTransforms.ComputeCentroid(self.mol.GetConformer())  # centroid of the residue


    def __repr__(self):
        return self.resname
