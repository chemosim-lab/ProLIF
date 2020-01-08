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
