from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .utils import get_resid


class Residue(Chem.Mol):
    """Residue"""

    def __init__(self, mol):
        super().__init__(mol)
        self.resid = get_resid(mol.GetAtomWithIdx(0))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} {self.resname} at {id(self):#x}>"

    @property
    def resname(self):
        if self.resid.chain:
            return f"{self.resid.name}{self.resid.number}.{self.resid.chain}"
        return f"{self.resid.name}{self.resid.number}"

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())
