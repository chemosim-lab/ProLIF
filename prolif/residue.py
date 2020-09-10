import re
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


_RE_RESID = r'(\w{3})(\d+)\.?(\w)?'


class ResidueId:
    """Residue Id"""
    def __init__(self, name: str, number: int, chain: str):
        self.name = name
        self.number = number
        self.chain = chain
        self.resid = f"{self.name}{self.number}"
        if self.chain:
            self.resid += f".{self.chain}"

    def __repr__(self):
        return self.resid

    def __hash__(self):
        return hash((self.name, self.number, self.chain))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def from_atom(cls, atom):
        mi = atom.GetMonomerInfo()
        if mi:
            resname = mi.GetResidueName() if mi.GetResidueName() else "UNK"
            return cls(resname, mi.GetResidueNumber(), mi.GetChainId())
        return cls("UNK", 1, "")

    @classmethod
    def from_string(cls, resid_str):
        matches = re.search(_RE_RESID, resid_str)
        resname, resnumber, chain = matches.groups()
        return cls(resname, int(resnumber), chain if chain else "")


class Residue(Chem.Mol):
    """Residue"""

    def __init__(self, mol):
        super().__init__(mol)
        self.resid = ResidueId.from_atom(mol.GetAtomWithIdx(0))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} {self.resid} at {id(self):#x}>"

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())
