import re
from collections import UserDict
from collections.abc import Iterable
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


class ResidueGroup(UserDict):
    """Residue container"""

    def __init__(self, residues):
        """Create a ResidueGroup from a dict {ResidueId: Residue} or an
        iterable of tuples in the form (ResidueId, Residue)
        """
        super().__init__(residues)
        self._residues_indices = list(self.keys())

    def __getitem__(self, key):
        if isinstance(key, ResidueId):
            return self.data[key]
        elif isinstance(key, int):
            resid = self._residues_indices[key]
            return self.data[resid]
        elif isinstance(key, str):
            resid = ResidueId.from_string(key)
            return self.data[resid]
        elif isinstance(key, slice):
            resids = self._residues_indices[key]
            return ResidueGroup((resid, self.data[resid]) for resid in resids)
        elif isinstance(key, Iterable):
            if isinstance(key[0], ResidueId):
                resids = key
            elif isinstance(key[0], int):
                resids = [self._residues_indices[i] for i in key]
            elif isinstance(key[0], str):
                resids = [ResidueId.from_string(s) for s in key]
            return ResidueGroup((resid, self.data[resid]) for resid in resids)
        raise KeyError("Expected a ResidueId, int, str, an iterable of those "
                       "or a slice, got %s instead" % type(key))
    
    def __setitem__(self, key, value):
        if isinstance(key, ResidueId):
            resid = key
        elif isinstance(key, int):
            resid = self._residues_indices[key]
        elif isinstance(key, str):
            resid = ResidueId.from_string(key)
        else:
            raise KeyError("Expected a ResidueId, int or str, got "
                           f"{type(key)} instead")
        self.data[resid] = value

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} with {self.n_residues} residues at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self)