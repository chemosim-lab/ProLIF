"""
prolif.residue
==============
"""
import re
from collections import UserDict
from collections.abc import Iterable
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


_RE_RESID = r'(\w{3})(\d+)\.?(\w)?'


class ResidueId:
    """A unique residue identifier
    
    Parameters
    ----------
    name : str
        3-letter residue name
    number : int or None
        residue number
    chain : str or None, optionnal
        1-letter protein chain
    """
    def __init__(self, name:str, number=None, chain=None):
        self.name = name
        self.number = number
        self.chain = chain
        self.resid = f"{self.name or ''}{self.number or ''}"
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
        """Creates a ResidueId from an RDKit atom
        
        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            An atom that contains an RDKit :class:`~rdkit.Chem.rdchem.AtomMonomerInfo`
            (also know as ``PDBResidueInfo``)
        """
        mi = atom.GetMonomerInfo()
        if mi:
            resname = mi.GetResidueName() if mi.GetResidueName() else "UNK"
            return cls(resname, mi.GetResidueNumber(), mi.GetChainId())
        return cls("UNK", 1, "")

    @classmethod
    def from_string(cls, resid_str):
        """Creates a ResidueId from a string
        
        Parameters
        ----------
        resid_str : str
            A string in the format ``<3-letter code><residue number>.<chain>``,
            i.e. for alanine number 10 and chain B: ``ALA10.B``
            The ".chain" is optionnal
        """
        matches = re.search(_RE_RESID, resid_str)
        resname, resnumber, chain = matches.groups()
        return cls(resname, int(resnumber), chain if chain else "")


class Residue(Chem.Mol):
    """An RDKit molecule with custom attributes

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The residue as an RDKit molecule

    Attributes
    ----------
    resid : prolif.residue.ResidueId
        The residue identifier
    centroid : numpy.ndarray
        XYZ coordinates of the centroid of the molecule
    xyz : numpy.ndarray
        XYZ coordinates of all atoms in the molecule
    
    Notes
    -----
    The name of the residue can be converted to a string by using
    ``str(Residue)``
    """
    def __init__(self, mol):
        super().__init__(mol)
        self.resid = ResidueId.from_atom(mol.GetAtomWithIdx(0))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} {self.resid} at {id(self):#x}>"

    def __str__(self):
        return str(self.resid)

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())


class ResidueGroup(UserDict):
    """A container to store and retrieve Residue instances easily"""

    def __init__(self, residues):
        """Create a ResidueGroup from a dict {ResidueId: Residue} or an
        iterable of tuples in the form (ResidueId, Residue)
        """
        super().__init__(residues)
        self._residues_indices = list(self.keys())

    def __getitem__(self, key):
        if isinstance(key, ResidueId):
            if key.number is None:
                return ResidueGroup((resid, self.data[resid])
                                    for resid in self.data.keys()
                                    if resid.name == key.name)
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
                       f"or a slice, got {type(key).__name__!r} instead")

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

    @property
    def resid(self):
        return self._residues_indices