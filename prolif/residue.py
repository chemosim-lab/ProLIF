"""
Residue-related classes --- :mod:`prolif.residue`
=================================================
"""
import re
from typing import Optional, List, Tuple
from collections import UserDict
from collections.abc import Iterable
import numpy as np
from .rdkitmol import BaseRDKitMol


_RE_RESID = re.compile(r'([A-Z]{3})?(\d*)\.?([A-Z])?')


class ResidueId:
    """A unique residue identifier
    
    Parameters
    ----------
    name : str or None, optionnal
        3-letter residue name
    number : int or None, optionnal
        residue number
    chain : str or None, optionnal
        1-letter protein chain
    """
    def __init__(self,
                 name: Optional[str] = None,
                 number: Optional[int] = None,
                 chain: Optional[str] = None):
        self.name = name or None
        self.number = number or None
        self.chain = chain or None
        self.resid = f"{self.name or ''}{self.number or ''}"
        if self.chain:
            self.resid += f".{self.chain}"

    def __repr__(self):
        return self.resid

    def __hash__(self):
        return hash((self.name, self.number, self.chain))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __contains__(self, other):
        attributes = [attr for attr in ["name", "number", "chain"]
                      if getattr(other, attr)]
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attributes)

    @classmethod
    def from_atom(cls, atom):
        """Creates a ResidueId from an RDKit atom
        
        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            An atom that contains an RDKit :class:`~rdkit.Chem.rdchem.AtomMonomerInfo`
        """
        mi = atom.GetMonomerInfo()
        if mi:
            name = mi.GetResidueName()
            number = mi.GetResidueNumber()
            chain = mi.GetChainId()
            return cls(name, number, chain)
        return cls()

    @classmethod
    def from_string(cls, resid_str):
        """Creates a ResidueId from a string
        
        Parameters
        ----------
        resid_str : str
            A string in the format ``<3-letter code><residue number>.<chain>``
            All arguments are optionnal, and the dot should be present only if
            the chain identifier is also present
        
        Examples
        --------

        +-----------+----------------------------------+
        | string    | Corresponding ResidueId          |
        +===========+==================================+
        | "ALA10.A" | ``ResidueId("ALA", 10, "A")``    |
        +-----------+----------------------------------+
        | "GLU33"   | ``ResidueId("GLU", 33, None)``   |
        +-----------+----------------------------------+
        | "LYS.B"   | ``ResidueId("LYS", None, "B")``  |
        +-----------+----------------------------------+
        | "ARG"     | ``ResidueId("ARG", None, None)`` |
        +-----------+----------------------------------+
        | "5.C"     | ``ResidueId(None, 5, "C")``      |
        +-----------+----------------------------------+
        | "42"      | ``ResidueId(None, 42, None)``    |
        +-----------+----------------------------------+
        | ".D"      | ``ResidueId(None, None, "D")``   |
        +-----------+----------------------------------+
        | ""        | ``ResidueId(None, None, None)``  |
        +-----------+----------------------------------+

        """
        matches = _RE_RESID.search(resid_str)
        name, number, chain = matches.groups()
        number = int(number) if number else None
        return cls(name, number, chain)


class Residue(BaseRDKitMol):
    """A class for residues as RDKit molecules

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The residue as an RDKit molecule

    Attributes
    ----------
    resid : prolif.residue.ResidueId
        The residue identifier
    
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


class ResidueGroup(UserDict):
    """A container to store and retrieve Residue instances easily
    
    Parameters
    ----------
    residues : list
        A list of (:class:`ResidueId`, :class:`Residue`) tuples

    Attributes
    ----------
    n_residues : int
        Number of residues in the ResidueGroup
    
    Notes
    -----
    Residues in the group can be accessed by :class:`ResidueId`, string, index,
    slice, or a list of those.
    See the :class:`~prolif.molecule.Molecule` class for an example
    """

    def __init__(self, residues: List[Tuple[ResidueId, Residue]]):
        super().__init__(residues)
        _resids, _residues = zip(*residues)
        self._resids = np.asarray(_resids, dtype=object)
        self._residues = np.asarray(_residues, dtype=object)

    def __getitem__(self, key):
        if isinstance(key, [int, slice]):
            return self._residues[key]
        elif isinstance(key, str):
            resid = ResidueId.from_string(key)
            try:
                return self.data[key]
            except KeyError:
                ix = [i for i, resid in enumerate(self.keys()) if key in resid]
                return self._residues[ix]
        elif isinstance(key, ResidueId):
            try:
                return self.data[key]
            except KeyError:
                ix = [i for i, resid in enumerate(self.keys()) if key in resid]
                return self._residues[ix]
        elif isinstance(key, Iterable):
            if isinstance(key[0], int):
                return self._residues[key]
            elif isinstance(key[0], str):
                resids = [ResidueId.from_string(s) for s in key]
                ix = [i for i, resid in enumerate(self.keys())
                      if any(key in resid for key in resids)]
            elif isinstance(key[0], ResidueId):
                resids = key
                ix = [i for i, resid in enumerate(self.keys())
                      if any(key in resid for key in resids)]
            return self._residues[ix]
        raise KeyError("Expected a ResidueId, int, str, an iterable of those "
                       f"or a slice, got {type(key).__name__!r} instead")

    def __setitem__(self, key, value):
        if isinstance(key, ResidueId):
            resid = key
        elif isinstance(key, int):
            resid = self._resids[key]
        elif isinstance(key, str):
            resid = ResidueId.from_string(key)
        else:
            raise KeyError("Expected a ResidueId, int or str, got "
                           f"{type(key).__name__!r} instead")
        self.data[resid] = value

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} with {self.n_residues} residues at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self)
