"""
Residue-related classes --- :mod:`prolif.residue`
=================================================
"""
import re
from collections import UserDict
from typing import List, Optional

import numpy as np

from .rdkitmol import BaseRDKitMol

_RE_RESID = re.compile(r'([A-Z]{,3})?(\d*)\.?(\w)?')
NoneType = type(None)


class ResidueId:
    """A unique residue identifier

    Parameters
    ----------
    name : str
        3-letter residue name
    number : int
        residue number
    chain : str or None, optionnal
        1-letter protein chain
    """
    def __init__(self,
                 name: str = "UNK",
                 number: int = 0,
                 chain: Optional[str] = None):
        self.name = name or "UNK"
        self.number = number or 0
        self.chain = chain or None

    def __repr__(self):
        return f"ResidueId({self.name}, {self.number}, {self.chain})"

    def __str__(self):
        resid = f"{self.name}{self.number}"
        if self.chain:
            resid += f".{self.chain}"
        return resid

    def __hash__(self):
        return hash((self.name, self.number, self.chain))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return (self.chain, self.number) < (other.chain, other.number)

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
        | "LYS.B"   | ``ResidueId("LYS", 0, "B")``     |
        +-----------+----------------------------------+
        | "ARG"     | ``ResidueId("ARG", 0, None)``    |
        +-----------+----------------------------------+
        | "5.C"     | ``ResidueId("UNK", 5, "C")``     |
        +-----------+----------------------------------+
        | "42"      | ``ResidueId("UNK", 42, None)``   |
        +-----------+----------------------------------+
        | ".D"      | ``ResidueId("UNK", 0, "D")``     |
        +-----------+----------------------------------+
        | ""        | ``ResidueId("UNK", 0, None)``    |
        +-----------+----------------------------------+

        """
        matches = _RE_RESID.search(resid_str)
        name, number, chain = matches.groups()
        number = int(number) if number else 0
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
        self.resid = ResidueId.from_atom(self.GetAtomWithIdx(0))

    def __repr__(self): # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} {self.resid} at {id(self):#x}>"

    def __str__(self):
        return str(self.resid)


class ResidueGroup(UserDict):
    """A container to store and retrieve Residue instances easily

    Parameters
    ----------
    residues : list
        A list of :class:`~prolif.residue.Residue`

    Attributes
    ----------
    n_residues : int
        Number of residues in the ResidueGroup

    Notes
    -----
    Residues in the group can be accessed by :class:`ResidueId`, string, or
    index. See the :class:`~prolif.molecule.Molecule` class for an example.
    You can also use the :meth:`~prolif.residue.ResidueGroup.select` method to
    access a subset of a ResidueGroup.
    """
    def __init__(self, residues: List[Residue]):
        self._residues = np.asarray(residues, dtype=object)
        resinfo = [(r.resid.name, r.resid.number, r.resid.chain)
                   for r in self._residues]
        try:
            name, number, chain = zip(*resinfo)
        except ValueError:
            self.name = np.array([], dtype=object)
            self.number = np.array([], dtype=np.uint8)
            self.chain = np.array([], dtype=object)
        else:
            self.name = np.asarray(name, dtype=object)
            self.number = np.asarray(number, dtype=np.uint16)
            self.chain = np.asarray(chain, dtype=object)
        super().__init__([(r.resid, r) for r in self._residues])

    def __getitem__(self, key):
        # bool is a subclass of int but shouldn't be used here
        if isinstance(key, bool):
            raise KeyError("Expected a ResidueId, int, or str, "
                           f"got {type(key).__name__!r} instead")
        if isinstance(key, int):
            return self._residues[key]
        elif isinstance(key, str):
            key = ResidueId.from_string(key)
            return self.data[key]
        elif isinstance(key, ResidueId):
            return self.data[key]
        raise KeyError("Expected a ResidueId, int, or str, "
                       f"got {type(key).__name__!r} instead")

    def select(self, mask):
        """Locate a subset of a ResidueGroup based on a boolean mask

        Parameters
        ----------
        mask : numpy.ndarray
            A 1D array of ``dtype=bool`` with the same length as the number of
            residues in the ResidueGroup. The mask should be constructed by
            using conditions on the "name", "number", and "chain" residue
            attributes as defined in the :class:`~prolif.residue.ResidueId`
            class

        Returns
        -------
        rg : prolif.residue.ResidueGroup
            A subset of the original ResidueGroup

        Examples
        --------
        ::

            >>> rg
            <prolif.residue.ResidueGroup with 200 residues at 0x7f9a68719ac0>
            >>> rg.select(rg.chain == "A")
            <prolif.residue.ResidueGroup with 42 residues at 0x7fe3fdb86ca0>
            >>> rg.select((10 <= rg.number) & (rg.number < 30))
            <prolif.residue.ResidueGroup with 20 residues at 0x7f5f3c69aaf0>
            >>> rg.select((rg.chain == "B") & (np.isin(rg.name, ["ASP", "GLU"])))
            <prolif.residue.ResidueGroup with 3 residues at 0x7f5f3c510c70>

        As seen in these examples, you can combine masks with different
        operators, similarly to numpy boolean indexing or pandas
        :meth:`~pandas.DataFrame.loc` method

            * AND --> ``&``
            * OR --> ``|``
            * XOR --> ``^``
            * NOT --> ``~``

        """
        return ResidueGroup(self._residues[mask])

    def __repr__(self): # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} with {self.n_residues} residues at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self)
