"""
Residue-related classes --- :mod:`prolif.residue`
=================================================
"""

import re
from collections import UserDict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from rdkit.Chem.rdmolops import FastFindRings

from prolif.rdkitmol import BaseRDKitMol

if TYPE_CHECKING:
    from rdkit import Chem

    from prolif.typeshed import ResidueKey

_RE_RESID = re.compile(
    r"(TIP[234]|T[234]P|H2O|[0-9][A-Z]{2}|[A-Z ]+)?(\d*)\.?([A-Z\d]{1,2})?"
)


class ResidueId:
    """Residue identifier

    Parameters
    ----------
    name : str or None, default = "UNK"
        Residue name
    number : int or None, default = 0
        Residue number
    chain : str or None, default = None
        Protein chain or segment index

    Notes
    -----
    Whitespaces are stripped from the name and chain.

    .. versionchanged:: 2.1.0
        Whitespaces are now stripped from the name and chain. Better support for water
        and monatomic ion residue names. Ability to use the segment index as chain.
    """

    def __init__(
        self,
        name: str | None = "UNK",
        number: int | None = 0,
        chain: str | None = None,
    ):
        self.name = "UNK" if not name else name.strip()
        self.number = number or 0
        self.chain = None if not chain else chain.strip()

    def __repr__(self) -> str:
        return f"ResidueId({self.name}, {self.number}, {self.chain})"

    def __str__(self) -> str:
        resid = f"{self.name}{self.number}"
        if self.chain:
            return f"{resid}.{self.chain}"
        return resid

    def __hash__(self) -> int:
        return hash((self.name, self.number, self.chain))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueId):
            return NotImplemented
        return (self.name, self.number, self.chain) == (
            other.name,
            other.number,
            other.chain,
        )

    def __lt__(self, other: "ResidueId") -> bool:
        return (_chain_key(self.chain), self.number) < (
            _chain_key(other.chain),
            other.number,
        )

    @classmethod
    def from_atom(cls, atom: "Chem.Atom", use_segid: bool = False) -> "ResidueId":
        """Creates a ResidueId from an RDKit atom

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            An atom that contains an RDKit :class:`~rdkit.Chem.rdchem.AtomMonomerInfo`
        use_segid: bool, default = False
            Use the segment number rather than the chain identifier as a chain
        """
        mi = atom.GetMonomerInfo()
        if mi:
            name = mi.GetResidueName()
            number = mi.GetResidueNumber()
            chain = str(mi.GetSegmentNumber()) if use_segid else mi.GetChainId()
            return cls(name, number, chain)
        return cls()

    @classmethod
    def from_string(cls, resid_str: str) -> "ResidueId":
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
        matches = cast(re.Match, _RE_RESID.search(resid_str))
        name, number, chain = matches.groups()
        number = int(number) if number else 0
        return cls(name, number, chain)


def _chain_key(chain: str | None) -> tuple[bool, str | None]:
    """Handles the case where the two chains are of different types"""
    # e.g., None from WAT123, and str from ALA42.A
    return (chain is not None, chain)


class Residue(BaseRDKitMol):
    """A class for residues as RDKit molecules

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The residue as an RDKit molecule
    use_segid: bool, default = False
        Use the segment number rather than the chain identifier as a chain

    Attributes
    ----------
    resid : prolif.residue.ResidueId
        The residue identifier

    Notes
    -----
    The name of the residue can be converted to a string by using
    ``str(Residue)``

    .. versionchanged:: 2.1.0
        Added `use_segid`.
    """

    def __init__(self, mol: "Chem.Mol", *, use_segid: bool = False):
        super().__init__(mol)
        FastFindRings(self)
        self.resid = ResidueId.from_atom(self.GetAtomWithIdx(0), use_segid=use_segid)

    def __repr__(self) -> str:  # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} {self.resid} at {id(self):#x}>"

    def __str__(self) -> str:
        return str(self.resid)


class ResidueGroup(UserDict[ResidueId, Residue]):
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

    def __init__(self, residues: Iterable[Residue]):
        self._residues = cast(Sequence[Residue], np.asarray(residues, dtype=object))
        resinfo = [
            (r.resid.name, r.resid.number, r.resid.chain) for r in self._residues
        ]
        try:
            name, number, chain = zip(*resinfo, strict=True)
        except ValueError:
            self.name = np.array([], dtype=object)
            self.number = np.array([], dtype=np.uint8)
            self.chain = np.array([], dtype=object)
        else:
            self.name = np.asarray(name, dtype=object)
            self.number = np.asarray(number, dtype=np.uint16)
            self.chain = np.asarray(chain, dtype=object)
        super().__init__([(r.resid, r) for r in self._residues])

    def __getitem__(self, key: "ResidueKey") -> Residue:
        # bool is a subclass of int but shouldn't be used here
        if isinstance(key, bool):
            raise KeyError(
                f"Expected a ResidueId, int, or str, got {type(key).__name__!r}"
                " instead",
            )
        if isinstance(key, int):
            return self._residues[key]
        if isinstance(key, str):
            key = ResidueId.from_string(key)
            return self.data[key]
        if isinstance(key, ResidueId):
            return self.data[key]
        raise KeyError(
            f"Expected a ResidueId, int, or str, got {type(key).__name__!r} instead",
        )

    def select(self, mask: Any) -> "ResidueGroup":
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

    def __repr__(self) -> str:  # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} with {self.n_residues} residues at {id(self):#x}>"

    @property
    def n_residues(self) -> int:
        return len(self)
