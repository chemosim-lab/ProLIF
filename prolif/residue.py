"""
Residue-related classes --- :mod:`prolif.residue`
=================================================
"""

import re
from collections import Counter, UserDict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from rdkit.Chem.rdmolops import GetSSSR

from prolif.exceptions import FragmentedResidueError, trigger
from prolif.rdkitmol import BaseRDKitMol

if TYPE_CHECKING:
    from rdkit import Chem

    from prolif.typeshed import ResidueKey

T = TypeVar("T")

_RE_RESID = re.compile(
    r"""
    (?P<name>TIP[234]|T[234]P|H2O|[0-9][A-Z]{2}|[A-Z ]+)?
    (?P<number>\d+)?
    (?P<segment>:\d)?
    (?P<chain>\.[A-Z\d]+)?
    """,
    re.VERBOSE,
)


class ResidueId:
    """Residue identifier

    Parameters
    ----------
    name : str or None, default = None
        Residue name
    number : int or None, default = None
        Residue number
    segment : int or None, default = None
        Segment index
    chain : str or None, default = None
        Residue chain

    Notes
    -----
    Whitespaces are stripped from the name and chain.

    .. versionchanged:: 2.1.0
        Whitespaces are now stripped from the name and chain. Better support for water
        and monatomic ion residue names. Ability to use the segment index as chain.

    .. versionchanged:: 3.0.0
        The ``segment`` is now a separate attribute from the chain.
    """

    def __init__(
        self,
        name: str | None = None,
        number: int | None = None,
        segment: int | None = None,
        chain: str | None = None,
    ) -> None:
        # same defaults as RDKit
        self.name = _name if (_name := (name or "").strip()) else ""
        self.number = number or 0
        self.segment = segment or 0
        self.chain = _chain if (_chain := (chain or "").strip()) else ""

    def __repr__(self) -> str:
        parts = f"{self.name!r}, {self.number}, {self.segment}"
        if self.chain:
            return f"ResidueId({parts}, {self.chain!r})"
        return f"ResidueId({parts})"

    def __str__(self) -> str:
        parts = f"{self.name}{self.number}:{self.segment}"
        if self.chain:
            return f"{parts}.{self.chain}"
        return parts

    def __hash__(self) -> int:
        return hash((self.name, self.number, self.segment, self.chain))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueId):
            return NotImplemented
        return (self.name, self.number, self.segment, self.chain) == (
            other.name,
            other.number,
            other.segment,
            other.chain,
        )

    def __lt__(self, other: "ResidueId") -> bool:
        return (self.chain, self.segment, self.number) < (
            other.chain,
            other.segment,
            other.number,
        )

    @classmethod
    def from_atom(cls, atom: "Chem.Atom") -> "ResidueId":
        """Creates a ResidueId from an RDKit atom

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            An atom that contains an RDKit :class:`~rdkit.Chem.rdchem.AtomMonomerInfo`

        .. versionchanged:: 3.0.0
            Removed the ``use_segid`` parameter. Both the chain and the segment are now
            always extracted from the atom's PDB residue info.
        """
        if mi := atom.GetPDBResidueInfo():
            return cls(
                name=mi.GetResidueName(),
                number=mi.GetResidueNumber(),
                segment=mi.GetSegmentNumber(),
                chain=mi.GetChainId(),
            )
        return cls()

    @classmethod
    def from_string(cls, resid_str: str) -> "ResidueId":
        """Creates a ResidueId from a string

        Parameters
        ----------
        resid_str : str
            A string in the format ``<3-letter code><residue number>.<chain>:<segment>``
            All arguments are optionnal, and the dot should be present only if
            the chain identifier is also present, same for the colon and segment number.

        Examples
        --------

        +------------+----------------------------------+
        | string     | Corresponding ResidueId          |
        +============+==================================+
        | "ALA10.A"  | ``ResidueId('ALA', 10, 0, 'A')`` |
        +------------+----------------------------------+
        | "GLU33"    | ``ResidueId('GLU', 33, 0)``      |
        +------------+----------------------------------+
        | "LYS.B"    | ``ResidueId('LYS', 0, 0, 'B')``  |
        +------------+----------------------------------+
        | "WAT3:2.B" | ``ResidueId('WAT', 3, 2, 'B')``  |
        +------------+----------------------------------+
        | "WAT3:2"   | ``ResidueId('WAT', 3, 2)``       |
        +------------+----------------------------------+
        | "ARG"      | ``ResidueId('ARG', 0, 0)``       |
        +------------+----------------------------------+
        | "5.C"      | ``ResidueId('', 5, 0, 'C')``     |
        +------------+----------------------------------+
        | "42"       | ``ResidueId('', 42, 0)``         |
        +------------+----------------------------------+
        | ".D"       | ``ResidueId('', 0, 0, 'D')``     |
        +------------+----------------------------------+
        | ":2"       | ``ResidueId('', 0, 2)``          |
        +------------+----------------------------------+
        | ""         | ``ResidueId('', 0, 0)``          |
        +------------+----------------------------------+

        """
        if not (matches := _RE_RESID.search(resid_str)):
            return cls()
        g = matches.groupdict()
        return cls(
            name=g["name"],
            number=int(number) if (number := g["number"]) is not None else None,
            segment=int(segment)
            if (segment := cls._strip_prefix(g["segment"])) is not None
            else None,
            chain=cls._strip_prefix(g["chain"]),
        )

    @classmethod
    def _strip_prefix(cls, x: str | None) -> str | None:
        if x is None:
            return None
        return x[1:]


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

    .. versionchanged:: 2.1.0
        Added ``use_segid``.

    .. versionchanged:: 3.0.0
        Removed the ``use_segid`` parameter. The segment is now a separate attribute
        of :class:`~prolif.residue.ResidueId` and is always extracted.
    """

    def __init__(self, mol: "Chem.Mol") -> None:
        super().__init__(mol)
        GetSSSR(self)
        self.resid = ResidueId.from_atom(self.GetAtomWithIdx(0))

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

    def __init__(self, residues: Iterable[Residue]) -> None:
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
            self.number = np.asarray(number, dtype=np.uint32)
            self.chain = np.asarray(chain, dtype=object)
        super().__init__([(r.resid, r) for r in self._residues])
        if self.n_residues != len(self._residues):
            by_resid = Counter([r.resid for r in self._residues])
            fragmented = ", ".join(
                map(str, [resid for resid, count in by_resid.items() if count >= 2])
            )
            trigger(
                FragmentedResidueError,
                "The following residues are fragmented and ProLIF only allows "
                f"one fragment per residue identifier: {fragmented}. This is typically "
                "caused by two atoms from the same residue being too far apart "
                "and not being detected as properly bonded by MDAnalysis/RDKit. "
                "If you're using MDAnalysis, try "
                "passing a custom van der Waals radius for the responsible atoms: "
                "`<selection>.guess_bonds(vdwradii={'<element>': <radius>})`. "
                "If you're using RDKit, try calling rdkit's "
                ":func:`~rdkit.Chem.rdDetermineBonds.DetermineConnectivity` function "
                "with different options. "
                "Alternatively, add explicit bonds to your input files.",
                self,
            )

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


def ignore_self_interactions(
    res1: Residue,
    res2: Residue,
) -> bool:
    """
    Ignore interactions between the same residue

    Parameters
    ----------
    res1 : Residue
        The first residue
    res2 : Residue
        The second residue

    Returns
    -------
    bool
        ``True`` if the residues are the same and should be skipped, ``False``
        otherwise.


    .. versionadded:: 2.2.0
    """
    return res1.resid == res2.resid
