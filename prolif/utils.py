"""
Helper functions --- :mod:`prolif.utils`
========================================
"""

import warnings
from collections import defaultdict
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from importlib.util import find_spec
from math import pi
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

import numpy as np
import pandas as pd
from MDAnalysis import AtomGroup, Universe
from MDAnalysis.coordinates.timestep import Timestep
from rdkit import Chem, rdBase
from rdkit.Chem import FragmentOnBonds, GetMolFrags, SplitMolByPDBResidues
from rdkit.DataStructs import ExplicitBitVect, UIntSparseIntVect
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from prolif.residue import ResidueId

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prolif.rdkitmol import BaseRDKitMol
    from prolif.typeshed import IFPResults, Trajectory

_90_deg_to_rad = pi / 2

P = ParamSpec("P")
R = TypeVar("R")


def requires(
    module: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def inner(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if find_spec(module):
                return func(*args, **kwargs)
            raise ModuleNotFoundError(
                f"The module {module!r} is required to use {func.__name__!r} "
                "but it is not installed!",
            )

        return wrapper

    return inner


@contextmanager
def catch_rdkit_logs() -> Iterator[None]:
    log_status = rdBase.LogStatus()
    rdBase.DisableLog("rdApp.*")
    yield
    log_status = {st.split(":")[0]: st.split(":")[1] for st in log_status.split("\n")}
    log_status = {k: v == "enabled" for k, v in log_status.items()}
    for k, v in log_status.items():
        if v is True:
            rdBase.EnableLog(k)
        else:
            rdBase.DisableLog(k)


@contextmanager
def catch_warning(**kwargs: Any) -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", **kwargs)
        yield


def get_centroid(
    coordinates: "NDArray[np.float64]",
) -> "NDArray[np.float64]":
    """Centroid for an array of XYZ coordinates"""
    return np.mean(coordinates, axis=0)  # type: ignore[no-any-return]


def get_ring_normal_vector(
    centroid: Point3D, coordinates: "NDArray[np.float64]"
) -> Point3D:
    """Returns a vector that is normal to the ring plane"""
    # A & B are two edges of the ring
    a = Point3D(*coordinates[0])
    b = Point3D(*coordinates[1])
    # vectors between centroid and these edges
    ca = centroid.DirectionVector(a)
    cb = centroid.DirectionVector(b)
    # cross product between these two vectors
    # cb.CrossProduct(ca) is the normal vector in the opposite direction
    return ca.CrossProduct(cb)


def angle_between_limits(
    angle: float, min_angle: float, max_angle: float, *, ring: bool = False
) -> bool:
    """Checks if an angle value is between min and max angles in radian

    Parameters
    ----------
    angle : float
        Angle to check, in radians
    min_angle : float
        Lower bound angle, in radians
    max_angle : float
        Upper bound angle, in radians
    ring : bool
        Whether the angle being checked involves a ring or not

    Notes
    -----
    When ``ring=True``, the angle is capped between 0 and 90, and so should be
    the min and max angles. This is useful for angles involving a ring's plane
    normal vector.
    """
    if ring:
        if angle >= pi:
            angle %= _90_deg_to_rad
        elif angle > _90_deg_to_rad:
            angle = _90_deg_to_rad - (angle % _90_deg_to_rad)
    return min_angle <= angle <= max_angle


def get_residues_near_ligand(
    lig: "BaseRDKitMol",
    prot: "BaseRDKitMol",
    cutoff: float = 6.0,
    use_segid: bool = False,
) -> list[ResidueId]:
    """Detects residues close to a reference ligand

    Parameters
    ----------
    lig : prolif.molecule.Molecule or prolif.residue.Residue
        Select residues that are near this ligand/residue
    prot : prolif.molecule.Molecule
        Protein containing the residues
    cutoff : float
        If any interatomic distance between the ligand reference points and a
        residue is below or equal to this cutoff, the residue will be selected
    use_segid: bool, default = False
        Use the segment number rather than the chain identifier as a chain.

    Returns
    -------
    residues : list
        A list of unique :class:`~prolif.residue.ResidueId` that are close to
        the ligand

    .. versionchanged:: 2.1.0
        Added `use_segid`.
    """
    tree = cKDTree(prot.xyz)
    ix: Sequence[list[int]] = tree.query_ball_point(  # type: ignore[assignment]
        lig.xyz, cutoff
    )
    ix = {i for lst in ix for i in lst}
    resids = [
        ResidueId.from_atom(prot.GetAtomWithIdx(i), use_segid=use_segid) for i in ix
    ]
    return list(set(resids))


def split_mol_by_residues(mol: Chem.Mol) -> list[Chem.Mol]:
    """Splits a molecule in multiple fragments based on residues

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to fragment

    Returns
    -------
    residues : list
        A list of :class:`rdkit.Chem.rdchem.Mol`

    Notes
    -----
    Code adapted from Maciek Wójcikowski on the RDKit discussion list
    """
    residues: list[Chem.Mol] = []
    for res in SplitMolByPDBResidues(mol).values():
        for frag in GetMolFrags(res, asMols=True, sanitizeFrags=False):
            # count number of unique residues in the fragment
            resids: dict[int, ResidueId] = {
                a.GetIdx(): ResidueId.from_atom(a) for a in frag.GetAtoms()
            }
            if len(set(resids.values())) > 1:
                # split on peptide bonds
                bonds = [
                    b.GetIdx() for b in frag.GetBonds() if is_peptide_bond(b, resids)
                ]
                mols = FragmentOnBonds(frag, bonds, addDummies=False)
                mols = GetMolFrags(mols, asMols=True, sanitizeFrags=False)
                residues.extend(mols)
            else:
                residues.append(frag)
    return residues


def is_peptide_bond(bond: Chem.Bond, resids: dict[int, ResidueId]) -> bool:
    """Checks if a bond is a peptide bond based on the ResidueId of the atoms
    on each part of the bond. Also works for disulfide bridges or any bond that
    links two residues in biopolymers.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        The bond to check
    resids : dict
        A dictionnary of ResidueId indexed by atom index
    """
    return resids[bond.GetBeginAtomIdx()] != resids[bond.GetEndAtomIdx()]


def to_dataframe(
    ifp: "IFPResults",
    interactions: Collection[str],
    count: bool = False,
    dtype: type | None = None,
    drop_empty: bool = True,
    index_col: str = "Frame",
) -> pd.DataFrame:
    """Converts IFPs to a pandas DataFrame

    Parameters
    ----------
    ifp : dict
        A dict in the format
        ``{<frame number>: {(<residue_id>, <residue_id>): <interactions>}}``.
        ``<interactions>`` is either a :class:`numpy.ndarray` bitvector, or a tuple of
        dict in the format ``{<interaction name>: <metadata dict>}``.
    interactions : list
        A list of interactions, in the same order as used to detect the interactions.
    count : bool
        Whether to output a count fingerprint or not.
    dtype : object or None
        Cast the dataframe values to this type. If ``None``, uses ``np.uint8`` if
        ``count=True``, else ``bool``.
    drop_empty : bool
        Drop columns with only empty values
    index_col : str
        Name of the index column in the DataFrame

    Returns
    -------
    df : pandas.DataFrame
        A 3-levels DataFrame where each ligand residue, protein residue, and
        interaction type are in separate columns

    Example
    -------
    ::

        >>> df = prolif.to_dataframe(results, fp.interactions.keys(), dtype=int)
        >>> print(df)
        ligand             LIG1.G
        protein             ILE59                  ILE55       TYR93
        interaction   Hydrophobic HBAcceptor Hydrophobic Hydrophobic PiStacking
        Frame
        0                       0          1           0           0          0
        ...

    .. versionchanged:: 0.3.2
        Moved the ``return_atoms`` parameter from the ``run`` methods to the
        dataframe conversion code

    .. versionchanged:: 2.0.0
        Removed the ``return_atoms`` parameter. Added the ``count`` parameter. Removed
        support for ``ifp`` containing ``np.ndarray`` bitvectors.
    """
    ifp = deepcopy(ifp)
    n_interactions = len(interactions)
    if dtype is None:
        dtype = np.uint8 if count else bool
    empty_value = dtype(0)
    # create empty array for each residue pair interaction that doesn't exist
    # in a particular frame
    empty_arr: NDArray = np.array(
        [empty_value for _ in range(n_interactions)], dtype=dtype
    )
    # residue pairs
    residue_pairs = sorted(
        {residue_tuple for frame_ifp in ifp.values() for residue_tuple in frame_ifp},
    )
    # sparse to dense
    data = defaultdict(list)
    index = []
    for i, frame_ifp in ifp.items():
        index.append(i)
        for residue_tuple in residue_pairs:
            try:
                ifp_dict = frame_ifp[residue_tuple]
            except KeyError:
                data[residue_tuple].append(empty_arr[:])
            else:
                if count:
                    bitvector: NDArray = np.array(
                        [len(ifp_dict.get(i, ())) for i in interactions],
                        dtype=dtype,
                    )
                else:
                    bitvector = np.array(
                        [i in ifp_dict for i in interactions],
                        dtype=bool,
                    )
                data[residue_tuple].append(bitvector)
    index = pd.Series(index, name=index_col)
    # create dataframe
    if not data:
        warnings.warn("No interaction detected", stacklevel=2)
        return pd.DataFrame([], index=index)
    values = np.array(
        [
            np.hstack([bitvector_list[frame] for bitvector_list in data.values()])
            for frame in range(len(index))
        ],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            (str(lig_res), str(prot_res), i)
            for lig_res, prot_res in residue_pairs
            for i in interactions
        ],
        names=["ligand", "protein", "interaction"],
    )
    df = pd.DataFrame(values, columns=columns, index=index)
    df = df.astype(dtype)
    if drop_empty:
        mask = (df != empty_value).any(axis=0)
        df = df.loc[:, mask]
    return df


def pandas_series_to_bv(s: pd.Series) -> ExplicitBitVect:
    bv = ExplicitBitVect(len(s))
    on_bits = np.where(s >= True)[0].tolist()
    bv.SetBitsFromList(on_bits)
    return bv


def to_bitvectors(df: pd.DataFrame) -> list[ExplicitBitVect]:
    """Converts an interaction DataFrame to a list of RDKit ExplicitBitVector

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where each column corresponds to an interaction between two
        residues

    Returns
    -------
    bv : list
        A list of :class:`~rdkit.DataStructs.cDataStructs.ExplicitBitVect`
        for each frame

    Example
    -------
    ::

        >>> from rdkit.DataStructs import TanimotoSimilarity
        >>> bv = prolif.to_bitvectors(df)
        >>> TanimotoSimilarity(bv[0], bv[1])
        0.42

    """
    return df.apply(pandas_series_to_bv, axis=1).tolist()  # type: ignore[no-any-return]


def pandas_series_to_countvector(s: pd.Series) -> UIntSparseIntVect:
    size = len(s)
    cv = UIntSparseIntVect(size)
    for i in range(size):
        cv[i] = int(s.iloc[i])
    return cv


def to_countvectors(df: pd.DataFrame) -> list[UIntSparseIntVect]:
    """Converts an interaction DataFrame to a list of RDKit UIntSparseIntVect

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where each column corresponds to the count for an interaction
        between two residues

    Returns
    -------
    cv : list
        A list of :class:`~rdkit.DataStructs.cDataStructs.UIntSparseIntVect`
        for each frame
    """
    return df.apply(  # type: ignore[no-any-return]
        pandas_series_to_countvector, axis=1
    ).tolist()


@overload
def select_over_trajectory(  # type: ignore[no-any-unimported]
    u: Universe, trajectory: "Trajectory", selections: str, **kwargs: Any
) -> AtomGroup: ...
@overload
def select_over_trajectory(  # type: ignore[no-any-unimported]
    u: Universe, trajectory: "Trajectory", *selections: str, **kwargs: Any
) -> list[AtomGroup]: ...
def select_over_trajectory(  # type: ignore[no-any-unimported]
    u: Universe, trajectory: "Trajectory", *selections: str, **kwargs: Any
) -> AtomGroup | list[AtomGroup]:
    """Returns AtomGroup(s) that satisfy each distance-based ``selection`` over the
    entire trajectory rather than only the first frame. Only useful for distance-based
    selections.
    Use ``group {0}`` in any additional selection to refer to the AtomGroup generated
    by the first selection, or ``group {N-1}`` for the Nth selection.
    """
    ix: list[set[int]] = [set() for _ in selections]
    groups: dict[str, AtomGroup] = {  # type: ignore[no-any-unimported]
        f"group{i}": Universe.empty(0).atoms for i, _ in enumerate(selections)
    }
    trajectory = (trajectory,) if isinstance(trajectory, Timestep) else trajectory
    for _ in tqdm(trajectory):
        for i, sel in enumerate(selections):
            groups[f"group{i}"] = group = u.select_atoms(
                sel.format(*groups), **{**kwargs, **groups}
            )
            ix[i].update(group.ix)
    atomgroups = [AtomGroup(sorted(indices), u) for indices in ix]
    return atomgroups[0] if len(atomgroups) == 1 else atomgroups
