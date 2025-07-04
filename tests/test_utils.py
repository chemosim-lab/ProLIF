from copy import deepcopy
from math import radians
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from rdkit import Chem

from prolif.ifp import IFP
from prolif.residue import Residue, ResidueGroup, ResidueId
from prolif.utils import (
    angle_between_limits,
    get_centroid,
    get_residues_near_ligand,
    is_peptide_bond,
    pandas_series_to_bv,
    requires,
    select_over_trajectory,
    split_mol_by_residues,
    to_bitvectors,
    to_dataframe,
)

if TYPE_CHECKING:
    from MDAnalysis import Universe

    from prolif.typeshed import IFPResults


@pytest.fixture()
def ifp_single() -> "IFPResults":
    return {
        0: IFP(
            {
                (ResidueId.from_string("LIG1"), ResidueId.from_string("ALA1")): {
                    "A": ({"indices": {"ligand": (0,), "protein": (1,)}},)
                },
                (ResidueId.from_string("LIG1"), ResidueId.from_string("GLU2")): {
                    "B": ({"indices": {"ligand": (1,), "protein": (3,)}},)
                },
            }
        ),
        1: IFP(
            {
                (ResidueId.from_string("LIG1"), ResidueId.from_string("ALA1")): {
                    "A": ({"indices": {"ligand": (2,), "protein": (4,)}},),
                    "B": ({"indices": {"ligand": (2,), "protein": (5,)}},),
                },
                (ResidueId.from_string("LIG1"), ResidueId.from_string("ASP3")): {
                    "B": ({"indices": {"ligand": (8,), "protein": (10,)}},)
                },
            }
        ),
    }


@pytest.fixture()
def ifp_count() -> "IFPResults":
    return {
        0: IFP(
            {
                (ResidueId.from_string("LIG1"), ResidueId.from_string("ALA1")): {
                    "A": (
                        {"indices": {"ligand": (0,), "protein": (1,)}},
                        {"indices": {"ligand": (1,), "protein": (1,)}},
                    ),
                },
                (ResidueId.from_string("LIG1"), ResidueId.from_string("GLU2")): {
                    "B": ({"indices": {"ligand": (1,), "protein": (3,)}},)
                },
            }
        ),
        1: IFP(
            {
                (ResidueId.from_string("LIG1"), ResidueId.from_string("ALA1")): {
                    "A": (
                        {"indices": {"ligand": (2,), "protein": (4,)}},
                        {"indices": {"ligand": (2,), "protein": (1,)}},
                    ),
                    "B": ({"indices": {"ligand": (2,), "protein": (5,)}},),
                },
                (ResidueId.from_string("LIG1"), ResidueId.from_string("ASP3")): {
                    "B": ({"indices": {"ligand": (8,), "protein": (10,)}},)
                },
            }
        ),
    }


@pytest.fixture(params=["ifp_single", "ifp_count"])
def ifp(request: pytest.FixtureRequest) -> "IFPResults":
    return cast("IFPResults", request.getfixturevalue(request.param))


def test_requires() -> None:
    @requires("this_module_does_not_exist")
    def dummy() -> None:
        pass

    with pytest.raises(ModuleNotFoundError, match=r"The module '.+' is required"):
        dummy()


def test_centroid() -> None:
    xyz = np.array(
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        dtype=np.float64,
    )
    ctd = get_centroid(xyz)
    assert ctd.shape == (3,)
    assert_equal(ctd, [1, 1, 1])


@pytest.mark.parametrize(
    ("angle", "mina", "maxa", "ring", "expected"),
    [
        (0, 0, 30, False, True),
        (30, 0, 30, False, True),
        (10, 0, 30, False, True),
        (60, 0, 30, False, False),
        (150, 0, 30, False, False),
        (150, 0, 30, True, True),
        (60, 0, 30, True, False),
        (120, 0, 30, True, False),
    ],
)
def test_angle_limits(
    angle: float, mina: float, maxa: float, ring: bool, expected: bool
) -> None:
    angle = radians(angle)
    mina = radians(mina)
    maxa = radians(maxa)
    assert angle_between_limits(angle, mina, maxa, ring=ring) is expected


def test_pocket_residues(ligand_mol: "Chem.Mol", protein_mol: "Chem.Mol") -> None:
    resids = get_residues_near_ligand(ligand_mol, protein_mol)
    residues = [
        "TYR38.A",
        "TYR40.A",
        "GLN41.A",
        "VAL102.A",
        "SER106.A",
        "TYR109.A",
        "THR110.A",
        "TRP115.A",
        "TRP125.A",
        "LEU126.A",
        "ASP129.A",
        "ILE130.A",
        "THR131.A",
        "CYS133.A",
        "THR134.A",
        "ILE137.A",
        "ILE180.A",
        "GLU198.A",
        "CYS199.A",
        "VAL200.A",
        "VAL201.A",
        "ASN202.A",
        "THR203.A",
        "TYR208.A",
        "THR209.A",
        "VAL210.A",
        "TYR211.A",
        "SER212.A",
        "THR213.A",
        "VAL214.A",
        "GLY215.A",
        "ALA216.A",
        "PHE217.A",
        "TRP327.B",
        "PHE330.B",
        "PHE331.B",
        "ILE333.B",
        "SER334.B",
        "LEU335.B",
        "MET337.B",
        "PRO338.B",
        "LEU348.B",
        "ALA349.B",
        "ILE350.B",
        "PHE351.B",
        "ASP352.B",
        "PHE353.B",
        "PHE354.B",
        "THR355.B",
        "TRP356.B",
        "GLY358.B",
        "TYR359.B",
    ]
    for res in residues:
        r = ResidueId.from_string(res)
        assert r in resids


def test_split_residues() -> None:
    sequence = "ARNDCQEGHILKMFPSTWYV"
    prot = Chem.MolFromSequence(sequence)
    rg = ResidueGroup(
        [Residue(res) for res in Chem.SplitMolByPDBResidues(prot).values()],
    )
    residues = [Residue(mol) for mol in split_mol_by_residues(prot)]
    residues.sort(key=lambda x: x.resid)
    for molres, res in zip(residues, rg.values(), strict=True):
        assert molres.resid == res.resid
        assert molres.HasSubstructMatch(res) and res.HasSubstructMatch(molres)


def test_is_peptide_bond() -> None:
    mol = Chem.RWMol()
    for _ in range(3):
        a = Chem.Atom(6)
        mol.AddAtom(a)
    mol.AddBond(0, 1)
    mol.AddBond(1, 2)
    resids = {
        0: ResidueId.from_string("ALA1.A"),
        1: ResidueId.from_string("ALA1.A"),
        2: ResidueId.from_string("GLU2.A"),
    }
    assert is_peptide_bond(mol.GetBondWithIdx(0), resids) is False
    assert is_peptide_bond(mol.GetBondWithIdx(1), resids) is True


def test_series_to_bv() -> None:
    v = pd.Series([0, 1, 1, 0, 1])
    bv = pandas_series_to_bv(v)
    assert bv.GetNumBits() == len(v)
    assert bv.GetNumOnBits() == 3


def test_to_df(ifp: "IFPResults") -> None:
    df = to_dataframe(ifp, ["A", "B", "C"])
    assert df.shape == (2, 4)
    assert df.dtypes.iloc[0].type is np.bool_
    assert df.index.name == "Frame"
    assert ("LIG1", "ALA1", "A") in df.columns
    assert df["LIG1", "ALA1", "A"][0] is np.bool_(True)
    assert ("LIG1", "ALA1", "B") in df.columns
    assert df["LIG1", "ALA1", "B"][0] is np.bool_(False)
    assert ("LIG1", "ALA1", "C") not in df.columns
    assert ("LIG1", "GLU2", "A") not in df.columns
    assert ("LIG1", "ASP3", "B") in df.columns
    assert df["LIG1", "ASP3", "B"][0] is np.bool_(False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.int16,
        np.bool_,
    ],
)
def test_to_df_dtype(dtype: type, ifp: "IFPResults") -> None:
    df = to_dataframe(ifp, ["A", "B", "C"], dtype=dtype)
    assert df.dtypes.iloc[0].type is dtype
    assert df["LIG1", "ALA1", "A"][0] == dtype(True)
    assert df["LIG1", "ALA1", "B"][0] == dtype(False)
    assert df["LIG1", "ASP3", "B"][0] == dtype(False)


def test_to_df_drop_empty(ifp: "IFPResults") -> None:
    df = to_dataframe(ifp, ["A", "B", "C"], drop_empty=False)
    assert df.shape == (2, 9)


def test_to_df_no_interaction_in_first_frame(
    ifp_single: "IFPResults",
) -> None:
    fp = deepcopy(ifp_single)
    fp[0] = IFP()
    to_dataframe(fp, ["A", "B", "C"])


def test_to_df_count(ifp_count: "IFPResults") -> None:
    df = to_dataframe(ifp_count, ["A", "B", "C"], count=True)
    assert df[df > 1].any().any()
    value = df["LIG1", "ALA1", "A"][0]
    assert value.dtype == np.uint8
    assert value == 2


def test_to_df_empty_ifp() -> None:
    ifp = {0: IFP(), 1: IFP()}
    df = to_dataframe(ifp, ["A"])
    assert df.to_numpy().shape == (2, 0)


def test_to_bv(ifp: "IFPResults") -> None:
    df = to_dataframe(ifp, ["A", "B", "C"])
    bvs = to_bitvectors(df)
    assert len(bvs) == 2
    assert bvs[0].GetNumOnBits() == 2


@pytest.mark.parametrize(
    ("selections", "expected_num_groups", "expected_num_atoms"),
    [
        (
            [
                "protein and byres around 4 group ligand",
                "resname TIP3 and byres around 4 (group ligand or group {0})",
            ],
            2,
            [314, 333],
        ),
        (
            [
                "protein and byres around 4 group ligand",
                "resname TIP3 and byres around 4 (group ligand or group {0})",
                "protein and byres around 4 (group {1})",
            ],
            3,
            [314, 333, 1020],
        ),
    ],
)
def test_select_over_trajectory(
    water_u: "Universe",
    selections: list[str],
    expected_num_groups: int,
    expected_num_atoms: list[int],
) -> None:
    ligand_selection = water_u.select_atoms("resname QNB")
    atomgroups = select_over_trajectory(
        water_u, water_u.trajectory[:5], *selections, ligand=ligand_selection
    )
    assert len(atomgroups) == expected_num_groups
    for group, expected in zip(atomgroups, expected_num_atoms, strict=True):
        assert group.n_atoms == expected


def test_select_over_trajectory_single(water_u: "Universe") -> None:
    ligand_selection = water_u.select_atoms("resname QNB")
    atomgroup = select_over_trajectory(
        water_u,
        water_u.trajectory[:5],
        "protein and byres around 4 group ligand",
        ligand=ligand_selection,
    )
    # also indirectly tests for typing
    assert atomgroup.n_atoms == 314
