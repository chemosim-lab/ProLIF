from copy import deepcopy
from math import radians

import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit import Chem

from prolif.residue import Residue, ResidueGroup, ResidueId
from prolif.utils import (
    angle_between_limits,
    get_centroid,
    get_residues_near_ligand,
    is_peptide_bond,
    pandas_series_to_bv,
    split_mol_by_residues,
    to_bitvectors,
    to_dataframe,
)


@pytest.fixture
def ifp_single():
    return {
        0: {
            ("LIG", "ALA1"): {"A": ({"indices": {"ligand": (0,), "protein": (1,)}},)},
            ("LIG", "GLU2"): {"B": ({"indices": {"ligand": (1,), "protein": (3,)}},)},
        },
        1: {
            ("LIG", "ALA1"): {
                "A": ({"indices": {"ligand": (2,), "protein": (4,)}},),
                "B": ({"indices": {"ligand": (2,), "protein": (5,)}},),
            },
            ("LIG", "ASP3"): {"B": ({"indices": {"ligand": (8,), "protein": (10,)}},)},
        },
    }


@pytest.fixture
def ifp_count():
    return {
        0: {
            ("LIG", "ALA1"): {
                "A": (
                    {"indices": {"ligand": (0,), "protein": (1,)}},
                    {"indices": {"ligand": (1,), "protein": (1,)}},
                )
            },
            ("LIG", "GLU2"): {"B": ({"indices": {"ligand": (1,), "protein": (3,)}},)},
        },
        1: {
            ("LIG", "ALA1"): {
                "A": (
                    {"indices": {"ligand": (2,), "protein": (4,)}},
                    {"indices": {"ligand": (2,), "protein": (1,)}},
                ),
                "B": ({"indices": {"ligand": (2,), "protein": (5,)}},),
            },
            ("LIG", "ASP3"): {"B": ({"indices": {"ligand": (8,), "protein": (10,)}},)},
        },
    }


@pytest.fixture(params=["ifp_single", "ifp_count"])
def ifp(request):
    return request.getfixturevalue(request.param)


def test_centroid():
    xyz = np.array(
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        dtype=np.float32,
    )
    ctd = get_centroid(xyz)
    assert ctd.shape == (3,)
    assert_equal(ctd, [1, 1, 1])


@pytest.mark.parametrize(
    "angle, mina, maxa, ring, expected",
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
def test_angle_limits(angle, mina, maxa, ring, expected):
    angle = radians(angle)
    mina = radians(mina)
    maxa = radians(maxa)
    assert angle_between_limits(angle, mina, maxa, ring) is expected


def test_pocket_residues(ligand_mol, protein_mol):
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


def test_split_residues():
    sequence = "ARNDCQEGHILKMFPSTWYV"
    prot = Chem.MolFromSequence(sequence)
    rg = ResidueGroup(
        [Residue(res) for res in Chem.SplitMolByPDBResidues(prot).values()]
    )
    residues = [Residue(mol) for mol in split_mol_by_residues(prot)]
    residues.sort(key=lambda x: x.resid)
    for molres, res in zip(residues, rg.values()):
        assert molres.resid == res.resid
        assert molres.HasSubstructMatch(res) and res.HasSubstructMatch(molres)


def test_is_peptide_bond():
    mol = Chem.RWMol()
    for i in range(3):
        a = Chem.Atom(6)
        mol.AddAtom(a)
    mol.AddBond(0, 1)
    mol.AddBond(1, 2)
    resids = {
        0: "ALA1.A",
        1: "ALA1.A",
        2: "GLU2.A",
    }
    assert is_peptide_bond(mol.GetBondWithIdx(0), resids) is False
    assert is_peptide_bond(mol.GetBondWithIdx(1), resids) is True


def test_series_to_bv():
    v = np.array([0, 1, 1, 0, 1])
    bv = pandas_series_to_bv(v)
    assert bv.GetNumBits() == len(v)
    assert bv.GetNumOnBits() == 3


def test_to_df(ifp):
    df = to_dataframe(ifp, ["A", "B", "C"])
    assert df.shape == (2, 4)
    assert df.dtypes[0].type is np.bool_
    assert df.index.name == "Frame"
    assert ("LIG", "ALA1", "A") in df.columns
    assert df[("LIG", "ALA1", "A")][0] is np.bool_(True)
    assert ("LIG", "ALA1", "B") in df.columns
    assert df[("LIG", "ALA1", "B")][0] is np.bool_(False)
    assert ("LIG", "ALA1", "C") not in df.columns
    assert ("LIG", "GLU2", "A") not in df.columns
    assert ("LIG", "ASP3", "B") in df.columns
    assert df[("LIG", "ASP3", "B")][0] is np.bool_(False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.int16,
        np.bool_,
    ],
)
def test_to_df_dtype(dtype, ifp):
    df = to_dataframe(ifp, ["A", "B", "C"], dtype=dtype)
    assert df.dtypes[0].type is dtype
    assert df[("LIG", "ALA1", "A")][0] == dtype(True)
    assert df[("LIG", "ALA1", "B")][0] == dtype(False)
    assert df[("LIG", "ASP3", "B")][0] == dtype(False)


def test_to_df_drop_empty(ifp):
    df = to_dataframe(ifp, ["A", "B", "C"], drop_empty=False)
    assert df.shape == (2, 9)


def test_to_df_no_interaction_in_first_frame(ifp_single):
    fp = deepcopy(ifp_single)
    fp[0] = {}
    to_dataframe(fp, ["A", "B", "C"])


def test_to_df_count(ifp_count):
    df = to_dataframe(ifp_count, ["A", "B", "C"], count=True)
    assert df[df > 1].any().any()
    value = df[("LIG", "ALA1", "A")][0]
    assert value.dtype == np.uint8
    assert value == 2


def test_to_df_empty_ifp():
    ifp = {0: {}, 1: {}}
    df = to_dataframe(ifp, ["A"])
    assert df.to_numpy().shape == (2, 0)


def test_to_bv(ifp):
    df = to_dataframe(ifp, ["A", "B", "C"])
    bvs = to_bitvectors(df)
    assert len(bvs) == 2
    assert bvs[0].GetNumOnBits() == 2
