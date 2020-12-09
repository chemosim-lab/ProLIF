from math import radians
import pytest
from rdkit import Chem
import numpy as np
from numpy.testing import assert_equal
from prolif.residue import ResidueId, Residue, ResidueGroup
from prolif.utils import (get_centroid,
                          angle_between_limits,
                          get_residues_near_ligand,
                          split_mol_by_residues,
                          is_peptide_bond,
                          pandas_series_to_bv,
                          to_dataframe,
                          to_bitvectors)
from .test_base import ligand_mol, protein_mol


def test_centroid():
    xyz = np.array([(0,0,0),
                    (0,0,0),
                    (0,0,0),
                    (2,2,2),
                    (2,2,2),
                    (2,2,2)],
                   dtype=np.float32)
    ctd = get_centroid(xyz)
    assert ctd.shape == (3,)
    assert_equal(ctd, [1, 1, 1])


@pytest.mark.parametrize("angle, mina, maxa, ring, expected", [
    (0, 0, 30, False, True),
    (30, 0, 30, False, True),
    (10, 0, 30, False, True),
    (60, 0, 30, False, False),
    (150, 0, 30, False, False),
    (150, 0, 30, True, True),
    (60, 0, 30, True, False),
    (120, 0, 30, True, False),
])
def test_angle_limits(angle, mina, maxa, ring, expected):
    angle = radians(angle)
    mina = radians(mina)
    maxa = radians(maxa)
    assert angle_between_limits(angle, mina, maxa, ring) is expected


def test_pocket_residues():
    resids = get_residues_near_ligand(ligand_mol, protein_mol)
    residues = ["TYR38.A",  "TYR40.A",  "GLN41.A",  "VAL102.A", "SER106.A",
                "TYR109.A", "THR110.A", "TRP115.A", "TRP125.A", "LEU126.A",
                "ASP129.A", "ILE130.A", "THR131.A", "CYS133.A", "THR134.A",
                "ILE137.A", "ILE180.A", "GLU198.A", "CYS199.A", "VAL200.A",
                "VAL201.A", "ASN202.A", "THR203.A", "TYR208.A", "THR209.A",
                "VAL210.A", "TYR211.A", "SER212.A", "THR213.A", "VAL214.A",
                "GLY215.A", "ALA216.A", "PHE217.A", "TRP327.B", "PHE330.B",
                "PHE331.B", "ILE333.B", "SER334.B", "LEU335.B", "MET337.B",
                "PRO338.B", "LEU348.B", "ALA349.B", "ILE350.B", "PHE351.B",
                "ASP352.B", "PHE353.B", "PHE354.B", "THR355.B", "TRP356.B",
                "GLY358.B", "TYR359.B"]
    for res in residues:
        r = ResidueId.from_string(res)
        assert r in resids


def test_split_residues():
    sequence = "ARNDCQEGHILKMFPSTWYV"
    prot = Chem.MolFromSequence(sequence)
    rg = ResidueGroup([Residue(res)
                       for res in Chem.SplitMolByPDBResidues(prot).values()])
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
    v = np.array([0,1,1,0,1])
    bv = pandas_series_to_bv(v)
    assert bv.GetNumBits() == len(v)
    assert bv.GetNumOnBits() == 3


class DummyFp:
    interactions = ["A", "B", "C"]
    n_interactions = 3

ifp = [{"Frame": 0,
        "foo": "bar",
        "ALA1": np.array([True, False, False]),
        "GLU2": np.array([False, True, False])},
       {"Frame": 1,
        "foo": "bar",
        "ALA1": np.array([True, True, False]),
        "ASP3": np.array([False, True, False])}]


def test_to_df():
    fp = DummyFp()
    df = to_dataframe(ifp, fp)
    assert df.shape == (2, 6)
    assert ("Frame", "") in df.columns
    assert ("foo", "") in df.columns
    assert ("ALA1", "A") in df.columns
    assert ("ALA1", "B") in df.columns
    assert ("ALA1", "C") not in df.columns
    assert ("GLU2", "A") not in df.columns
    assert ("ASP3", "B") in df.columns


def test_to_bv():
    fp = DummyFp()
    bvs = to_bitvectors(ifp, fp)
    assert len(bvs) == 2
    assert bvs[0].GetNumOnBits() == 2


def test_to_bv_raise_no_bits():
    fp = DummyFp()
    ifp = [{"Frame": 0,
            "foo": "bar",
            "ALA1": np.array([False, False, False]),
            "GLU2": np.array([False, False, False])}]
    with pytest.raises(ValueError, match="input IFP only contains off bits"):
        to_bitvectors(ifp, fp)
