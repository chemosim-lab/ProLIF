from math import radians
import pytest
from rdkit import Chem
from MDAnalysis import Universe
import numpy as np
from numpy.testing import assert_equal
from prolif.residue import ResidueId, Residue, ResidueGroup
from prolif.molecule import Molecule
from prolif.utils import (get_centroid,
                          angle_between_limits,
                          get_residues_near_ligand,
                          split_mol_by_residues,
                          is_peptide_bond,
                          _series_to_bv,
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
    residues = ["TYR38.0",  "TYR40.0",  "GLN41.0",  "VAL102.0", "SER106.0",
                "TYR109.0", "THR110.0", "TRP115.0", "TRP125.0", "LEU126.0",
                "ASP129.0", "ILE130.0", "THR131.0", "CYS133.0", "THR134.0",
                "ILE137.0", "ILE180.0", "GLU198.0", "CYS199.0", "VAL200.0",
                "VAL201.0", "ASN202.0", "THR203.0", "TYR208.0", "THR209.0",
                "VAL210.0", "TYR211.0", "SER212.0", "THR213.0", "VAL214.0",
                "GLY215.0", "ALA216.0", "PHE217.0", "TRP327.1", "PHE330.1",
                "PHE331.1", "ILE333.1", "SER334.1", "LEU335.1", "MET337.1",
                "PRO338.1", "LEU348.1", "ALA349.1", "ILE350.1", "PHE351.1",
                "ASP352.1", "PHE353.1", "PHE354.1", "THR355.1", "TRP356.1",
                "GLY358.1", "TYR359.1"]
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
    bv = _series_to_bv(v)
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
        bvs = to_bitvectors(ifp, fp)
