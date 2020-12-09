import pytest
from pandas import DataFrame
import numpy as np
from rdkit.DataStructs import ExplicitBitVect
from prolif.fingerprint import (Fingerprint,
                                _return_first_element)
from prolif.interactions import Interaction
from prolif.residue import ResidueId
from .test_base import protein_mol, ligand_mol, u, ligand_ag, protein_ag


class Dummy(Interaction):
    def detect(self, res1, res2):
        return 1, 2, 3


def test_wrapper_return():
    foo = Dummy().detect
    bar = _return_first_element(foo)
    assert foo("foo", "bar") == (1, 2, 3)
    assert bar("foo", "bar") == 1
    assert bar.__wrapped__("foo", "bar") == (1, 2, 3)


class TestFingerprint:
    @pytest.fixture
    def fp(self):
        return Fingerprint()

    @pytest.fixture(scope="class")
    def fp_class(self):
        return Fingerprint(["Hydrophobic"])

    def test_init(self, fp):
        assert "HBDonor" in fp.interactions.keys()
        assert hasattr(fp, "hbdonor") and callable(fp.hbdonor)
        assert "Dummy" not in fp.interactions.keys()
        assert hasattr(fp, "dummy") and callable(fp.dummy)
        assert "_BaseHBond" not in fp.interactions.keys()
        assert not hasattr(fp, "_basehbond")
        assert "Interaction" not in fp.interactions.keys()
        assert not hasattr(fp, "interaction")

    def test_init_all(self):
        fp = Fingerprint("all")
        for name, func in fp.interactions.items():
            assert getattr(fp, name.lower()) is func

    def test_n_interactions(self, fp):
        assert fp.n_interactions == len(fp.interactions)

    def test_wrapped(self, fp):
        assert fp.dummy("foo", "bar") == 1
        assert fp.dummy.__wrapped__("foo", "bar") == (1, 2, 3)

    def test_bitvector(self, fp):
        bv = fp.bitvector(ligand_mol, protein_mol["ASP129.A"])
        assert len(bv) == fp.n_interactions
        assert bv.sum() > 0

    def test_bitvector_atoms(self, fp):
        bv, atoms = fp.bitvector_atoms(ligand_mol, protein_mol["ASP129.A"])
        assert len(bv) == fp.n_interactions
        assert len(atoms) == fp.n_interactions
        assert bv.sum() > 0
        ids = np.where(bv == 1)[0]
        assert atoms[ids[0]] != (None, None)

    def test_run(self, fp_class):
        fp_class.run(u.trajectory[0:1], ligand_ag, protein_ag,
                     residues="all", progress=False)
        assert hasattr(fp_class, "ifp")
        assert len(fp_class.ifp) == 1
        res = ResidueId.from_string("LYS387.B")
        assert res in fp_class.ifp[0].keys()
        fp_class.run(u.trajectory[1:2], ligand_ag, protein_ag,
                     residues=["ASP129.A"], progress=False)
        assert hasattr(fp_class, "ifp")
        assert len(fp_class.ifp) == 1
        res = ResidueId.from_string("ASP129.A")
        assert res in fp_class.ifp[0].keys()
        fp_class.run(u.trajectory[:3], ligand_ag, protein_ag,
                     residues=None, progress=False)
        assert hasattr(fp_class, "ifp")
        assert len(fp_class.ifp) == 3
        assert len(fp_class.ifp[0]) > 1
        res = ResidueId.from_string("VAL201.A")
        assert res in fp_class.ifp[0].keys()
        u.trajectory[0]

    def test_to_df(self, fp, fp_class):
        # depends on successfull test_run
        with pytest.raises(AttributeError, match="use the run method"):
            fp.to_dataframe()
        df = fp_class.to_dataframe()
        assert isinstance(df, DataFrame)
        assert len(df) == 3

    def test_to_bv(self, fp, fp_class):
        # depends on successfull test_run
        with pytest.raises(AttributeError, match="use the run method"):
            fp.to_bitvectors()
        bvs = fp_class.to_bitvectors()
        assert isinstance(bvs[0], ExplicitBitVect)
        assert len(bvs) == 3
