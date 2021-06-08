import pytest
from pandas import DataFrame
import numpy as np
from rdkit.DataStructs import ExplicitBitVect
from prolif.fingerprint import (Fingerprint,
                                _return_first_element)
from prolif.interactions import Interaction
from prolif.residue import ResidueId
from prolif.datafiles import datapath
from prolif.molecule import sdf_supplier
from .test_base import protein_mol, ligand_mol, u, ligand_ag, protein_ag


class Dummy(Interaction):
    def detect(self, res1, res2):
        return 1, 2, 3


def func_return_single_val():
    return 0


def test_wrapper_return():
    foo = Dummy().detect
    bar = _return_first_element(foo)
    assert foo("foo", "bar") == (1, 2, 3)
    assert bar("foo", "bar") == 1
    assert bar.__wrapped__("foo", "bar") == (1, 2, 3)
    baz = _return_first_element(func_return_single_val)
    assert baz() == 0
    assert baz.__wrapped__() == 0


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
        bv, lig_ix, prot_ix = fp.bitvector_atoms(ligand_mol,
                                                 protein_mol["ASP129.A"])
        assert len(bv) == fp.n_interactions
        assert len(lig_ix) == fp.n_interactions
        assert len(prot_ix) == fp.n_interactions
        assert bv.sum() > 0
        ids = np.where(bv == 1)[0]
        assert (lig_ix[ids[0]] is not None and prot_ix[ids[0]] is not None)

    def test_run_residues(self, fp_class):
        fp_class.run(u.trajectory[0:1], ligand_ag, protein_ag,
                     residues="all", progress=False)
        lig_id = ResidueId.from_string("LIG1.G")
        assert hasattr(fp_class, "ifp")
        assert len(fp_class.ifp) == 1
        res = ResidueId.from_string("LYS387.B")
        assert (lig_id, res) in fp_class.ifp[0].keys()
        fp_class.run(u.trajectory[1:2], ligand_ag, protein_ag,
                     residues=["ASP129.A"], progress=False)
        assert hasattr(fp_class, "ifp")
        assert len(fp_class.ifp) == 1
        res = ResidueId.from_string("ASP129.A")
        assert (lig_id, res) in fp_class.ifp[0].keys()
        fp_class.run(u.trajectory[:3], ligand_ag, protein_ag,
                     residues=None, progress=False)
        assert hasattr(fp_class, "ifp")
        assert len(fp_class.ifp) == 3
        assert len(fp_class.ifp[0]) > 1
        res = ResidueId.from_string("VAL201.A")
        assert (lig_id, res) in fp_class.ifp[0].keys()
        u.trajectory[0]

    def test_generate(self, fp_class):
        ifp = fp_class.generate(ligand_mol, protein_mol)
        key = (ResidueId("LIG", 1, "G"), ResidueId("THR", 355, "B"))
        bv = ifp[key]
        assert isinstance(bv, np.ndarray)
        assert bv[0] == True

    def test_run(self, fp_class):
        fp_class.run(u.trajectory[0:1], ligand_ag, protein_ag,
                     residues=None, progress=False)
        assert hasattr(fp_class, "ifp")
        ifp = fp_class.ifp[0]
        ifp.pop("Frame")
        data = list(ifp.values())[0]
        assert isinstance(data[0], np.ndarray)
        assert isinstance(data[1], list)
        assert isinstance(data[2], list)
    
    def test_run_from_iterable(self, fp):
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        assert len(fp.ifp) == 2

    def test_to_df(self, fp, fp_class):
        with pytest.raises(AttributeError, match="use the `run` method"):
            fp.to_dataframe()
        fp_class.run(u.trajectory[:3], ligand_ag, protein_ag,
                     residues=None, progress=False)
        df = fp_class.to_dataframe()
        assert isinstance(df, DataFrame)
        assert len(df) == 3

    def test_to_df_kwargs(self, fp_class):
        fp_class.run(u.trajectory[:3], ligand_ag, protein_ag,
                     residues=None, progress=False)
        df = fp_class.to_dataframe(dtype=np.uint8)
        assert df.dtypes[0].type is np.uint8
        df = fp_class.to_dataframe(drop_empty=False)
        resids = set([key for d in fp_class.ifp for key in d.keys()
                  if key != "Frame"])
        assert df.shape == (3, len(resids))

    def test_to_bv(self, fp, fp_class):
        with pytest.raises(AttributeError, match="use the `run` method"):
            fp.to_bitvectors()
        fp_class.run(u.trajectory[:3], ligand_ag, protein_ag,
                     residues=None, progress=False)
        bvs = fp_class.to_bitvectors()
        assert isinstance(bvs[0], ExplicitBitVect)
        assert len(bvs) == 3

    def test_list_avail(self):
        avail = Fingerprint.list_available()
        assert "Hydrophobic" in avail
        assert "HBDonor" in avail
        assert "_BaseHBond" not in avail
        avail = Fingerprint.list_available(show_hidden=True)
        assert "Hydrophobic" in avail
        assert "HBDonor" in avail
        assert "_BaseHBond" in avail
        assert "_Distance" in avail
        assert "Interaction" in avail
