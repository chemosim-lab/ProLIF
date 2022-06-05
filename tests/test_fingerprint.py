from tempfile import NamedTemporaryFile
import pytest
from pandas import DataFrame
import numpy as np
from rdkit.DataStructs import ExplicitBitVect
from prolif.fingerprint import (Fingerprint,
                                _InteractionWrapper)
from prolif.interactions import Interaction
from prolif.residue import ResidueId
from prolif.datafiles import datapath
from prolif.molecule import sdf_supplier
from .test_base import protein_mol, ligand_mol, u, ligand_ag, protein_ag


class Dummy(Interaction):
    def detect(self, res1, res2):
        return True, 4, 2


class Return:
    def detect(self, *args):
        return args if len(args) > 1 else args[0]


def test_wrapper_return():
    detect = Dummy().detect
    mod = _InteractionWrapper(detect)
    assert detect("foo", "bar") == (True, 4, 2)
    assert mod("foo", "bar") is True
    assert mod.__wrapped__("foo", "bar") == (True, 4, 2)


@pytest.mark.parametrize("returned", [
    True,
    (True,),
    (True, 4)
])
def test_wrapper_incorrect_return(returned):
    mod = _InteractionWrapper(Return().detect)
    assert mod.__wrapped__(returned) == returned
    with pytest.raises(TypeError,
                       match="Incorrect function signature"):
        mod(returned)

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
        assert fp.dummy.__wrapped__("foo", "bar") == (True, 4, 2)

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
        assert bv[0] is np.True_

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

    def test_unknown_interaction(self):
        with pytest.raises(NameError, match="Unknown interaction"):
            Fingerprint(["Cationic", "foo"])

    @pytest.fixture
    def fp_unpkl(self, fp):
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        pkl = fp.to_pickle()
        return Fingerprint.read_pickle(pkl)

    @pytest.fixture
    def fp_unpkl_file(self, fp):
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        with NamedTemporaryFile("w+b") as tempf:
            fp.to_pickle(tempf.name)
            fp_unpkl = Fingerprint.read_pickle(tempf.name)
        return fp_unpkl

    @pytest.fixture(params=["fp_unpkl", "fp_unpkl_file"])
    def fp_pkled(self, request):
        return request.getfixturevalue(request.param)

    def test_pickle(self, fp, fp_pkled):
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        assert fp.interactions.keys() == fp_pkled.interactions.keys()
        assert len(fp.ifp) == len(fp_pkled.ifp)
        for d1, d2 in zip(fp.ifp, fp_pkled.ifp):
            d1.setdefault("Frame", None)
            assert d1.keys() == d2.keys()
            d1.pop("Frame", None)
            d2.pop("Frame")
            for (fp1, fpal1, fpap1), (fp2, fpal2, fpap2) in zip(d1.values(),
                                                                d2.values()):
                assert (fp1 == fp2).all()
                assert fpal1 == fpal2
                assert fpap1 == fpap2

    def test_pickle_custom_interaction(self, fp_unpkl):
        assert hasattr(fp_unpkl, "dummy")
        assert callable(fp_unpkl.dummy)

    def test_run_multiproc_serial_same(self, fp):
        fp.run(u.trajectory[0:100:10], ligand_ag, protein_ag,
               n_jobs=1, progress=False)
        serial = fp.to_dataframe()
        fp.run(u.trajectory[0:100:10], ligand_ag, protein_ag,
               n_jobs=None, progress=False)
        multi = fp.to_dataframe()
        assert serial.equals(multi)

    def test_run_iter_multiproc_serial_same(self, fp):
        run = fp.run_from_iterable
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = sdf_supplier(path)
        run(lig_suppl, protein_mol, n_jobs=1, progress=False)
        serial = fp.to_dataframe()
        run(lig_suppl, protein_mol, n_jobs=None, progress=False)
        multi = fp.to_dataframe()
        assert serial.equals(multi)
