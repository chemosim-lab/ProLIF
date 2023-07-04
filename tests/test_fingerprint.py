import MDAnalysis as mda
import numpy as np
import pytest
from pandas import DataFrame
from rdkit.DataStructs import ExplicitBitVect, UIntSparseIntVect

from prolif.datafiles import datapath
from prolif.fingerprint import Fingerprint
from prolif.interactions.base import _INTERACTIONS, Interaction
from prolif.molecule import sdf_supplier
from prolif.residue import ResidueId


class Dummy(Interaction):
    def detect(self, res1, res2):
        yield self.metadata(res1, res2, (2,), (4,), distance=4.2)


def test_interaction_base(sdf_suppl):
    interaction = Dummy()
    _repr = repr(interaction)
    assert _repr.startswith("<") and ".Dummy at " in _repr
    assert callable(interaction)
    mol = sdf_suppl[0]
    metadata = next(interaction.detect(mol, mol))
    assert metadata["indices"] == {"ligand": (2,), "protein": (4,)}
    assert metadata["parent_indices"] == {"ligand": (2,), "protein": (4,)}
    # invert
    inverted = Dummy.invert_role("Dummy", "inverted")
    inverted_interaction = inverted()
    metadata = next(inverted_interaction.detect(mol, mol))
    assert metadata["indices"] == {"ligand": (4,), "protein": (2,)}
    assert metadata["parent_indices"] == {"ligand": (4,), "protein": (2,)}


class TestFingerprint:
    @pytest.fixture(scope="class")
    def fp(self):
        yield Fingerprint()
        _INTERACTIONS.pop("Dummy", None)

    @pytest.fixture(scope="class")
    def fp_count(self):
        yield Fingerprint(count=True)
        _INTERACTIONS.pop("Dummy", None)

    @pytest.fixture(scope="class", params=["fp", "fp_count"])
    def any_fp(self, request):
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="class")
    def fp_simple(self):
        yield Fingerprint(["Hydrophobic"])
        _INTERACTIONS.pop("Dummy", None)

    def test_init(self, fp_simple):
        assert "Hydrophobic" in fp_simple.interactions.keys()
        assert hasattr(fp_simple, "hydrophobic") and callable(fp_simple.hydrophobic)
        assert "Dummy" not in fp_simple.interactions.keys()
        assert hasattr(fp_simple, "dummy") and callable(fp_simple.dummy)
        assert "_BaseHBond" not in fp_simple.interactions.keys()
        assert not hasattr(fp_simple, "_basehbond")
        assert "Interaction" not in fp_simple.interactions.keys()
        assert not hasattr(fp_simple, "interaction")

    def test_init_all(self):
        fp = Fingerprint("all")
        for name, func in fp.interactions.items():
            assert getattr(fp, name.lower()) is func.__wrapped__

    def test_n_interactions(self, fp):
        assert fp.n_interactions == len(fp.interactions)

    def test_bitvector(self, any_fp, ligand_mol, protein_mol):
        bv = any_fp.bitvector(ligand_mol[0], protein_mol["ASP129.A"])
        assert len(bv) == any_fp.n_interactions
        assert (bv == 0).any()
        assert bv.sum() > 0
        if any_fp.count:
            assert (bv > 1).any()

    def test_metadata(self, any_fp, ligand_mol, protein_mol):
        metadata = any_fp.metadata(ligand_mol[0], protein_mol["ASP129.A"])
        assert metadata
        assert "HBAcceptor" not in metadata
        assert all(
            isinstance(cationic["indices"]["ligand"], tuple)
            for cationic in metadata["Cationic"]
        )

    def test_run_residues(self, fp_simple, u, ligand_ag, protein_ag):
        fp_simple.run(
            u.trajectory[0:1],
            ligand_ag,
            protein_ag,
            residues=["TYR109.A"],
            progress=False,
        )
        lig_id = ResidueId.from_string("LIG1.G")
        assert hasattr(fp_simple, "ifp")
        assert len(fp_simple.ifp) == 1
        res = ResidueId.from_string("TYR109.A")
        assert (lig_id, res) in fp_simple.ifp[0].keys()
        fp_simple.run(
            u.trajectory[1:2],
            ligand_ag,
            protein_ag,
            residues="all",
            progress=False,
        )
        assert hasattr(fp_simple, "ifp")
        assert len(fp_simple.ifp) == 1
        res = ResidueId.from_string("TRP125.A")
        assert (lig_id, res) in fp_simple.ifp[1].keys()
        fp_simple.run(
            u.trajectory[:3], ligand_ag, protein_ag, residues=None, progress=False
        )
        assert hasattr(fp_simple, "ifp")
        assert len(fp_simple.ifp) == 3
        assert len(fp_simple.ifp[0]) > 1
        res = ResidueId.from_string("ALA216.A")
        assert (lig_id, res) in fp_simple.ifp[0].keys()
        u.trajectory[0]

    def test_generate(self, fp_simple, ligand_mol, protein_mol):
        ifp = fp_simple.generate(ligand_mol, protein_mol)
        key = (ResidueId("LIG", 1, "G"), ResidueId("VAL", 201, "A"))
        bv = ifp[key]
        assert isinstance(bv, np.ndarray)
        assert bv[0] is np.True_

    def test_generate_metadata(self, fp_simple, ligand_mol, protein_mol):
        ifp = fp_simple.generate(ligand_mol, protein_mol, metadata=True)
        key = (ResidueId("LIG", 1, "G"), ResidueId("VAL", 201, "A"))
        int_data = ifp[key]
        assert "Hydrophobic" in int_data

    def test_run(self, fp_simple, u, ligand_ag, protein_ag):
        fp_simple.run(
            u.trajectory[0:1], ligand_ag, protein_ag, residues=None, progress=False
        )
        assert hasattr(fp_simple, "ifp")
        ifp = fp_simple.ifp[0]
        interactions = next(iter(ifp.values()))
        assert isinstance(interactions, dict)
        metadata_tuple = next(iter(interactions.values()))
        assert all(
            [
                key in metadata_tuple[0]
                for key in ["indices", "parent_indices", "distance"]
            ]
        )

    def test_run_from_iterable(self, fp_simple, protein_mol):
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp_simple.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        assert len(fp_simple.ifp) == 2

    def test_to_df(self, fp_simple, u, ligand_ag, protein_ag):
        with pytest.raises(AttributeError, match="use the `run` method"):
            Fingerprint().to_dataframe()
        fp_simple.run(
            u.trajectory[:3], ligand_ag, protein_ag, residues=None, progress=False
        )
        df = fp_simple.to_dataframe()
        assert isinstance(df, DataFrame)
        assert len(df) == 3

    def test_to_df_kwargs(self, fp_simple, u, ligand_ag, protein_ag):
        fp_simple.run(
            u.trajectory[:3], ligand_ag, protein_ag, residues=None, progress=False
        )
        df = fp_simple.to_dataframe(dtype=np.uint8)
        assert df.dtypes[0].type is np.uint8
        df = fp_simple.to_dataframe(drop_empty=False)
        resids = set([key for d in fp_simple.ifp.values() for key in d.keys()])
        assert df.shape == (3, len(resids))

    def test_to_bitvector(self, fp_simple, u, ligand_ag, protein_ag):
        with pytest.raises(AttributeError, match="use the `run` method"):
            Fingerprint().to_bitvectors()
        fp_simple.run(
            u.trajectory[:3], ligand_ag, protein_ag, residues=None, progress=False
        )
        bvs = fp_simple.to_bitvectors()
        assert isinstance(bvs[0], ExplicitBitVect)
        assert len(bvs) == 3

    def test_to_countvectors(self, fp_count, u, ligand_ag, protein_ag):
        with pytest.raises(AttributeError, match="use the `run` method"):
            Fingerprint().to_countvectors()
        fp_count.run(
            u.trajectory[:3], ligand_ag, protein_ag, residues=None, progress=False
        )
        cvs = fp_count.to_countvectors()
        assert isinstance(cvs[0], UIntSparseIntVect)
        assert cvs[0][0] == 4
        assert len(cvs) == 3

    def test_list_avail(self):
        available = Fingerprint.list_available()
        assert "Hydrophobic" in available
        assert "HBDonor" in available
        assert "Distance" not in available
        assert "Interaction" not in available
        available = Fingerprint.list_available(show_hidden=True)
        assert "Hydrophobic" in available
        assert "HBDonor" in available
        assert "Distance" in available
        assert "Interaction" not in available

    def test_unknown_interaction(self):
        with pytest.raises(
            NameError, match=r"Unknown interaction\(s\) in 'interactions': foo"
        ):
            Fingerprint(["Cationic", "foo"])
        with pytest.raises(
            NameError, match=r"Unknown interaction\(s\) in 'parameters': bar"
        ):
            Fingerprint(["Cationic"], parameters={"bar": {}})

    @pytest.fixture
    def fp_to_pickle(self, fp, protein_mol):
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        return fp

    @pytest.fixture
    def fp_unpkl(self, fp_to_pickle):
        pkl = fp_to_pickle.to_pickle()
        return Fingerprint.from_pickle(pkl)

    @pytest.fixture
    def fp_unpkl_file(self, fp_to_pickle, tmp_path):
        pkl_path = tmp_path / "fp.pkl"
        fp_to_pickle.to_pickle(pkl_path)
        return Fingerprint.from_pickle(pkl_path)

    @pytest.fixture(params=["fp_unpkl", "fp_unpkl_file"])
    def fp_pkled(self, request):
        return request.getfixturevalue(request.param)

    def test_pickle(self, fp_to_pickle, fp_pkled):
        assert fp_to_pickle.interactions.keys() == fp_pkled.interactions.keys()
        assert len(fp_to_pickle.ifp) == len(fp_pkled.ifp)
        for frame_ifp, frame_pkl_ifp in zip(fp_to_pickle.ifp, fp_pkled.ifp):
            assert frame_ifp == frame_pkl_ifp

    def test_pickle_custom_interaction(self, fp_unpkl):
        assert hasattr(fp_unpkl, "dummy")
        assert callable(fp_unpkl.dummy)

    def test_run_multiproc_serial_same(self, fp, u, ligand_ag, protein_ag):
        fp.run(u.trajectory[0:100:10], ligand_ag, protein_ag, n_jobs=1, progress=False)
        serial = fp.to_dataframe()
        fp.run(
            u.trajectory[0:100:10], ligand_ag, protein_ag, n_jobs=None, progress=False
        )
        multi = fp.to_dataframe()
        assert serial.equals(multi)

    def test_run_iter_multiproc_serial_same(self, fp, protein_mol):
        run = fp.run_from_iterable
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = sdf_supplier(path)
        run(lig_suppl, protein_mol, n_jobs=1, progress=False)
        serial = fp.to_dataframe()
        run(lig_suppl, protein_mol, n_jobs=None, progress=False)
        multi = fp.to_dataframe()
        assert serial.equals(multi)

    def test_converter_kwargs_raises_error(self, fp, u, ligand_ag, protein_ag):
        with pytest.raises(
            ValueError, match="converter_kwargs must be a list of 2 dicts"
        ):
            fp.run(
                u.trajectory[0:5],
                ligand_ag,
                protein_ag,
                n_jobs=1,
                progress=False,
                converter_kwargs=[dict(force=True)],
            )

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_converter_kwargs(self, fp, n_jobs):
        u = mda.Universe.from_smiles("O=C=O.O=C=O")
        lig, prot = u.atoms.fragments
        fp.run(
            u.trajectory,
            lig,
            prot,
            n_jobs=n_jobs,
            converter_kwargs=[dict(force=True), dict(force=True)],
        )
        assert fp.ifp

    def test_interaction_params(self):
        fp = Fingerprint()
        assert fp.hydrophobic.distance == 4.5
        fp = Fingerprint(parameters={"Hydrophobic": {"distance": 1.0}})
        assert fp.hydrophobic.distance == 1.0
        fp = Fingerprint()
        assert fp.hydrophobic.distance == 4.5
