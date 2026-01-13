from typing import TYPE_CHECKING
from unittest.mock import Mock

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.converters.RDKit import set_converter_cache_size
from pandas import DataFrame
from rdkit.DataStructs import ExplicitBitVect, UIntSparseIntVect

from prolif.datafiles import datapath
from prolif.fingerprint import DEFAULT_INTERACTIONS, Fingerprint
from prolif.interactions.base import _INTERACTIONS, Interaction
from prolif.molecule import sdf_supplier
from prolif.residue import ResidueId

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from MDAnalysis.core.groups import AtomGroup

    from prolif.molecule import Molecule
    from prolif.typeshed import InteractionMetadata


@pytest.fixture
def dummy_cls(cleanup_dummy: "Iterator[None]") -> type[Interaction]:
    class Dummy(Interaction):
        def detect(
            self, res1: "Molecule", res2: "Molecule"
        ) -> "Iterator[InteractionMetadata]":
            yield self.metadata(res1, res2, (2,), (4,), distance=4.2)

    return Dummy


def test_interaction_base(
    sdf_suppl: sdf_supplier, dummy_cls: type[Interaction]
) -> None:
    interaction = dummy_cls()
    repr_ = repr(interaction)
    assert repr_.startswith("<")
    assert ".Dummy at " in repr_
    assert callable(interaction)
    mol = sdf_suppl[0]
    metadata = next(interaction.detect(mol, mol))
    assert metadata["indices"] == {"ligand": (2,), "protein": (4,)}
    assert metadata["parent_indices"] == {"ligand": (2,), "protein": (4,)}
    # invert
    inverted = dummy_cls.invert_role("Dummy", "inverted")
    inverted_interaction = inverted()
    metadata = next(inverted_interaction.detect(mol, mol))
    assert metadata["indices"] == {"ligand": (4,), "protein": (2,)}
    assert metadata["parent_indices"] == {"ligand": (4,), "protein": (2,)}


class TestFingerprint:
    @pytest.fixture(scope="class")
    def fp(self) -> Fingerprint:
        return Fingerprint()

    @pytest.fixture(scope="class")
    def fp_count(self) -> Fingerprint:
        return Fingerprint(count=True)

    @pytest.fixture(scope="class", params=["fp", "fp_count"])
    def any_fp(self, request: pytest.FixtureRequest) -> Fingerprint:
        return request.getfixturevalue(request.param)  # type: ignore[no-any-return]

    @pytest.fixture(scope="class")
    def fp_simple(self) -> Fingerprint:
        return Fingerprint(["Hydrophobic"])

    def test_init(self, fp_simple: Fingerprint) -> None:
        assert "Hydrophobic" in fp_simple.interactions
        assert hasattr(fp_simple, "hydrophobic") and callable(fp_simple.hydrophobic)
        assert "Dummy" not in fp_simple.interactions
        assert "Interaction" not in fp_simple.interactions
        assert not hasattr(fp_simple, "interaction")

    def test_init_all(self) -> None:
        fp = Fingerprint("all")
        assert set(fp.interactions) == set(_INTERACTIONS)

    def test_n_interactions(self, fp: Fingerprint) -> None:
        assert fp.n_interactions == len(fp.interactions)

    def test_bitvector(
        self, any_fp: Fingerprint, ligand_mol: "Molecule", protein_mol: "Molecule"
    ) -> None:
        bv = any_fp.bitvector(ligand_mol[0], protein_mol["ASP129.A"])
        assert len(bv) == any_fp.n_interactions
        assert (bv == 0).any()
        assert bv.sum() > 0
        if any_fp.count:
            assert (bv > 1).any()

    def test_metadata(
        self, any_fp: Fingerprint, ligand_mol: "Molecule", protein_mol: "Molecule"
    ) -> None:
        metadata = any_fp.metadata(ligand_mol[0], protein_mol["ASP129.A"])
        assert metadata
        assert "HBAcceptor" not in metadata
        assert all(
            isinstance(cationic["indices"]["ligand"], tuple)
            for cationic in metadata["Cationic"]
        )

    def test_run_residues(
        self,
        fp_simple: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        fp_simple.run(
            u.trajectory[0:1],
            ligand_ag,
            protein_ag,
            residues=["VAL201.A"],
            progress=False,
        )
        lig_id = ResidueId.from_string("LIG1.G")
        assert hasattr(fp_simple, "ifp")
        assert len(fp_simple.ifp) == 1
        res = ResidueId.from_string("VAL201.A")
        assert (lig_id, res) in fp_simple.ifp[0]
        fp_simple.run(
            u.trajectory[1:2],
            ligand_ag,
            protein_ag,
            residues="all",
            progress=False,
        )
        assert hasattr(fp_simple, "ifp")
        assert len(fp_simple.ifp) == 1
        res = ResidueId.from_string("MET337.B")
        assert (lig_id, res) in fp_simple.ifp[1]
        fp_simple.run(
            u.trajectory[:3],
            ligand_ag,
            protein_ag,
            residues=None,
            progress=False,
        )
        assert hasattr(fp_simple, "ifp")
        assert len(fp_simple.ifp) == 3
        assert len(fp_simple.ifp[0]) > 1
        res = ResidueId.from_string("PHE351.B")
        assert (lig_id, res) in fp_simple.ifp[0]
        u.trajectory[0]

    def test_generate(
        self, fp_simple: Fingerprint, ligand_mol: "Molecule", protein_mol: "Molecule"
    ) -> None:
        ifp = fp_simple.generate(ligand_mol, protein_mol)
        key = (ResidueId("LIG", 1, "G"), ResidueId("VAL", 201, "A"))
        bv = ifp[key]
        assert isinstance(bv, np.ndarray)
        assert bv[0] is np.True_

    def test_generate_metadata(
        self, fp_simple: Fingerprint, ligand_mol: "Molecule", protein_mol: "Molecule"
    ) -> None:
        ifp = fp_simple.generate(ligand_mol, protein_mol, metadata=True)
        key = (ResidueId("LIG", 1, "G"), ResidueId("VAL", 201, "A"))
        int_data = ifp[key]
        assert "Hydrophobic" in int_data

    @pytest.mark.parametrize(
        "trajectory_slice",
        [
            slice(0, 1),  # test for MDAnalysis.coordinates.base.FrameIteratorSliced
            0,  # test for MDAnalysis.coordinates.timestep.Timestep
            np.array(
                [
                    0,
                    2,
                ]
            ),  # test for MDAnalysis.coordinates.base.FrameIteratorIndices
        ],
    )
    @pytest.mark.parametrize("n_jobs", [None, 1])
    def test_run(
        self,
        fp_simple: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
        trajectory_slice: slice | int | np.ndarray,
        n_jobs: int | None,
        tmp_path: "Path",
    ) -> None:
        if isinstance(trajectory_slice, np.ndarray):
            pdb_path = tmp_path / "multi_frame.pdb"
            with mda.Writer(str(pdb_path), u.atoms.n_atoms) as W:
                for _ in range(3):
                    W.write(u.atoms)
            u_test = mda.Universe(str(pdb_path))
            traj = u_test.trajectory[trajectory_slice]
        else:
            traj = u.trajectory[trajectory_slice]

        fp_simple.run(
            traj, ligand_ag, protein_ag, residues=None, progress=False, n_jobs=n_jobs
        )

        assert hasattr(fp_simple, "ifp")
        ifp = fp_simple.ifp[0]
        interactions = next(iter(ifp.values()))
        assert isinstance(interactions, dict)
        metadata_tuple = next(iter(interactions.values()))
        assert all(
            key in metadata_tuple[0]
            for key in ["indices", "parent_indices", "distance"]
        )

    def test_run_from_iterable(
        self, fp_simple: Fingerprint, protein_mol: "Molecule"
    ) -> None:
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp_simple.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        assert len(fp_simple.ifp) == 2

    def test_to_df(
        self,
        fp_simple: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        with pytest.raises(AttributeError, match="use the `run` method"):
            Fingerprint().to_dataframe()
        fp_simple.run(
            u.trajectory[:3],
            ligand_ag,
            protein_ag,
            residues=None,
            progress=False,
        )
        df = fp_simple.to_dataframe()
        assert isinstance(df, DataFrame)
        assert len(df) == 3

    def test_to_df_kwargs(
        self,
        fp_simple: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        fp_simple.run(
            u.trajectory[:3],
            ligand_ag,
            protein_ag,
            residues=None,
            progress=False,
        )
        df = fp_simple.to_dataframe(dtype=np.uint8)
        assert df.dtypes.iloc[0].type is np.uint8
        df = fp_simple.to_dataframe(drop_empty=False)
        resids = {key for d in fp_simple.ifp.values() for key in d}
        assert df.shape == (3, len(resids))

    def test_to_bitvector(
        self,
        fp_simple: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        with pytest.raises(AttributeError, match="use the `run` method"):
            Fingerprint().to_bitvectors()
        fp_simple.run(
            u.trajectory[:3],
            ligand_ag,
            protein_ag,
            residues=None,
            progress=False,
        )
        bvs = fp_simple.to_bitvectors()
        assert isinstance(bvs[0], ExplicitBitVect)
        assert len(bvs) == 3

    def test_to_countvectors(
        self,
        fp_count: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        with pytest.raises(AttributeError, match="use the `run` method"):
            Fingerprint().to_countvectors()
        fp_count.run(
            u.trajectory[:3],
            ligand_ag,
            protein_ag,
            residues=None,
            progress=False,
        )
        cvs = fp_count.to_countvectors()
        assert isinstance(cvs[0], UIntSparseIntVect)
        assert list(cvs[0])[:5] == [0, 1, 2, 3, 1]
        assert len(cvs) == 3

    def test_list_avail(self) -> None:
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

    def test_unknown_interaction(self) -> None:
        with pytest.raises(
            NameError,
            match=r"Unknown interaction\(s\) in 'interactions': foo",
        ):
            Fingerprint(["Cationic", "foo"])
        with pytest.raises(
            NameError,
            match=r"Unknown interaction\(s\) in 'parameters': bar",
        ):
            Fingerprint(["Cationic"], parameters={"bar": {}})

    @pytest.fixture()
    def fp_to_pickle(
        self,
        protein_mol: "Molecule",
        dummy_cls: type[Interaction],  # noqa: ARG002
    ) -> Fingerprint:
        fp = Fingerprint([*DEFAULT_INTERACTIONS, "Dummy"])
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = list(sdf_supplier(path))
        fp.run_from_iterable(lig_suppl[:2], protein_mol, progress=False)
        return fp

    @pytest.fixture()
    def fp_unpkl(self, fp_to_pickle: Fingerprint) -> Fingerprint:
        pkl = fp_to_pickle.to_pickle()
        return Fingerprint.from_pickle(pkl)

    @pytest.fixture()
    def fp_unpkl_file(self, fp_to_pickle: Fingerprint, tmp_path: "Path") -> Fingerprint:
        pkl_path = tmp_path / "fp.pkl"
        fp_to_pickle.to_pickle(pkl_path)
        return Fingerprint.from_pickle(pkl_path)

    @pytest.fixture(params=["fp_unpkl", "fp_unpkl_file"])
    def fp_pkled(self, request: pytest.FixtureRequest) -> Fingerprint:
        return request.getfixturevalue(request.param)  # type: ignore[no-any-return]

    def test_pickle(self, fp_to_pickle: Fingerprint, fp_pkled: Fingerprint) -> None:
        assert fp_to_pickle.interactions.keys() == fp_pkled.interactions.keys()
        assert len(fp_to_pickle.ifp) == len(fp_pkled.ifp)
        for frame_ifp, frame_pkl_ifp in zip(
            fp_to_pickle.ifp, fp_pkled.ifp, strict=True
        ):
            assert frame_ifp == frame_pkl_ifp

    def test_pickle_custom_interaction(self, fp_unpkl: Fingerprint) -> None:
        assert hasattr(fp_unpkl, "dummy")
        assert callable(fp_unpkl.dummy)

    def test_run_multiproc_serial_same(
        self,
        fp: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        fp.run(u.trajectory[0:100:10], ligand_ag, protein_ag, n_jobs=1, progress=False)
        serial = fp.to_dataframe()
        fp.run(
            u.trajectory[0:100:10],
            ligand_ag,
            protein_ag,
            n_jobs=None,
            progress=False,
        )
        multi = fp.to_dataframe()
        assert serial.equals(multi)

    def test_run_multiproc_on_single_frame_runs_serial(
        self,
        fp: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mocked = Mock(wraps=fp._run_serial)
        monkeypatch.setattr(fp, "_run_serial", mocked)
        fp.run(
            u.trajectory[1],
            ligand_ag,
            protein_ag,
            n_jobs=2,
            progress=False,
        )
        mocked.assert_called_once()

    def test_run_serial_on_xtc_runs_whole_trajectory(
        self, u: mda.Universe, ligand_ag: "AtomGroup"
    ) -> None:
        """When checking wether a single timestep was passed or a trajectory,
        :meth:`Fingerprint._run_serial` was only checking if the object had a
        `frame` attribute, without checking for `n_frames` first. This would lead to
        only running the analysis on the first frame if an XTC was passed."""
        assert hasattr(u.trajectory, "frame")
        assert hasattr(u.trajectory, "n_frames")
        fp = Fingerprint(["VdWContact"])
        fp.run(
            u.trajectory,
            ligand_ag,
            ligand_ag,
            n_jobs=1,
            progress=False,
        )
        assert len(fp.ifp) == u.trajectory.n_frames

    def test_run_iter_multiproc_serial_same(
        self, fp: Fingerprint, protein_mol: "Molecule"
    ) -> None:
        run = fp.run_from_iterable
        path = str(datapath / "vina" / "vina_output.sdf")
        lig_suppl = sdf_supplier(path)
        run(lig_suppl, protein_mol, n_jobs=1, progress=False)
        serial = fp.to_dataframe()
        run(lig_suppl, protein_mol, n_jobs=None, progress=False)
        multi = fp.to_dataframe()
        assert serial.equals(multi)

    def test_converter_kwargs_raises_error(
        self,
        fp: Fingerprint,
        u: mda.Universe,
        ligand_ag: "AtomGroup",
        protein_ag: "AtomGroup",
    ) -> None:
        with pytest.raises(
            ValueError,
            match="converter_kwargs must be a list of 2 dicts",
        ):
            fp.run(
                u.trajectory[0:5],
                ligand_ag,
                protein_ag,
                n_jobs=1,
                progress=False,
                converter_kwargs=({"force": True},),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_converter_kwargs(self, fp: Fingerprint, n_jobs: int) -> None:
        u = mda.Universe.from_smiles("O=C=O.O=C=O")
        lig, prot = u.atoms.fragments
        fp.run(
            u.trajectory,
            lig,
            prot,
            n_jobs=n_jobs,
            converter_kwargs=({"force": True}, {"force": True}),
        )
        assert fp.ifp

    def test_interaction_params(self) -> None:
        fp = Fingerprint()
        assert fp.hydrophobic.distance == 4.5  # type: ignore[attr-defined]
        fp = Fingerprint(parameters={"Hydrophobic": {"distance": 1.0}})
        assert fp.hydrophobic.distance == 1.0  # type: ignore[attr-defined]
        fp = Fingerprint()
        assert fp.hydrophobic.distance == 4.5  # type: ignore[attr-defined]

    def test_water_bridge_instance_without_params_raises_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Must specify settings for bridged interaction 'WaterBridge'",
        ):
            Fingerprint(["WaterBridge"])

    def test_mix_water_bridge_and_other_interactions(
        self,
        water_u: mda.Universe,
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
    ) -> None:
        ligand, protein, water = water_atomgroups
        fp = Fingerprint(
            ["HBDonor", "WaterBridge"], parameters={"WaterBridge": {"water": water}}
        )
        fp.run(water_u.trajectory[:1], ligand, protein, n_jobs=1)

        assert "WaterBridge" in fp.ifp[0]["QNB1.X", "TRP400.X"]
        assert "HBDonor" in fp.ifp[0]["QNB1.X", "ASN404.X"]

    def test_water_bridge_run_iter(
        self, water_mols: tuple["Molecule", "Molecule", "Molecule"]
    ) -> None:
        ligand, protein, water = water_mols
        fp = Fingerprint(
            ["HBDonor", "WaterBridge"], parameters={"WaterBridge": {"water": water}}
        )
        fp.run_from_iterable([ligand], protein)

        assert "WaterBridge" in fp.ifp[0]["QNB1.X", "TRP400.X"]
        assert "HBDonor" in fp.ifp[0]["QNB1.X", "ASN404.X"]

    def test_water_bridge_updates_cache_size(
        self,
        water_u: mda.Universe,
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ligand, protein, water = water_atomgroups
        set_converter_cache_size(2)
        mocked = Mock(wraps=set_converter_cache_size)
        monkeypatch.setattr(
            Fingerprint, "_run_bridged_analysis", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr("prolif.fingerprint.set_converter_cache_size", mocked)

        fp = Fingerprint(["WaterBridge"], parameters={"WaterBridge": {"water": water}})
        fp.run(water_u.trajectory[:1], ligand, protein)
        mocked.assert_called_once_with(3)

    def test_run_can_switch_to_segid_over_chains(
        self,
        water_u: "mda.Universe",
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        If more segids than chains are present, Molecule should switch to segindex.
        """
        lig, prot, water = water_atomgroups
        fp = Fingerprint(["WaterBridge"], parameters={"WaterBridge": {"water": water}})
        assert fp._use_segid(lig, prot) is False
        water = water_u.select_atoms("resname TIP3 and byres around 6 protein")
        fp = Fingerprint(["WaterBridge"], parameters={"WaterBridge": {"water": water}})
        assert fp._use_segid(lig, prot) is True

        def patch_run_serial(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return {0: {}}

        monkeypatch.setattr(fp, "_run_serial", patch_run_serial)
        fp.run(
            water_u.trajectory[:1],
            lig,
            prot,
            n_jobs=1,
        )
        assert fp.use_segid is True
