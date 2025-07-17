from functools import partial
from html import escape
from typing import TYPE_CHECKING, cast

import MDAnalysis as mda
import pytest

import prolif as plf
from prolif.exceptions import RunRequiredError
from prolif.plotting.network import LigNetwork

if TYPE_CHECKING:
    from pathlib import Path


class TestLigNetwork:
    @pytest.fixture(scope="class")
    def simple_fp(self) -> plf.Fingerprint:
        return plf.Fingerprint(count=False)

    @pytest.fixture(scope="class")
    def count_fp(self) -> plf.Fingerprint:
        return plf.Fingerprint(count=True)

    @pytest.fixture(scope="class", params=["simple_fp", "count_fp"])
    def fp(self, request: pytest.FixtureRequest) -> plf.Fingerprint:
        return cast(plf.Fingerprint, request.getfixturevalue(request.param))

    @pytest.fixture(scope="class")
    def fp_mol(self, fp: plf.Fingerprint) -> tuple[plf.Fingerprint, plf.Molecule]:
        u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
        lig = u.select_atoms("resname LIG")
        prot = u.select_atoms("protein and byres around 6.5 group ligand", ligand=lig)
        fp.run(u.trajectory[0:2], lig, prot)
        lig_mol = plf.Molecule.from_mda(lig)
        return fp, lig_mol

    @pytest.fixture(scope="class")
    def get_ligplot(
        self, fp_mol: tuple[plf.Fingerprint, plf.Molecule]
    ) -> partial[LigNetwork]:
        fp, lig_mol = fp_mol
        return partial(LigNetwork.from_fingerprint, fp, lig_mol)

    def test_integration_frame(
        self, fp_mol: tuple[plf.Fingerprint, plf.Molecule]
    ) -> None:
        fp, lig_mol = fp_mol
        net = LigNetwork.from_fingerprint(
            fp,
            lig_mol,
            kind="frame",
            frame=0,
            display_all=fp.count,
        )
        view = net.display()
        assert view._iframe
        html = view._iframe
        assert escape('"from": 5, "to": "PHE331.B", "title": "Hydrophobic') in html
        if fp.count:
            assert escape('"from": 14, "to": "PHE331.B", "title": "Hydrophobic') in html

    def test_integration_agg(self, get_ligplot: partial[LigNetwork]) -> None:
        net = get_ligplot(kind="aggregate", threshold=0)
        view = net.display()
        assert view._iframe
        assert "PHE331.B" in view._iframe

    def test_kwargs(self, get_ligplot: partial[LigNetwork]) -> None:
        net = get_ligplot(
            kekulize=True,
            use_coordinates=True,
            flatten_coordinates=False,
            rotation=42,
            carbon=0,
        )
        view = net.display()
        assert view._iframe
        assert "PHE331.B" in view._iframe

    def test_save_file(
        self, get_ligplot: partial[LigNetwork], tmp_path: "Path"
    ) -> None:
        net = get_ligplot()
        output = tmp_path / "lignetwork.html"
        net.save(output)
        assert "PHE331.B" in output.read_text()

    def test_from_fingerprint_raises_kind(
        self, get_ligplot: partial[LigNetwork]
    ) -> None:
        with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
            get_ligplot(kind="foo")

    def test_from_fingerprint_raises_not_executed(
        self, ligand_mol: plf.Molecule
    ) -> None:
        fp = plf.Fingerprint()
        with pytest.raises(
            RunRequiredError,
            match="Please run the fingerprint analysis before attempting to display"
            " results",
        ):
            LigNetwork.from_fingerprint(fp, ligand_mol)

    def test_show_interaction_data(
        self, fp_mol: tuple[plf.Fingerprint, plf.Molecule]
    ) -> None:
        fp, lig_mol = fp_mol
        view = fp.plot_lignetwork(lig_mol, show_interaction_data=True)
        assert view._iframe
        assert "50%" in view._iframe

    def test_fp_plot_lignetwork(
        self, fp_mol: tuple[plf.Fingerprint, plf.Molecule]
    ) -> None:
        fp, lig_mol = fp_mol
        view = fp.plot_lignetwork(lig_mol, kind="frame", frame=0, display_all=fp.count)
        assert view._iframe
        assert "<iframe" in view._iframe

    def test_water(
        self, water_mols: tuple[plf.Molecule, plf.Molecule, plf.Molecule]
    ) -> None:
        ligand, protein, water = water_mols
        fp = plf.Fingerprint(
            ["HBDonor", "WaterBridge"], parameters={"WaterBridge": {"water": water}}
        )
        fp.run_from_iterable([ligand], protein)
        view = fp.plot_lignetwork(ligand, kind="frame", frame=0)
        assert view._iframe
        assert "TIP383.X" in view._iframe
