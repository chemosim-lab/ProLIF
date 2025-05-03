from typing import cast

import MDAnalysis as mda
import pytest

import prolif as plf
from prolif.exceptions import RunRequiredError
from prolif.plotting.complex3d import Complex3D


class TestComplex3D:
    @pytest.fixture(scope="class")
    def simple_fp(self) -> plf.Fingerprint:
        return plf.Fingerprint(count=False)

    @pytest.fixture(scope="class")
    def count_fp(self) -> plf.Fingerprint:
        return plf.Fingerprint(count=True)

    @pytest.fixture(scope="class", params=["simple_fp", "count_fp"])
    def fp(self, request: pytest.FixtureRequest) -> plf.Fingerprint:
        return cast(plf.Fingerprint, request.getfixturevalue(request.param))

    def execute_fp(
        self, fp: plf.Fingerprint
    ) -> tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]:
        u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
        lig = u.select_atoms("resname LIG")
        prot = u.select_atoms("protein and byres around 6.5 group ligand", ligand=lig)
        fp.run(u.trajectory[0:2], lig, prot)
        lig_mol = plf.Molecule.from_mda(lig)
        prot_mol = plf.Molecule.from_mda(prot)
        return fp, lig_mol, prot_mol

    @pytest.fixture(scope="class")
    def fp_mols(
        self, fp: plf.Fingerprint
    ) -> tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]:
        return self.execute_fp(fp)

    @pytest.fixture(scope="class")
    def simple_fp_results(
        self, simple_fp: plf.Fingerprint
    ) -> tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]:
        return self.execute_fp(simple_fp)

    @pytest.fixture(scope="class")
    def plot_3d(
        self, fp_mols: tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]
    ) -> Complex3D:
        fp, lig_mol, prot_mol = fp_mols
        return Complex3D.from_fingerprint(fp, lig_mol, prot_mol, frame=0)

    def test_integration_display_single(self, plot_3d: Complex3D) -> None:
        view = plot_3d.display(display_all=False)
        assert view._view
        html = view._view._make_html()
        assert "Hydrophobic" in html

    def test_integration_display_all(self, plot_3d: Complex3D) -> None:
        view = plot_3d.display(display_all=True)
        assert view._view
        html = view._view._make_html()
        assert "Hydrophobic" in html

    def test_integration_compare(self, plot_3d: Complex3D) -> None:
        view = plot_3d.compare(plot_3d)
        assert view._view
        html = view._view._make_html()
        assert "Hydrophobic" in html

    def test_from_fingerprint_raises_not_executed(
        self, ligand_mol: plf.Molecule, protein_mol: plf.Molecule
    ) -> None:
        fp = plf.Fingerprint()
        with pytest.raises(
            RunRequiredError,
            match="Please run the fingerprint analysis before attempting to display"
            " results",
        ):
            Complex3D.from_fingerprint(fp, ligand_mol, protein_mol, frame=0)

    def test_fp_plot_3d(
        self, fp_mols: tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]
    ) -> None:
        fp, lig_mol, prot_mol = fp_mols
        view = fp.plot_3d(lig_mol, prot_mol, frame=0, display_all=fp.count)
        assert view._view
        html = view._view._make_html()
        assert "Hydrophobic" in html

    def test_getattr_raises_error_if_not_initialized(
        self, simple_fp_results: tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]
    ) -> None:
        fp, lig_mol, prot_mol = simple_fp_results
        plot_3d = Complex3D.from_fingerprint(fp, lig_mol, prot_mol, frame=0)
        with pytest.raises(ValueError, match="View not initialized"):
            plot_3d._make_html()

    def test_passing_complex_to_populate_view(
        self, simple_fp_results: tuple[plf.Fingerprint, plf.Molecule, plf.Molecule]
    ) -> None:
        """For backwards compatibility"""
        fp, lig_mol, prot_mol = simple_fp_results
        plot_3d = Complex3D.from_fingerprint(fp, lig_mol, prot_mol, frame=0)
        plot_3d.display()
        plot_3d._populate_view(plot_3d)

    def test_water(
        self, water_mols: tuple[plf.Molecule, plf.Molecule, plf.Molecule]
    ) -> None:
        ligand, protein, water = water_mols
        fp = plf.Fingerprint(
            ["HBDonor", "WaterBridge"],
            parameters={"WaterBridge": {"water": water, "order": 2}},
        )
        fp.run_from_iterable([ligand], protein)
        view = fp.plot_3d(ligand, protein, water, frame=0)
        html = view._make_html()
        assert "TIP383.X" in html
