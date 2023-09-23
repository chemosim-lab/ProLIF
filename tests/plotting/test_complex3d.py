import MDAnalysis as mda
import pytest

import prolif as plf
from prolif.exceptions import RunRequiredError
from prolif.plotting.complex3d import Complex3D


class TestComplex3D:
    @pytest.fixture(scope="class")
    def simple_fp(self):
        return plf.Fingerprint(count=False)

    @pytest.fixture(scope="class")
    def count_fp(self):
        return plf.Fingerprint(count=True)

    @pytest.fixture(scope="class", params=["simple_fp", "count_fp"])
    def fp(self, request):
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="class")
    def fp_mols(self, fp):
        u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
        lig = u.select_atoms("resname LIG")
        prot = u.select_atoms("protein and byres around 6.5 group ligand", ligand=lig)
        fp.run(u.trajectory[0:2], lig, prot)
        lig_mol = plf.Molecule.from_mda(lig)
        prot_mol = plf.Molecule.from_mda(prot)
        return fp, lig_mol, prot_mol

    @pytest.fixture(scope="class")
    def plot_3d(self, fp_mols):
        fp, lig_mol, prot_mol = fp_mols
        return Complex3D.from_fingerprint(fp, lig_mol, prot_mol, frame=0)

    def test_integration_display_single(self, plot_3d):
        view = plot_3d.display(display_all=False)
        html = view._make_html()
        assert "Hydrophobic" in html

    def test_integration_display_all(self, plot_3d):
        view = plot_3d.display(display_all=True)
        html = view._make_html()
        assert "Hydrophobic" in html

    def test_integration_compare(self, plot_3d):
        view = plot_3d.compare(plot_3d)
        html = view._make_html()
        assert "Hydrophobic" in html

    def test_from_fingerprint_raises_not_executed(self, ligand_mol, protein_mol):
        fp = plf.Fingerprint()
        with pytest.raises(
            RunRequiredError,
            match="Please run the fingerprint analysis before attempting to display results",
        ):
            Complex3D.from_fingerprint(
                fp,
                ligand_mol,
                protein_mol,
                frame=0,
            )

    def test_fp_plot_3d(self, fp_mols):
        fp, lig_mol, prot_mol = fp_mols
        view = fp.plot_3d(lig_mol, prot_mol, frame=0, display_all=fp.count)
        html = view._make_html()
        assert "Hydrophobic" in html
