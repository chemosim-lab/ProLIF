from functools import partial
from io import StringIO

import MDAnalysis as mda
import pytest

import prolif as plf
from prolif.exceptions import RunRequiredError
from prolif.plotting.network import LigNetwork


class TestLigNetwork:
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
    def fp_mol(self, fp):
        u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
        lig = u.select_atoms("resname LIG")
        prot = u.select_atoms("protein and byres around 6.5 group ligand", ligand=lig)
        fp.run(u.trajectory[0:2], lig, prot)
        lig_mol = plf.Molecule.from_mda(lig)
        return fp, lig_mol

    @pytest.fixture(scope="class")
    def get_ligplot(self, fp_mol):
        fp, lig_mol = fp_mol
        return partial(LigNetwork.from_fingerprint, fp, lig_mol)

    def test_integration_frame(self, fp_mol):
        fp, lig_mol = fp_mol
        net = LigNetwork.from_fingerprint(
            fp, lig_mol, kind="frame", frame=0, display_all=fp.count
        )
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert '"from": 18, "to": "PHE351.B", "title": "Hydrophobic' in html
        if fp.count:
            assert '"from": 19, "to": "PHE351.B", "title": "Hydrophobic' in html

    def test_integration_agg(self, get_ligplot):
        net = get_ligplot(kind="aggregate", threshold=0)
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_kwargs(self, get_ligplot):
        net = get_ligplot(
            kekulize=True,
            use_coordinates=True,
            flatten_coordinates=False,
            rotation=42,
            carbon=0,
        )
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_save_file(self, get_ligplot, tmp_path):
        net = get_ligplot()
        output = tmp_path / "lignetwork.html"
        net.save(output)
        with open(output, "r") as f:
            assert "PHE331.B" in f.read()

    def test_from_fingerprint_raises_kind(self, get_ligplot):
        with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
            get_ligplot(kind="foo")

    def test_from_fingerprint_raises_not_executed(self, ligand_mol):
        fp = plf.Fingerprint()
        with pytest.raises(
            RunRequiredError,
            match="Please run the fingerprint analysis before attempting to display results",
        ):
            LigNetwork.from_fingerprint(fp, ligand_mol)

    def test_fp_plot_lignetwork(self, fp_mol):
        fp, lig_mol = fp_mol
        html = fp.plot_lignetwork(lig_mol, kind="frame", frame=0, display_all=fp.count)
        assert "<iframe" in html.data
