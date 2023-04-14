from functools import partial
from io import StringIO

import MDAnalysis as mda
import pytest

import prolif as plf


class TestLigNetwork:
    @pytest.fixture(scope="class")
    def get_ligplot(self):
        u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
        lig = u.select_atoms("resname LIG")
        prot = u.select_atoms("protein and byres around 6.5 group ligand", ligand=lig)
        fp = plf.Fingerprint()
        fp.run(u.trajectory[0:2], lig, prot)
        lig = plf.Molecule.from_mda(lig)
        return partial(fp.to_ligplot, lig)

    def test_integration_frame(self, get_ligplot):
        net = get_ligplot(kind="frame", frame=0)
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_integration_agg(self, get_ligplot):
        net = get_ligplot(kind="aggregate", threshold=0)
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_kwargs(self, get_ligplot):
        net = get_ligplot(kekulize=True, match3D=False, rotation=42, carbon=0)
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

    def test_from_ifp_raises_kind(self, get_ligplot):
        with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
            get_ligplot(kind="foo")
