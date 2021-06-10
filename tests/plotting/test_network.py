from io import StringIO
from tempfile import NamedTemporaryFile
import os
from contextlib import contextmanager
import MDAnalysis as mda
import prolif as plf
from prolif.plotting.network import LigNetwork
import pytest


@contextmanager
def TempFilename(mode='w'):
    f = NamedTemporaryFile(delete=False, mode=mode)
    try:
        f.close()
        yield f.name
    finally:
        os.unlink(f.name)


class TestLigNetwork:
    @pytest.fixture(scope="class")
    def lignetwork_data(self):
        u = mda.Universe(plf.datafiles.TOP)
        prot = u.select_atoms("protein and resid 200:331")
        lig = u.select_atoms("resname LIG")
        fp = plf.Fingerprint()
        fp.run(u.trajectory, lig, prot)
        df = fp.to_dataframe(return_atoms=True)
        lig = plf.Molecule.from_mda(lig)
        return lig, df

    def test_integration_frame(self, lignetwork_data):
        lig, df = lignetwork_data
        net = LigNetwork.from_ifp(df, lig, kind="frame", frame=0)
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_integration_agg(self, lignetwork_data):
        lig, df = lignetwork_data
        net = LigNetwork.from_ifp(df, lig, kind="aggregate", threshold=0)
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_kwargs(self, lignetwork_data):
        lig, df = lignetwork_data
        net = LigNetwork.from_ifp(df, lig, kekulize=True, match3D=False,
                                  rotation=42, carbon=0)
        with StringIO() as buffer:
            net.save(buffer)
            buffer.seek(0)
            html = buffer.read()
        assert "PHE331.B" in html

    def test_save_file(self, lignetwork_data):
        lig, df = lignetwork_data
        net = LigNetwork.from_ifp(df, lig)
        with TempFilename("w+") as filename:
            net.save(filename)
            with open(filename, "r") as f:
                assert "PHE331.B" in f.read()

    def test_from_ifp_raises_kind(self, lignetwork_data):
        lig, df = lignetwork_data
        with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
            LigNetwork.from_ifp(df, lig, kind="foo")
