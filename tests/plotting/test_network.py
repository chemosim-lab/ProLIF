import MDAnalysis as mda
import prolif as plf
from prolif.plotting.network import LigNetwork
import pytest

class TestLigNetwork:
    @pytest.fixture(scope="class")
    def lignetwork_data(self):
        u = mda.Universe(plf.datafiles.TOP)
        prot = u.select_atoms("protein and resid 120-130")
        lig = u.select_atoms("resname LIG")
        fp = plf.Fingerprint(["Hydrophobic"])
        fp.run(u.trajectory, lig, prot)
        df = fp.to_dataframe(return_atoms=True)
        lig = plf.Molecule.from_mda(lig)
        return lig, df


    def test_integration_frame(self, lignetwork_data):
        lig, df = lignetwork_data
        net = LigNetwork.from_ifp(df, lig, kind="frame", frame=0)
        html = net._get_html()
        assert html


    def test_integration_agg(self, lignetwork_data):
        lig, df = lignetwork_data
        net = LigNetwork.from_ifp(df, lig, kind="aggregate", threshold=0)
        html = net._get_html()
        assert html


    def test_from_ifp_raises_kind(self, lignetwork_data):
        lig, df = lignetwork_data
        with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
            LigNetwork.from_ifp(df, lig, kind="foo")
