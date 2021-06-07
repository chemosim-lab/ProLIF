import MDAnalysis as mda
import prolif as plf
from prolif.plotting.network import LigNetwork
import pytest

def test_from_ifp_raises_kind():
    with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
        LigNetwork.from_ifp(None, kind="foo")

@pytest.fixture(scope="module")
def lignetwork_data():
    u = mda.Universe(plf.datafiles.TOP)
    prot = u.select_atoms("protein")
    lig = u.select_atoms("resname LIG")
    fp = plf.Fingerprint(["HBDonor"])
    fp.run(u.trajectory, lig, prot, return_atoms=True)
    df = fp.to_dataframe()
    return lig, df

def test_integration_frame(lignetwork_data):
    lig, df = lignetwork_data
    net = LigNetwork.from_ifp(df, lig, kind="frame", frame=0)
    html = net._get_html()
    assert html

def test_integration_agg(lignetwork_data):
    lig, df = lignetwork_data
    net = LigNetwork.from_ifp(df, lig, kind="aggregate", threshold=0)
    html = net._get_html()
    assert html
