import prolif as plf
from prolif.plotting.network import LigNetwork
import pytest

def test_from_ifp_raises_kind():
    with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
        LigNetwork.from_ifp(None, kind="foo")
