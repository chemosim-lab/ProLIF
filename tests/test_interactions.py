import pytest
from rdkit import RDLogger
from prolif.fingerprint import Fingerprint
from prolif.interactions import _INTERACTIONS, Interaction, get_mapindex
import prolif
from .test_base import ligand_mol
from . import mol2factory

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


@pytest.fixture(scope="module")
def mol1(request):
    return getattr(mol2factory, request.param)()


@pytest.fixture(scope="module")
def mol2(request):
    return getattr(mol2factory, request.param)()


class TestInteractions:
    @pytest.fixture(scope="class")
    def fingerprint(self):
        return Fingerprint()

    @pytest.mark.parametrize("func_name, mol1, mol2, expected", [
        ("cationic", "cation", "anion", True),
        ("cationic", "anion", "cation", False),
        ("cationic", "cation", "benzene", False),
        ("anionic", "cation", "anion", False),
        ("anionic", "anion", "cation", True),
        ("anionic", "anion", "benzene", False),
        ("cationpi", "cation", "benzene", True),
        ("cationpi", "cation_false", "benzene", False),
        ("cationpi", "benzene", "cation", False),
        ("cationpi", "cation", "cation", False),
        ("cationpi", "benzene", "benzene", False),
        ("pication", "benzene", "cation", True),
        ("pication", "benzene", "cation_false", False),
        ("pication", "cation", "benzene", False),
        ("pication", "cation", "cation", False),
        ("pication", "benzene", "benzene", False),
        ("pistacking", "benzene", "etf", True),
        ("pistacking", "etf", "benzene", True),
        ("pistacking", "ftf", "benzene", True),
        ("pistacking", "benzene", "ftf", True),
        ("facetoface", "benzene", "ftf", True),
        ("facetoface", "ftf", "benzene", True),
        ("facetoface", "benzene", "etf", False),
        ("facetoface", "etf", "benzene", False),
        ("edgetoface", "benzene", "etf", True),
        ("edgetoface", "etf", "benzene", True),
        ("edgetoface", "benzene", "ftf", False),
        ("edgetoface", "ftf", "benzene", False),
        ("hydrophobic", "benzene", "etf", True),
        ("hydrophobic", "benzene", "ftf", True),
        ("hydrophobic", "benzene", "chlorine", True),
        ("hydrophobic", "benzene", "anion", False),
        ("hydrophobic", "benzene", "cation", False),
        ("hbdonor", "hb_donor", "hb_acceptor", True),
        ("hbdonor", "hb_donor", "hb_acceptor_false", False),
        ("hbdonor", "hb_acceptor", "hb_donor", False),
        ("hbacceptor", "hb_acceptor", "hb_donor", True),
        ("hbacceptor", "hb_acceptor_false", "hb_donor", False),
        ("hbacceptor", "hb_donor", "hb_acceptor", False),
        ("xbdonor", "xb_donor", "xb_acceptor", True),
        ("xbdonor", "xb_donor", "xb_acceptor_false_xar", False),
        ("xbdonor", "xb_donor", "xb_acceptor_false_axd", False),
        ("xbdonor", "xb_acceptor", "xb_donor", False),
        ("xbacceptor", "xb_acceptor", "xb_donor", True),
        ("xbacceptor", "xb_acceptor_false_xar", "xb_donor", False),
        ("xbacceptor", "xb_acceptor_false_axd", "xb_donor", False),
        ("xbacceptor", "xb_donor", "xb_acceptor", False),
        ("metaldonor", "metal", "ligand", True),
        ("metaldonor", "metal_false", "ligand", False),
        ("metaldonor", "ligand", "metal", False),
        ("metalacceptor", "ligand", "metal", True),
        ("metalacceptor", "ligand", "metal_false", False),
        ("metalacceptor", "metal", "ligand", False),
    ], indirect=["mol1", "mol2"])
    def test_interaction(self, fingerprint, func_name, mol1, mol2, expected):
        interaction = getattr(fingerprint, func_name)
        assert interaction(mol1, mol2) is expected

    def test_warning_supersede(self):
        old = id(_INTERACTIONS["Hydrophobic"])
        with pytest.warns(UserWarning,
                          match="interaction has been superseded"):
            class Hydrophobic(Interaction):
                def detect(self):
                    pass
        new = id(_INTERACTIONS["Hydrophobic"])
        assert old != new
        # fix dummy Hydrophobic class being reused in later unrelated tests

        class Hydrophobic(prolif.interactions.Hydrophobic):
            pass

    def test_error_no_detect(self):
        class Dummy(Interaction):
            pass
        with pytest.raises(TypeError,
                           match="Can't instantiate abstract class Dummy"):
            Dummy()
        # fix Dummy class being reused in later unrelated tests
        del prolif.interactions._INTERACTIONS["Dummy"]

    @pytest.mark.parametrize("index", [
        0, 1, 3, 42, 78
    ])
    def test_get_mapindex(self, index):
        parent_index = get_mapindex(ligand_mol[0], index)
        assert parent_index == index
