import pytest
import os
from rdkit import Chem, RDLogger
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.topology.guessers import guess_atom_element
from prolif.molecule import Molecule
from prolif.fingerprint import Fingerprint
from prolif.datafiles import datapath
from prolif.interactions import _INTERACTIONS, Interaction

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


def from_mol2(f):
    path = str(datapath / f)
    u = Universe(path)
    elements = [guess_atom_element(n) for n in u.atoms.names]
    u.add_TopologyAttr("elements", np.array(elements, dtype=object))
    u.atoms.types = np.array([x.upper() for x in u.atoms.types], dtype=object)
    return Molecule.from_mda(u, force=True)


class MolFactory:
    def benzene():
        return from_mol2("benzene.mol2")

    def cation():
        return from_mol2("cation.mol2")

    def cation_false():
        return from_mol2("cation_false.mol2")

    def anion():
        return from_mol2("anion.mol2")

    def ftf():
        return from_mol2("facetoface.mol2")

    def etf():
        return from_mol2("edgetoface.mol2")

    def chlorine():
        return from_mol2("chlorine.mol2")

    def hb_donor():
        return from_mol2("donor.mol2")

    def hb_acceptor():
        return from_mol2("acceptor.mol2")

    def hb_acceptor_false():
        return from_mol2("acceptor_false.mol2")

    def xb_donor():
        return from_mol2("xbond_donor.mol2")

    def xb_acceptor():
        return from_mol2("xbond_acceptor.mol2")

    def xb_acceptor_false_xar():
        return from_mol2("xbond_acceptor_false_xar.mol2")

    def xb_acceptor_false_axd():
        return from_mol2("xbond_acceptor_false_axd.mol2")

    def ligand():
        return from_mol2("ligand.mol2")

    def metal():
        return from_mol2("metal.mol2")

    def metal_false():
        return from_mol2("metal_false.mol2")


@pytest.fixture(scope="module")
def mol1(request):
    return getattr(MolFactory, request.param)()


@pytest.fixture(scope="module")
def mol2(request):
    return getattr(MolFactory, request.param)()


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
                pass
        new = id(_INTERACTIONS["Hydrophobic"])
        assert old != new

    def test_error_no_detect(self):
        class Dummy(Interaction):
            pass
        with pytest.raises(TypeError,
                           match="Can't instantiate abstract class Dummy"):
            foo = Dummy()