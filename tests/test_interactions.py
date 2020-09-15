import pytest
import os
from rdkit import Chem, RDLogger
import prolif

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


def from_mol2(f):
    path = os.path.join(os.path.dirname(__file__), "data", f)
    mol = Chem.MolFromMol2File(path, removeHs=False)
    return prolif.Molecule(mol, cache=False)


class TestInteractions:
    @pytest.fixture
    def benzene(self):
        return from_mol2("benzene.mol2")
    
    @pytest.fixture
    def cation(self):
        return from_mol2("cation.mol2")

    @pytest.fixture
    def anion(self):
        return from_mol2("anion.mol2")

    @pytest.fixture
    def ftf(self):
        return from_mol2("facetoface.mol2")

    @pytest.fixture
    def etf(self):
        return from_mol2("edgetoface.mol2")
    
    @pytest.fixture(scope="class")
    def fingerprint(self):
        return prolif.Fingerprint("all")

    def test_ionic(self, fingerprint, cation, anion):
        assert fingerprint.anionic(anion, cation) is True
        assert fingerprint.cationic(cation, anion) is True
        assert fingerprint.anionic(cation, anion) is False
        assert fingerprint.cationic(anion, cation) is False

    def test_cation_pi(self, fingerprint, benzene, cation):
        false = from_mol2("cation_false.mol2")
        assert fingerprint.cationpi(cation, benzene) is True
        assert fingerprint.pication(benzene, cation) is True
        assert fingerprint.cationpi(false, benzene) is False
        assert fingerprint.cationpi(benzene, cation) is False
        assert fingerprint.pication(benzene, false) is False
        assert fingerprint.pication(cation, benzene) is False

    def test_pistacking(self, fingerprint, benzene, ftf, etf):
        assert fingerprint.facetoface(benzene, ftf) is True
        assert fingerprint.facetoface(ftf, benzene) is True
        assert fingerprint.facetoface(benzene, etf) is False
        assert fingerprint.facetoface(etf, benzene) is False
        assert fingerprint.edgetoface(benzene, etf) is True
        assert fingerprint.edgetoface(etf, benzene) is True
        assert fingerprint.edgetoface(benzene, ftf) is False
        assert fingerprint.edgetoface(ftf, benzene) is False
        assert fingerprint.pistacking(benzene, etf) is True
        assert fingerprint.pistacking(etf, benzene) is True
        assert fingerprint.pistacking(ftf, benzene) is True
        assert fingerprint.pistacking(benzene, ftf) is True

    def test_hydrophobic(self, fingerprint, ftf, etf, anion, cation, benzene):
        chlorine = from_mol2("chlorine.mol2")
        assert fingerprint.hydrophobic(benzene, etf) is True
        assert fingerprint.hydrophobic(benzene, ftf) is True
        assert fingerprint.hydrophobic(benzene, chlorine) is True
        assert fingerprint.hydrophobic(benzene, anion) is False
        assert fingerprint.hydrophobic(benzene, cation) is False

    def test_hbond(self, fingerprint):
        donor = from_mol2("donor.mol2")
        acceptor = from_mol2("acceptor.mol2")
        acceptor_false = from_mol2("acceptor_false.mol2")
        assert fingerprint.hbdonor(donor, acceptor) is True
        assert fingerprint.hbacceptor(acceptor, donor) is True
        assert fingerprint.hbacceptor(acceptor_false, donor) is False
        assert fingerprint.hbdonor(donor, acceptor_false) is False
        assert fingerprint.hbdonor(acceptor, donor) is False
        assert fingerprint.hbacceptor(donor, acceptor) is False

    def test_xbond(self, fingerprint):
        donor = from_mol2("xbond_donor.mol2")
        acceptor = from_mol2("xbond_acceptor.mol2")
        acceptor_false_xar = from_mol2("xbond_acceptor_false_xar.mol2")
        acceptor_false_axd = from_mol2("xbond_acceptor_false_axd.mol2")
        assert fingerprint.xbdonor(donor, acceptor) is True
        assert fingerprint.xbacceptor(acceptor, donor) is True
        assert fingerprint.xbacceptor(acceptor_false_xar, donor) is False
        assert fingerprint.xbdonor(donor, acceptor_false_xar) is False
        assert fingerprint.xbacceptor(acceptor_false_axd, donor) is False
        assert fingerprint.xbdonor(donor, acceptor_false_axd) is False
        assert fingerprint.xbdonor(acceptor, donor) is False
        assert fingerprint.xbacceptor(donor, acceptor) is False

    def test_metallic(self, fingerprint):
        ligand = from_mol2("ligand.mol2")
        metal = from_mol2("metal.mol2")
        metal_false = from_mol2("metal_false.mol2")
        assert fingerprint.metaldonor(metal, ligand) is True
        assert fingerprint.metalacceptor(ligand, metal) is True
        assert fingerprint.metalacceptor(ligand, metal_false) is False
        assert fingerprint.metaldonor(metal_false, ligand) is False
        assert fingerprint.metalacceptor(metal, ligand) is False
        assert fingerprint.metaldonor(ligand, metal) is False
