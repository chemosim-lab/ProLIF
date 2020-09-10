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
    return prolif.Molecule(mol)


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
    def encoder(self):
        return prolif.Encoder("all")

    def test_ionic(self, encoder, cation, anion):
        assert encoder.anionic(anion, cation) is True
        assert encoder.cationic(cation, anion) is True
        assert encoder.anionic(cation, anion) is False
        assert encoder.cationic(anion, cation) is False

    def test_cation_pi(self, encoder, benzene, cation):
        false = from_mol2("cation_false.mol2")
        assert encoder.cationpi(cation, benzene) is True
        assert encoder.pication(benzene, cation) is True
        assert encoder.cationpi(false, benzene) is False
        assert encoder.cationpi(benzene, cation) is False
        assert encoder.pication(benzene, false) is False
        assert encoder.pication(cation, benzene) is False

    def test_pistacking(self, encoder, benzene, ftf, etf):
        assert encoder.facetoface(benzene, ftf) is True
        assert encoder.facetoface(ftf, benzene) is True
        assert encoder.facetoface(benzene, etf) is False
        assert encoder.facetoface(etf, benzene) is False
        assert encoder.edgetoface(benzene, etf) is True
        assert encoder.edgetoface(etf, benzene) is True
        assert encoder.edgetoface(benzene, ftf) is False
        assert encoder.edgetoface(ftf, benzene) is False
        assert encoder.pistacking(benzene, etf) is True
        assert encoder.pistacking(etf, benzene) is True
        assert encoder.pistacking(ftf, benzene) is True
        assert encoder.pistacking(benzene, ftf) is True

    def test_hydrophobic(self, encoder, ftf, etf, anion, cation, benzene):
        chlorine = from_mol2("chlorine.mol2")
        assert encoder.hydrophobic(benzene, etf) is True
        assert encoder.hydrophobic(benzene, ftf) is True
        assert encoder.hydrophobic(benzene, chlorine) is True
        assert encoder.hydrophobic(benzene, anion) is False
        assert encoder.hydrophobic(benzene, cation) is False

    def test_hbond(self, encoder):
        donor = from_mol2("donor.mol2")
        acceptor = from_mol2("acceptor.mol2")
        acceptor_false = from_mol2("acceptor_false.mol2")
        assert encoder.hbdonor(donor, acceptor) is True
        assert encoder.hbacceptor(acceptor, donor) is True
        assert encoder.hbacceptor(acceptor_false, donor) is False
        assert encoder.hbdonor(donor, acceptor_false) is False
        assert encoder.hbdonor(acceptor, donor) is False
        assert encoder.hbacceptor(donor, acceptor) is False

    def test_xbond(self, encoder):
        donor = from_mol2("xbond_donor.mol2")
        acceptor = from_mol2("xbond_acceptor.mol2")
        acceptor_false_xar = from_mol2("xbond_acceptor_false_xar.mol2")
        acceptor_false_axd = from_mol2("xbond_acceptor_false_axd.mol2")
        assert encoder.xbdonor(donor, acceptor) is True
        assert encoder.xbacceptor(acceptor, donor) is True
        assert encoder.xbacceptor(acceptor_false_xar, donor) is False
        assert encoder.xbdonor(donor, acceptor_false_xar) is False
        assert encoder.xbacceptor(acceptor_false_axd, donor) is False
        assert encoder.xbdonor(donor, acceptor_false_axd) is False
        assert encoder.xbdonor(acceptor, donor) is False
        assert encoder.xbacceptor(donor, acceptor) is False

    def test_metallic(self, encoder):
        ligand = from_mol2("ligand.mol2")
        metal = from_mol2("metal.mol2")
        metal_false = from_mol2("metal_false.mol2")
        assert encoder.metaldonor(metal, ligand) is True
        assert encoder.metalacceptor(ligand, metal) is True
        assert encoder.metalacceptor(ligand, metal_false) is False
        assert encoder.metaldonor(metal_false, ligand) is False
        assert encoder.metalacceptor(metal, ligand) is False
        assert encoder.metaldonor(ligand, metal) is False
