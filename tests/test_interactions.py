import unittest
import os
from rdkit import Chem, RDLogger
import prolif

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class TestInteractions(unittest.TestCase):

    def get_file(self, f):
        return os.path.join(os.path.dirname(__file__), "data", f)

    def setUp(self):
        # Fingerprint Factory
        self.ff = prolif.FingerprintFactory()
        # Molecules used regularly
        benzene = Chem.MolFromMol2File(self.get_file("benzene.mol2"), removeHs=False)
        benzene = prolif.Trajectory.from_rdkit(benzene).get_frame().get_residue()
        self.benzene = benzene

    def test_ionic(self):
        # prepare inputs
        cation = Chem.MolFromMol2File(self.get_file("cation.mol2"), removeHs=False)
        anion = Chem.MolFromMol2File(self.get_file("anion.mol2"), removeHs=False)
        cation = prolif.Trajectory.from_rdkit(cation).get_frame().get_residue()
        anion = prolif.Trajectory.from_rdkit(anion).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_anionic(anion, cation), 1)
        self.assertEqual(self.ff.get_cationic(cation, anion), 1)
        # negative tests
        self.assertEqual(self.ff.get_anionic(cation, anion), 0)
        self.assertEqual(self.ff.get_cationic(anion, cation), 0)

    def test_cation_pi(self):
        # prepare inputs
        cation = Chem.MolFromMol2File(self.get_file("cation.mol2"), removeHs=False)
        false = Chem.MolFromMol2File(self.get_file("cation_false.mol2"), removeHs=False)
        cation = prolif.Trajectory.from_rdkit(cation).get_frame().get_residue()
        false = prolif.Trajectory.from_rdkit(false).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_cation_pi(cation, self.benzene), 1)
        self.assertEqual(self.ff.get_pi_cation(self.benzene, cation), 1)
        # negative tests
        self.assertEqual(self.ff.get_cation_pi(false, self.benzene), 0)
        self.assertEqual(self.ff.get_cation_pi(self.benzene, cation), 0)
        self.assertEqual(self.ff.get_pi_cation(self.benzene, false), 0)
        self.assertEqual(self.ff.get_pi_cation(cation, self.benzene), 0)

    def test_face_to_face(self):
        # prepare inputs
        f2f = Chem.MolFromMol2File(self.get_file("facetoface.mol2"), removeHs=False)
        e2f = Chem.MolFromMol2File(self.get_file("edgetoface.mol2"), removeHs=False)
        f2f = prolif.Trajectory.from_rdkit(f2f).get_frame().get_residue()
        e2f = prolif.Trajectory.from_rdkit(e2f).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_face_to_face(self.benzene, f2f), 1)
        self.assertEqual(self.ff.get_face_to_face(f2f, self.benzene), 1)
        # negative tests
        self.assertEqual(self.ff.get_face_to_face(self.benzene, e2f), 0)
        self.assertEqual(self.ff.get_face_to_face(e2f, self.benzene), 0)

    def test_edge_to_face(self):
        # prepare inputs
        f2f = Chem.MolFromMol2File(self.get_file("facetoface.mol2"), removeHs=False)
        e2f = Chem.MolFromMol2File(self.get_file("edgetoface.mol2"), removeHs=False)
        f2f = prolif.Trajectory.from_rdkit(f2f).get_frame().get_residue()
        e2f = prolif.Trajectory.from_rdkit(e2f).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_edge_to_face(self.benzene, e2f), 1)
        self.assertEqual(self.ff.get_edge_to_face(e2f, self.benzene), 1)
        # negative tests
        self.assertEqual(self.ff.get_edge_to_face(self.benzene, f2f), 0)
        self.assertEqual(self.ff.get_edge_to_face(f2f, self.benzene), 0)

    def test_hydrophobic(self):
        # prepare inputs
        f2f = Chem.MolFromMol2File(self.get_file("facetoface.mol2"), removeHs=False)
        e2f = Chem.MolFromMol2File(self.get_file("edgetoface.mol2"), removeHs=False)
        chlorine = Chem.MolFromMol2File(self.get_file("chlorine.mol2"), removeHs=False)
        anion = Chem.MolFromMol2File(self.get_file("anion.mol2"), removeHs=False)
        cation = Chem.MolFromMol2File(self.get_file("cation.mol2"), removeHs=False)
        f2f = prolif.Trajectory.from_rdkit(f2f).get_frame().get_residue()
        e2f = prolif.Trajectory.from_rdkit(e2f).get_frame().get_residue()
        chlorine = prolif.Trajectory.from_rdkit(chlorine).get_frame().get_residue()
        anion = prolif.Trajectory.from_rdkit(anion).get_frame().get_residue()
        cation = prolif.Trajectory.from_rdkit(cation).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_hydrophobic(self.benzene, e2f), 1)
        self.assertEqual(self.ff.get_hydrophobic(self.benzene, f2f), 1)
        self.assertEqual(self.ff.get_hydrophobic(self.benzene, chlorine), 1)
        # negative tests
        self.assertEqual(self.ff.get_hydrophobic(self.benzene, anion), 0)
        self.assertEqual(self.ff.get_hydrophobic(self.benzene, cation), 0)

    def test_hbond(self):
        # prepare inputs
        donor = Chem.MolFromMol2File(self.get_file("donor.mol2"), removeHs=False)
        acceptor = Chem.MolFromMol2File(self.get_file("acceptor.mol2"), removeHs=False)
        acceptor_false = Chem.MolFromMol2File(self.get_file("acceptor_false.mol2"), removeHs=False)
        donor = prolif.Trajectory.from_rdkit(donor).get_frame().get_residue()
        acceptor = prolif.Trajectory.from_rdkit(acceptor).get_frame().get_residue()
        acceptor_false = prolif.Trajectory.from_rdkit(acceptor_false).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_hbond_donor(donor, acceptor), 1)
        self.assertEqual(self.ff.get_hbond_acceptor(acceptor, donor), 1)
        # negative tests
        self.assertEqual(self.ff.get_hbond_acceptor(acceptor_false, donor), 0)
        self.assertEqual(self.ff.get_hbond_donor(donor, acceptor_false), 0)
        self.assertEqual(self.ff.get_hbond_donor(acceptor, donor), 0)
        self.assertEqual(self.ff.get_hbond_acceptor(donor, acceptor), 0)

    def test_xbond(self):
        # prepare inputs
        donor = Chem.MolFromMol2File(self.get_file("xbond_donor.mol2"), removeHs=False)
        acceptor = Chem.MolFromMol2File(self.get_file("xbond_acceptor.mol2"), removeHs=False)
        acceptor_false_xar = Chem.MolFromMol2File(self.get_file("xbond_acceptor_false_xar.mol2"), removeHs=False)
        acceptor_false_axd = Chem.MolFromMol2File(self.get_file("xbond_acceptor_false_axd.mol2"), removeHs=False)
        donor = prolif.Trajectory.from_rdkit(donor).get_frame().get_residue()
        acceptor = prolif.Trajectory.from_rdkit(acceptor).get_frame().get_residue()
        acceptor_false_xar = prolif.Trajectory.from_rdkit(acceptor_false_xar).get_frame().get_residue()
        acceptor_false_axd = prolif.Trajectory.from_rdkit(acceptor_false_axd).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_xbond_donor(donor, acceptor), 1)
        self.assertEqual(self.ff.get_xbond_acceptor(acceptor, donor), 1)
        # negative tests
        self.assertEqual(self.ff.get_xbond_acceptor(acceptor_false_xar, donor), 0)
        self.assertEqual(self.ff.get_xbond_donor(donor, acceptor_false_xar), 0)
        self.assertEqual(self.ff.get_xbond_acceptor(acceptor_false_axd, donor), 0)
        self.assertEqual(self.ff.get_xbond_donor(donor, acceptor_false_axd), 0)
        self.assertEqual(self.ff.get_xbond_donor(acceptor, donor), 0)
        self.assertEqual(self.ff.get_xbond_acceptor(donor, acceptor), 0)

    def test_metallic(self):
        # prepare inputs
        ligand = Chem.MolFromMol2File(self.get_file("ligand.mol2"), removeHs=False)
        metal = Chem.MolFromMol2File(self.get_file("metal.mol2"), removeHs=False)
        metal_false = Chem.MolFromMol2File(self.get_file("metal_false.mol2"), removeHs=False)
        ligand = prolif.Trajectory.from_rdkit(ligand).get_frame().get_residue()
        metal = prolif.Trajectory.from_rdkit(metal).get_frame().get_residue()
        metal_false = prolif.Trajectory.from_rdkit(metal_false).get_frame().get_residue()
        # positive tests
        self.assertEqual(self.ff.get_metal_donor(metal, ligand), 1)
        self.assertEqual(self.ff.get_metal_acceptor(ligand, metal), 1)
        # negative tests
        self.assertEqual(self.ff.get_metal_acceptor(ligand, metal_false), 0)
        self.assertEqual(self.ff.get_metal_donor(metal_false, ligand), 0)
        self.assertEqual(self.ff.get_metal_acceptor(metal, ligand), 0)
        self.assertEqual(self.ff.get_metal_donor(ligand, metal), 0)

if __name__ == '__main__':
    unittest.main()
