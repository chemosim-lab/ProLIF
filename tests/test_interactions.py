import unittest
import os
from rdkit import Chem, RDLogger
import prolif

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class TestInteractions(unittest.TestCase):

    def get_file(self, f):
        return os.path.join(os.path.dirname(__file__), f)

    def setUp(self):
        # Fingerprint Factory
        self.ff = prolif.FingerprintFactory()
        # Molecules used regularly
        benzene = Chem.MolFromMol2File(self.get_file("benzene.mol2"), removeHs=False)
        benzene = prolif.Trajectory.from_rdkit(benzene)
        benzene = next(iter(next(iter(benzene))))
        self.benzene = benzene

    def test_ionic(self):
        # prepare inputs
        cation = Chem.MolFromMol2File(self.get_file("cation.mol2"), removeHs=False)
        anion = Chem.MolFromMol2File(self.get_file("anion.mol2"), removeHs=False)
        cation = prolif.Trajectory.from_rdkit(cation)
        cation = next(iter(next(iter(cation))))
        anion = prolif.Trajectory.from_rdkit(anion)
        anion = next(iter(next(iter(anion))))
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
        cation = prolif.Trajectory.from_rdkit(cation)
        cation = next(iter(next(iter(cation))))
        false = prolif.Trajectory.from_rdkit(false)
        false = next(iter(next(iter(false))))
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
        f2f = prolif.Trajectory.from_rdkit(f2f)
        f2f = next(iter(next(iter(f2f))))
        e2f = prolif.Trajectory.from_rdkit(e2f)
        e2f = next(iter(next(iter(e2f))))
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
        f2f = prolif.Trajectory.from_rdkit(f2f)
        f2f = next(iter(next(iter(f2f))))
        e2f = prolif.Trajectory.from_rdkit(e2f)
        e2f = next(iter(next(iter(e2f))))
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
        f2f = prolif.Trajectory.from_rdkit(f2f)
        f2f = next(iter(next(iter(f2f))))
        e2f = prolif.Trajectory.from_rdkit(e2f)
        e2f = next(iter(next(iter(e2f))))
        chlorine = prolif.Trajectory.from_rdkit(chlorine)
        chlorine = next(iter(next(iter(chlorine))))
        anion = prolif.Trajectory.from_rdkit(anion)
        anion = next(iter(next(iter(anion))))
        cation = prolif.Trajectory.from_rdkit(cation)
        cation = next(iter(next(iter(cation))))
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
        donor = prolif.Trajectory.from_rdkit(donor)
        donor = next(iter(next(iter(donor))))
        acceptor = prolif.Trajectory.from_rdkit(acceptor)
        acceptor = next(iter(next(iter(acceptor))))
        acceptor_false = prolif.Trajectory.from_rdkit(acceptor_false)
        acceptor_false = next(iter(next(iter(acceptor_false))))
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
        donor = prolif.Trajectory.from_rdkit(donor)
        donor = next(iter(next(iter(donor))))
        acceptor = prolif.Trajectory.from_rdkit(acceptor)
        acceptor = next(iter(next(iter(acceptor))))
        acceptor_false_xar = prolif.Trajectory.from_rdkit(acceptor_false_xar)
        acceptor_false_xar = next(iter(next(iter(acceptor_false_xar))))
        acceptor_false_axd = prolif.Trajectory.from_rdkit(acceptor_false_axd)
        acceptor_false_axd = next(iter(next(iter(acceptor_false_axd))))
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
        ligand = prolif.Trajectory.from_rdkit(ligand)
        ligand = next(iter(next(iter(ligand))))
        metal = prolif.Trajectory.from_rdkit(metal)
        metal = next(iter(next(iter(metal))))
        metal_false = prolif.Trajectory.from_rdkit(metal_false)
        metal_false = next(iter(next(iter(metal_false))))
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
