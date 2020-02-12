import unittest
import os
from rdkit import Chem, RDLogger
import prolif

class TestInteractions(unittest.TestCase):

    def get_file(self, f):
        return os.path.join(os.path.dirname(__file__), f)

    def setUp(self):
        # Fingerprint Factory
        self.ff = prolif.FingerprintFactory()
        # Molecules used regularly
        benzene = Chem.MolFromMol2File(self.get_file("benzene.mol2"))
        benzene = prolif.Trajectory.from_rdkit(benzene)
        benzene = next(iter(next(iter(benzene))))
        self.benzene = benzene

    def test_cation_pi(self):
        # prepare inputs
        cation = Chem.MolFromMol2File(self.get_file("cation.mol2"))
        false = Chem.MolFromMol2File(self.get_file("cation_false.mol2"))
        cation = prolif.Trajectory.from_rdkit(cation)
        cation = next(iter(next(iter(cation))))
        false = prolif.Trajectory.from_rdkit(false)
        false = next(iter(next(iter(false))))
        # positive tests
        self.assertEqual(self.ff.get_cation_pi(cation, self.benzene), 1)
        # negative tests
        self.assertEqual(self.ff.get_cation_pi(false, self.benzene), 0)
        self.assertEqual(self.ff.get_cation_pi(self.benzene, cation), 0)
        self.assertEqual(self.ff.get_cation_pi(cation, cation), 0)
        self.assertEqual(self.ff.get_cation_pi(self.benzene, self.benzene), 0)

    def test_pi_cation(self):
        # prepare inputs
        cation = Chem.MolFromMol2File(self.get_file("cation.mol2"))
        false = Chem.MolFromMol2File(self.get_file("cation_false.mol2"))
        cation = prolif.Trajectory.from_rdkit(cation)
        cation = next(iter(next(iter(cation))))
        false = prolif.Trajectory.from_rdkit(false)
        false = next(iter(next(iter(false))))
        # positive tests
        self.assertEqual(self.ff.get_pi_cation(self.benzene, cation), 1)
        # negative tests
        self.assertEqual(self.ff.get_pi_cation(self.benzene, false), 0)
        self.assertEqual(self.ff.get_pi_cation(cation, self.benzene), 0)
        self.assertEqual(self.ff.get_pi_cation(cation, cation), 0)
        self.assertEqual(self.ff.get_pi_cation(self.benzene, self.benzene), 0)

    def test_face_to_face(self):
        # prepare inputs
        f2f = Chem.MolFromMol2File(self.get_file("facetoface.mol2"))
        e2f = Chem.MolFromMol2File(self.get_file("edgetoface.mol2"))
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
        f2f = Chem.MolFromMol2File(self.get_file("facetoface.mol2"))
        e2f = Chem.MolFromMol2File(self.get_file("edgetoface.mol2"))
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


if __name__ == '__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    unittest.main()
