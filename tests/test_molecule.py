import pytest
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from numpy.testing import assert_array_almost_equal
from prolif.molecule import Molecule
from prolif.rdkitmol import BaseRDKitMol
from prolif.datafiles import TOP

class TestBaseRDKitMol:
    def get_pdb_mol(self):
        return Chem.MolFromPDBFile(TOP)

    @pytest.fixture(scope="class")
    def mol(self):
        return BaseRDKitMol(self.get_pdb_mol())

    def test_init(self):
        rdmol = BaseRDKitMol(self.get_pdb_mol())
        assert isinstance(rdmol, Chem.Mol)

    def test_centroid(self, mol):
        expected = ComputeCentroid(mol.GetConformer())
        assert_array_almost_equal(mol.centroid, expected)

    def test_xyz(self, mol):
        expected = mol.GetConformer().GetPositions()
        assert_array_almost_equal(mol.xyz, expected)


class TestMolecule(TestBaseRDKitMol):
    @pytest.fixture(scope="class")
    def mol(self):
        return Molecule(self.get_pdb_mol())

    def test_mapindex(self, mol):
        for atom in mol.GetAtoms():
            assert atom.GetUnsignedProp("mapindex") == atom.GetIdx()
