import pytest
from numpy.testing import assert_array_almost_equal
from prolif.rdkitmol import BaseRDKitMol
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid


class TestBaseRDKitMol:
    @pytest.fixture(scope="class")
    def mol(self, rdkit_mol):
        return BaseRDKitMol(rdkit_mol)

    def test_init(self, mol):
        assert isinstance(mol, Chem.Mol)

    def test_centroid(self, mol):
        expected = ComputeCentroid(mol.GetConformer())
        assert_array_almost_equal(mol.centroid, expected)

    def test_xyz(self, mol):
        expected = mol.GetConformer().GetPositions()
        assert_array_almost_equal(mol.xyz, expected)
