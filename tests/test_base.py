import pytest
from MDAnalysis import Universe
from numpy.testing import assert_array_almost_equal
from prolif.datafiles import TOP, TRAJ
from prolif.molecule import Molecule
from prolif.rdkitmol import BaseRDKitMol
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

u = Universe(TOP, TRAJ)
rdkit_mol = Chem.MolFromPDBFile(TOP, removeHs=False)
ligand_ag = u.select_atoms("resname LIG")
ligand_rdkit = ligand_ag.convert_to.rdkit()
ligand_mol = Molecule.from_mda(ligand_ag)
protein_ag = u.select_atoms("protein")
protein_rdkit = protein_ag.convert_to.rdkit()
protein_mol = Molecule.from_mda(protein_ag)


class TestBaseRDKitMol:
    @pytest.fixture(scope="class")
    def mol(self):
        return BaseRDKitMol(rdkit_mol)

    def test_init(self, mol):
        assert isinstance(mol, Chem.Mol)

    def test_centroid(self, mol):
        expected = ComputeCentroid(mol.GetConformer())
        assert_array_almost_equal(mol.centroid, expected)

    def test_xyz(self, mol):
        expected = mol.GetConformer().GetPositions()
        assert_array_almost_equal(mol.xyz, expected)
