import pytest
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from numpy.testing import assert_array_almost_equal
from prolif.molecule import Molecule
from prolif.residue import Residue, ResidueGroup
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

    def test_split_residues(self):
        sequence = "ARNDCQEGHILKMFPSTWYV"
        protein = Chem.MolFromSequence(sequence)
        residues = [Residue(res) for res in Chem.SplitMolByPDBResidues(protein).values()]
        rg = ResidueGroup(residues)
        mol = Molecule(protein)
        assert len(sequence) == mol.n_residues
        for molres, res in zip(mol, rg.values()):
            assert molres.resid == res.resid
            assert molres.HasSubstructMatch(res) and res.HasSubstructMatch(molres)

    def test_mapindex(self, mol):
        for atom in mol.GetAtoms():
            assert atom.GetUnsignedProp("mapindex") == atom.GetIdx()

    def test_from_mda(self):
        pass
