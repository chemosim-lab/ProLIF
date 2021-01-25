import pytest
from prolif.molecule import (Molecule,
                             pdbqt_supplier,
                             mol2_supplier,
                             sdf_supplier,
                             _patch_rdkit_mol)
from prolif.residue import ResidueId
from .test_base import TestBaseRDKitMol, rdkit_mol, ligand_rdkit, u


class TestMolecule(TestBaseRDKitMol):
    @pytest.fixture(scope="class")
    def mol(self):
        return Molecule(rdkit_mol)

    def test_mapindex(self, mol):
        for atom in mol.GetAtoms():
            assert atom.GetUnsignedProp("mapindex") == atom.GetIdx()

    def test_from_mda(self):
        rdkit_mol = Molecule(ligand_rdkit)
        mda_mol = Molecule.from_mda(u, "resname LIG")
        assert rdkit_mol[0].resid == mda_mol[0].resid
        assert (rdkit_mol.HasSubstructMatch(mda_mol) and
                mda_mol.HasSubstructMatch(rdkit_mol))

    @pytest.mark.parametrize("key", [
        0,
        42,
        -1,
        "LYS49.A",
        ResidueId("LYS", 49, "A")
    ])
    def test_getitem(self, mol, key):
        assert mol[key].resid is mol.residues[key].resid

    def test_iter(self, mol):
        for i, r in enumerate(mol):
            assert r.resid == mol[i].resid

    def test_n_residues(self, mol):
        assert mol.n_residues == mol.residues.n_residues
