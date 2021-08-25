import pytest
from rdkit import Chem
from prolif.molecule import (Molecule,
                             pdbqt_supplier,
                             mol2_supplier,
                             sdf_supplier)
from prolif.residue import ResidueId
from prolif.datafiles import datapath
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
    
    def test_from_rdkit(self):
        rdkit_mol = Molecule(ligand_rdkit)
        newmol = Molecule.from_rdkit(ligand_rdkit)
        assert rdkit_mol[0].resid == newmol[0].resid

    def test_from_rdkit_default_resid(self):
        mol = Chem.MolFromSmiles("CCO")
        newmol = Molecule.from_rdkit(mol)
        assert newmol[0].resid == ResidueId("UNL", 1)

    def test_from_rdkit_resid_args(self):
        mol = Chem.MolFromSmiles("CCO")
        newmol = Molecule.from_rdkit(mol, "FOO", 42, "A")
        assert newmol[0].resid == ResidueId("FOO", 42, "A")

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


class TestSupplier:
    def test_pdbqt(self):
        path = datapath / "vina"
        pdbqts = sorted(path.glob("*.pdbqt"))
        template = Chem.MolFromSmiles("C[NH+]1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)"
                                      "C(Cc4ccccc4)N3C2=O)C=C2c3cccc4[nH]cc"
                                      "(c34)CC21")
        suppl = pdbqt_supplier(pdbqts, template)
        mols = list(suppl)
        assert isinstance(mols[0], Molecule)
        assert len(mols) == len(pdbqts)

    def test_sdf(self):
        path = str(datapath / "vina" / "vina_output.sdf")
        suppl = sdf_supplier(path)
        mols = list(suppl)
        assert isinstance(mols[0], Molecule)
        assert len(mols) == 9
        mi = mols[0].GetAtomWithIdx(0).GetMonomerInfo()
        assert all([mi.GetResidueName() == "UNL",
                    mi.GetResidueNumber() == 1,
                    mi.GetChainId() == ""])

    def test_mol2(self):
        path = str(datapath / "vina" / "vina_output.mol2")
        suppl = mol2_supplier(path)
        mols = list(suppl)
        assert isinstance(mols[0], Molecule)
        assert len(mols) == 9
        mi = mols[0].GetAtomWithIdx(0).GetMonomerInfo()
        assert all([mi.GetResidueName() == "UNL",
                    mi.GetResidueNumber() == 1,
                    mi.GetChainId() == ""])

    def test_mol2_starting_with_comment(self):
        path = str(datapath / "mol_comment.mol2")
        suppl = mol2_supplier(path)
        mol = next(suppl)
        assert mol is not None
