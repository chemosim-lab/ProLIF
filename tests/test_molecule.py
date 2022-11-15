import pytest
from MDAnalysis import SelectionError
from numpy.testing import assert_array_equal
from prolif.datafiles import datapath
from prolif.molecule import (Molecule, mol2_supplier, pdbqt_supplier,
                             sdf_supplier)
from prolif.residue import ResidueId
from rdkit import Chem

from .test_base import TestBaseRDKitMol, ligand_rdkit, rdkit_mol, u


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

    def test_from_mda_empty_ag(self):
        ag = u.select_atoms("resname FOO")
        with pytest.raises(SelectionError, match="AtomGroup is empty"):
            Molecule.from_mda(ag)
    
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


class SupplierBase:
    resid = ResidueId("UNL", 1, "")

    def test_len(self, suppl):
        assert len(suppl) == 9

    def test_returns_mol(self, suppl):
        mol = next(iter(suppl))
        assert isinstance(mol, Molecule)

    def test_monomer_info(self, suppl):
        mol = next(iter(suppl))
        resid = ResidueId.from_atom(mol.GetAtomWithIdx(0))
        assert resid == self.resid

    @pytest.mark.parametrize("index", [0, 2, 8, -1])
    def test_index(self, suppl, index):
        index %= 9
        mol_i = suppl[index]
        for i, mol in enumerate(suppl):
            if i == index:
                break
        assert_array_equal(mol.xyz, mol_i.xyz)


class TestPDBQTSupplier(SupplierBase):
    resid = ResidueId("LIG", 1, "G")

    @pytest.fixture
    def suppl(self):
        path = datapath / "vina"
        pdbqts = sorted(path.glob("*.pdbqt"))
        template = Chem.MolFromSmiles("C[NH+]1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)"
                                      "C(Cc4ccccc4)N3C2=O)C=C2c3cccc4[nH]cc"
                                      "(c34)CC21")
        return pdbqt_supplier(pdbqts, template)


class TestSDFSupplier(SupplierBase):
    @pytest.fixture
    def suppl(self):
        path = str(datapath / "vina" / "vina_output.sdf")
        return sdf_supplier(path)


class TestMOL2Supplier(SupplierBase):
    @pytest.fixture
    def suppl(self):
        path = str(datapath / "vina" / "vina_output.mol2")
        return mol2_supplier(path)

    def test_mol2_starting_with_comment(self):
        path = str(datapath / "mol_comment.mol2")
        suppl = mol2_supplier(path)
        mol = next(iter(suppl))
        assert mol is not None


def test_pdbqt_hydrogens_stay_in_mol():
    smiles_params = Chem.SmilesParserParams()
    smiles_params.sanitize = False
    smiles_params.removeHs = False
    pdb_mol = Chem.MolFromSmiles(
        "[C@@]1(C)(N(C(O)[C@H]2C[N@](C)([H])[C@@H]3CC4CN([H])C5[C@@H]4[C@H]([C@@H]3C2)CCC5)[H])[C@H](O)N2[C@@H](CC3CCCCC3)[C@@H](O)N3CCC[C@H]3[C@]2(O[H])O1",
        smiles_params
    )
    h_indices = set()
    for atom in pdb_mol.GetAtoms():
        atom.SetIntProp("_MDAnalysis_index", atom.GetIdx())
        if atom.GetAtomicNum() == 1:
            h_indices.add(atom.GetIdx())
            atom.SetBoolProp("flagged", True)
        else:
            atom.SetBoolProp("flagged", False)
    template = Chem.MolFromSmiles("C[NH+]1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)C(Cc4ccccc4)N3C2=O)C=C2c3cccc4[nH]cc(c34)CC21")
    mol = pdbqt_supplier._adjust_hydrogens(template, pdb_mol)
    hydrogens = [
        atom for atom in mol.GetAtoms()
        if atom.HasProp("_MDAnalysis_index")
        and atom.GetIntProp("_MDAnalysis_index") in h_indices
    ]
    assert all(atom.GetBoolProp("flagged") for atom in hydrogens)
    
