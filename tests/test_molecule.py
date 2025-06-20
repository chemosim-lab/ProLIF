from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, ClassVar

import pytest
from MDAnalysis import SelectionError
from numpy.testing import assert_array_equal
from rdkit import Chem

from prolif.datafiles import datapath
from prolif.molecule import Molecule, mol2_supplier, pdbqt_supplier, sdf_supplier
from prolif.residue import Residue, ResidueId

if TYPE_CHECKING:
    from collections.abc import Sequence

    from MDAnalysis import Universe


class TestMolecule(pytest.BaseTestMixinRDKitMol):  # type: ignore[name-defined]
    @pytest.fixture(scope="class")
    def mol(self, rdkit_mol: Chem.Mol) -> Molecule:
        return Molecule(rdkit_mol)

    def test_mapindex(self, mol: Molecule) -> None:
        for atom in mol.GetAtoms():
            assert atom.GetUnsignedProp("mapindex") == atom.GetIdx()

    def test_from_mda(self, u: "Universe", ligand_rdkit: Chem.Mol) -> None:
        rdkit_mol = Molecule(ligand_rdkit)
        mda_mol = Molecule.from_mda(u, "resname LIG")
        assert rdkit_mol[0].resid == mda_mol[0].resid
        assert rdkit_mol.HasSubstructMatch(mda_mol) and mda_mol.HasSubstructMatch(
            rdkit_mol,
        )

    def test_from_mda_empty_ag(self, u: "Universe") -> None:
        ag = u.select_atoms("resname FOO")
        with pytest.raises(SelectionError, match="AtomGroup is empty"):
            Molecule.from_mda(ag)

    def test_from_rdkit(self, ligand_rdkit: Chem.Mol) -> None:
        rdkit_mol = Molecule(ligand_rdkit)
        newmol = Molecule.from_rdkit(ligand_rdkit)
        assert rdkit_mol[0].resid == newmol[0].resid

    def test_from_rdkit_default_resid(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        newmol = Molecule.from_rdkit(mol)
        assert newmol[0].resid == ResidueId("UNL", 1)

    def test_from_rdkit_resid_args(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        newmol = Molecule.from_rdkit(mol, "FOO", 42, "A")
        assert newmol[0].resid == ResidueId("FOO", 42, "A")

    @pytest.mark.parametrize("key", [0, 42, -1, "LYS49.A", ResidueId("LYS", 49, "A")])
    def test_getitem(self, mol: Molecule, key: int | str | ResidueId) -> None:
        assert mol[key].resid is mol.residues[key].resid

    def test_iter(self, mol: Molecule) -> None:
        for r in mol:
            assert isinstance(r, Residue)

    def test_n_residues(self, mol: Molecule) -> None:
        assert mol.n_residues == mol.residues.n_residues


class SupplierBase:
    resid = ResidueId("UNL", 1, "")
    slice_behavior: ClassVar[AbstractContextManager] = nullcontext()

    def test_len(self, suppl: "Sequence[Molecule]") -> None:
        assert len(suppl) == 9

    def test_returns_mol(self, suppl: "Sequence[Molecule]") -> None:
        mol = next(iter(suppl))
        assert isinstance(mol, Molecule)

    def test_monomer_info(self, suppl: "Sequence[Molecule]") -> None:
        mol = next(iter(suppl))
        resid = ResidueId.from_atom(mol.GetAtomWithIdx(0))
        assert resid == self.resid

    def test_index(self, suppl: "Sequence[Molecule]") -> None:
        mols = list(suppl)
        for index in [0, 2, 8, -1]:
            mol_i = suppl[index]
            assert_array_equal(mols[index].xyz, mol_i.xyz)

    def test_slice(self, suppl: "Sequence[Molecule]") -> None:
        mols = list(suppl)
        s = slice(1, 5, 2)
        indices = range(*s.indices(len(mols)))
        with self.slice_behavior:
            sliced_suppl = suppl[s]
            for index, mol in zip(indices, sliced_suppl, strict=True):
                assert_array_equal(mols[index].xyz, mol.xyz)


class TestPDBQTSupplier(SupplierBase):
    resid = ResidueId("LIG", 1, "G")

    @pytest.fixture()
    def suppl(self) -> "Sequence[Molecule]":
        path = datapath / "vina"
        pdbqts = sorted(path.glob("*.pdbqt"))
        template = Chem.MolFromSmiles(
            "C[NH+]1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)"
            "C(Cc4ccccc4)N3C2=O)C=C2c3cccc4[nH]cc"
            "(c34)CC21"
        )
        return pdbqt_supplier(pdbqts, template)

    def test_pdbqt_hydrogens_stay_in_mol(self, ligand_rdkit: Chem.Mol) -> None:
        template = Chem.RemoveHs(ligand_rdkit)
        indices = []
        rwmol = Chem.RWMol(ligand_rdkit)
        rwmol.BeginBatchEdit()
        for atom in rwmol.GetAtoms():
            idx = atom.GetIdx()
            atom.SetIntProp("_MDAnalysis_index", idx)
            if atom.GetAtomicNum() == 1:
                if idx % 2:
                    indices.append(idx)
                else:
                    neighbor = atom.GetNeighbors()[0]
                    rwmol.RemoveBond(idx, neighbor.GetIdx())
                    rwmol.RemoveAtom(idx)
                    neighbor.SetNumExplicitHs(1)
        rwmol.CommitBatchEdit()
        pdbqt_mol = rwmol.GetMol()
        mol = pdbqt_supplier._adjust_hydrogens(template, pdbqt_mol)
        hydrogens = [
            idx
            for atom in mol.GetAtoms()
            if atom.HasProp("_MDAnalysis_index")
            and (idx := atom.GetIntProp("_MDAnalysis_index")) in indices
        ]
        assert hydrogens == indices


class TestSDFSupplier(SupplierBase):
    @pytest.fixture()
    def suppl(self) -> "Sequence[Molecule]":
        path = str(datapath / "vina" / "vina_output.sdf")
        return sdf_supplier(path)

    def test_sanitize(self) -> None:
        path = str(datapath / "vina" / "vina_output.sdf")
        suppl = sdf_supplier(path, sanitize=False)
        mol = next(iter(suppl))
        assert isinstance(mol, Molecule)


class TestMOL2Supplier(SupplierBase):
    slice_behavior = pytest.raises(NotImplementedError)

    @pytest.fixture()
    def suppl(self) -> "Sequence[Molecule]":
        path = str(datapath / "vina" / "vina_output.mol2")
        return mol2_supplier(path)

    def test_mol2_starting_with_comment(self) -> None:
        path = str(datapath / "mol_comment.mol2")
        suppl = mol2_supplier(path)
        mol = next(iter(suppl))
        assert mol is not None

    def test_sanitize(self) -> None:
        path = str(datapath / "vina" / "vina_output.mol2")
        suppl = mol2_supplier(path, sanitize=False)
        mol = next(iter(suppl))
        assert isinstance(mol, Molecule)

    def test_cleanup_substructures(self) -> None:
        path = str(datapath / "vina" / "vina_output.mol2")
        suppl = mol2_supplier(path, cleanup_substructures=False)
        mol = next(iter(suppl))
        assert isinstance(mol, Molecule)
