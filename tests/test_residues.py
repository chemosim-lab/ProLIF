import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from prolif.residue import ResidueId, Residue, ResidueGroup
from .test_molecule import TestBaseRDKitMol

class TestResidueId:
    @pytest.mark.parametrize("name, number, chain", [
        ("ALA", None, None),
        ("ALA", 1, None),
        ("ALA", None, "B"),
        ("ALA", 1, "B"),
        (None, 1, "B"),
        (None, None, "B"),
        (None, 1, None),
        (None, None, None),
    ])
    def test_init(self, name, number, chain):
        resid = ResidueId(name, number, chain)
        assert resid.name == name
        assert resid.number == number
        assert resid.chain == chain
    
    @pytest.mark.parametrize("name, number, chain", [
        ("", None, None),
        (None, 0, None),
        (None, None, ""),
        (None, None, None),
    ])
    def test_init_empty(self, name, number, chain):
        resid = ResidueId(name, number, chain)
        assert resid == ResidueId()

    @pytest.mark.parametrize("name, number, chain", [
        ("ALA", None, None),
        ("ALA", 1, None),
        ("ALA", None, "B"),
        ("ALA", 1, "B"),
        (None, 1, "B"),
        (None, None, "B"),
        (None, 1, None),
        (None, None, None),
    ])
    def test_from_atom(self, name, number, chain):
        atom = Chem.Atom(1)
        mi = Chem.AtomPDBResidueInfo()
        if name:
            mi.SetResidueName(name)
        if number:
            mi.SetResidueNumber(number)
        if chain:
            mi.SetChainId(chain)
        atom.SetMonomerInfo(mi)
        resid = ResidueId.from_atom(atom)
        assert resid.name == name
        assert resid.number == number
        assert resid.chain == chain

    @pytest.mark.parametrize("name, number, chain", [
        ("", None, None),
        (None, 0, None),
        (None, None, ""),
    ])
    def test_from_atom_empty(self, name, number, chain):
        atom = Chem.Atom(1)
        mi = Chem.AtomPDBResidueInfo()
        if name is not None:
            mi.SetResidueName(name)
        if number is not None:
            mi.SetResidueNumber(number)
        if chain is not None:
            mi.SetChainId(chain)
        atom.SetMonomerInfo(mi)
        resid = ResidueId.from_atom(atom)
        assert resid == ResidueId()

    def test_from_atom_no_mi(self):
        atom = Chem.Atom(1)
        resid = ResidueId.from_atom(atom)
        assert resid.name == None
        assert resid.number == None
        assert resid.chain == None

    @pytest.mark.parametrize("resid_str, expected", [
        ("ALA", ("ALA", None, None)),
        ("ALA1", ("ALA", 1, None)),
        ("ALA.B", ("ALA", None, "B")),
        ("ALA1.B", ("ALA", 1, "B")),
        ("1.B", (None, 1, "B")),
        (".B", (None, None, "B")),
        ("1", (None, 1, None)),
        ("", (None, None, None)),
    ])
    def test_string_methods(self, resid_str, expected):
        resid = ResidueId.from_string(resid_str)
        assert resid == ResidueId(*expected)
        assert str(resid) == resid_str

    def test_eq(self):
        name, number, chain = "ALA", 1, "A"
        res1 = ResidueId(name, number, chain)
        res2 = ResidueId(name, number, chain)
        assert res1 == res2

    @pytest.mark.parametrize("resid_str, expected", [
        ("ALA1.A", True),
        ("ALA1", True),
        ("ALA.A", True),
        ("ALA", True),
        ("1.A", True),
        (".A", True),
        ("1", True),
        ("ALA2.A", False),
        ("GLU1.A", False),
        ("ALA1.B", False),
        ("ALA2", False),
        ("GLU1", False),
        ("ALA.B", False),
        ("GLU.A", False),
        ("GLU", False),
        ("2.A", False),
        ("1.B", False),
        (".B", False),
        ("2", False),
    ])
    def test_contains(self, resid_str, expected):
        resid = ResidueId.from_string(resid_str)
        ref = ResidueId.from_string("ALA1.A")
        result = resid in ref
        assert result is expected

class TestResidue(TestBaseRDKitMol):
    @pytest.fixture(scope="class")
    def mol(self):
        mol = Chem.MolFromSequence("A")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return Residue(mol)

    def test_init(self, mol):
        resid = ResidueId.from_atom(mol.GetAtomWithIdx(0))
        assert mol.resid == resid

    def test_str(self, mol):
        assert str(mol) == "ALA1.A"


class TestResidueGroup:
    def test(self):
        pass