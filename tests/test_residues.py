import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem
from test_base import TestBaseRDKitMol

from prolif.residue import Residue, ResidueGroup, ResidueId


class TestResidueId:
    @pytest.mark.parametrize(
        "name, number, chain",
        [
            ("ALA", None, None),
            ("ALA", 1, None),
            ("ALA", 0, None),
            ("ALA", None, "B"),
            ("ALA", 1, "B"),
            (None, 1, "B"),
            (None, None, "B"),
            (None, 1, None),
            (None, None, None),
            ("", None, None),
            (None, None, ""),
            ("", None, ""),
        ],
    )
    def test_init(self, name, number, chain):
        resid = ResidueId(name, number, chain)
        name = name or "UNK"
        number = number or 0
        chain = chain or None
        assert resid.name == name
        assert resid.number == number
        assert resid.chain == chain

    @pytest.mark.parametrize(
        "name, number, chain",
        [
            ("ALA", None, None),
            ("ALA", 1, None),
            ("ALA", 0, None),
            ("ALA", None, "B"),
            ("ALA", 1, "B"),
            ("DA", 1, None),
            (None, 1, "B"),
            (None, None, "B"),
            (None, 1, None),
            (None, None, None),
            ("", None, None),
            (None, None, ""),
            ("", None, ""),
        ],
    )
    def test_from_atom(self, name, number, chain):
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
        name = name or "UNK"
        number = number or 0
        chain = chain or None
        assert resid.name == name
        assert resid.number == number
        assert resid.chain == chain

    def test_from_atom_no_mi(self):
        atom = Chem.Atom(1)
        resid = ResidueId.from_atom(atom)
        assert resid.name == "UNK"
        assert resid.number == 0
        assert resid.chain is None

    @pytest.mark.parametrize(
        "resid_str, expected",
        [
            ("ALA", ("ALA", 0, None)),
            ("ALA1", ("ALA", 1, None)),
            ("ALA.B", ("ALA", 0, "B")),
            ("ALA1.B", ("ALA", 1, "B")),
            ("1.B", ("UNK", 1, "B")),
            (".B", ("UNK", 0, "B")),
            (".0", ("UNK", 0, "0")),
            ("1", ("UNK", 1, None)),
            ("", ("UNK", 0, None)),
            ("DA2.A", ("DA", 2, "A")),
            ("DA2", ("DA", 2, None)),
            ("DA", ("DA", 0, None)),
        ],
    )
    def test_from_string(self, resid_str, expected):
        resid = ResidueId.from_string(resid_str)
        assert resid == ResidueId(*expected)

    def test_eq(self):
        name, number, chain = "ALA", 1, "A"
        res1 = ResidueId(name, number, chain)
        res2 = ResidueId(name, number, chain)
        assert res1 == res2

    @pytest.mark.parametrize(
        "res1, res2",
        [
            ("ALA1.A", "ALA1.B"),
            ("ALA2.A", "ALA3.A"),
            ("ALA4.A", "ALA1.B"),
        ],
    )
    def test_lt(self, res1, res2):
        res1 = ResidueId.from_string(res1)
        res2 = ResidueId.from_string(res2)
        assert res1 < res2

    @pytest.mark.parametrize("resid_str", ["ALA1.A", "DA2.B", "HIS3", "UNK0"])
    def test_repr(self, resid_str):
        resid = ResidueId.from_string(resid_str)
        expected = f"ResidueId({resid.name}, {resid.number}, {resid.chain})"
        assert repr(resid) == expected
        assert str(resid) == resid_str


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
    @pytest.fixture(scope="class")
    def residues(self):
        sequence = "ARNDCQEGHILKMFPSTWYV"
        protein = Chem.MolFromSequence(sequence)
        residues = [
            Residue(res) for res in Chem.SplitMolByPDBResidues(protein).values()
        ]
        return residues

    def test_init(self, residues):
        rg = ResidueGroup(residues)
        for rg_res, res in zip(rg._residues, residues):
            assert rg_res is res
        for (resid, rg_res), res in zip(rg.items(), residues):
            assert rg_res is res
            assert resid is rg_res.resid
        resinfo = [(r.resid.name, r.resid.number, r.resid.chain) for r in residues]
        name, number, chain = zip(*resinfo)
        assert_equal(rg.name, name)
        assert_equal(rg.number, number)
        assert_equal(rg.chain, chain)

    def test_init_empty(self):
        rg = ResidueGroup([])
        assert_equal(rg.name, [])
        assert_equal(rg.number, [])
        assert_equal(rg.chain, [])
        assert_equal(rg._residues, [])
        assert rg.data == {}

    def test_n_residues(self, residues):
        rg = ResidueGroup(residues)
        assert rg.n_residues == len(rg)
        assert rg.n_residues == 20

    @pytest.mark.parametrize(
        "ix, resid, resid_str",
        [
            (0, ("ALA", 1, "A"), "ALA1.A"),
            (4, ("CYS", 5, "A"), "CYS5.A"),
            (6, ("GLU", 7, "A"), "GLU7.A"),
            (9, ("ILE", 10, "A"), "ILE10.A"),
            (19, ("VAL", 20, "A"), "VAL20.A"),
            (-1, ("VAL", 20, "A"), "VAL20.A"),
        ],
    )
    def test_getitem(self, residues, ix, resid, resid_str):
        rg = ResidueGroup(residues)
        resid = ResidueId(*resid)
        assert rg[ix] == rg[resid]
        assert rg[ix] == rg[resid_str]

    def test_getitem_keyerror(self):
        rg = ResidueGroup([])
        with pytest.raises(KeyError, match="Expected a ResidueId, int, or str"):
            rg[True]
        with pytest.raises(KeyError, match="Expected a ResidueId, int, or str"):
            rg[1.5]

    def test_select(self, protein_mol):
        rg = protein_mol.residues
        assert rg.select(rg.name == "LYS").n_residues == 16
        assert rg.select(rg.number == 300).n_residues == 1
        assert rg.select(rg.number == 1).n_residues == 0
        assert rg.select(rg.chain == "B").n_residues == 90
        # and
        assert rg.select((rg.chain == "B") & (rg.name == "ALA")).n_residues == 7
        # or
        assert rg.select((rg.chain == "B") | (rg.name == "ALA")).n_residues == 110
        # xor
        assert rg.select((rg.chain == "B") ^ (rg.name == "ALA")).n_residues == 103
        # not
        assert rg.select(~(rg.chain == "B")).n_residues == 212

    def test_select_sameas_getitem(self, protein_mol):
        rg = protein_mol.residues
        sel = rg.select((rg.name == "LYS") & (rg.number == 49))[0]
        assert sel.resid is rg["LYS49.A"].resid
