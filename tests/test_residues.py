from typing import TYPE_CHECKING

import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from prolif.residue import Residue, ResidueGroup, ResidueId

if TYPE_CHECKING:
    from prolif.molecule import Molecule


class TestResidueId:
    @pytest.mark.parametrize(
        ("name", "number", "segment", "chain"),
        [
            ("ALA", None, None, None),
            ("ALA", 1, None, None),
            ("ALA", 0, None, None),
            ("ALA", None, None, "B"),
            ("ALA", 1, None, "B"),
            (None, 1, None, "B"),
            (None, None, None, "B"),
            (None, 1, None, None),
            (None, None, None, None),
            ("ALA", 1, 2, "B"),
            (None, 1, 2, "B"),
            (None, None, 2, "B"),
            (None, 1, 2, None),
            (None, None, 2, None),
            ("", None, None, None),
            (None, None, None, ""),
            ("", None, None, ""),
        ],
    )
    def test_init(
        self,
        name: str | None,
        number: int | None,
        segment: int | None,
        chain: str | None,
    ) -> None:
        resid = ResidueId(name, number, segment, chain)
        name = name or ""
        number = number or 0
        segment = segment or 0
        chain = chain or ""
        assert resid.name == name
        assert resid.number == number
        assert resid.segment == segment
        assert resid.chain == chain

    @pytest.mark.parametrize(
        ("name", "number", "segment", "chain"),
        [
            ("ALA", None, None, None),
            ("ALA", 1, None, None),
            ("ALA", 0, None, None),
            ("ALA", None, None, "B"),
            ("ALA", 1, None, "B"),
            ("DA", 1, None, None),
            (None, 1, None, "B"),
            (None, None, None, "B"),
            (None, 1, None, None),
            (None, None, None, None),
            ("ALA", 1, 2, "B"),
            (None, 1, 2, "B"),
            (None, None, 2, "B"),
            (None, 1, 2, None),
            (None, None, 2, None),
            ("", None, None, None),
            (None, None, None, ""),
            ("", None, None, ""),
        ],
    )
    def test_from_atom(
        self,
        name: str | None,
        number: int | None,
        segment: int | None,
        chain: str | None,
    ) -> None:
        atom = Chem.Atom(1)
        mi = Chem.AtomPDBResidueInfo()
        if name is not None:
            mi.SetResidueName(name)
        if number is not None:
            mi.SetResidueNumber(number)
        if segment is not None:
            mi.SetSegmentNumber(segment)
        if chain is not None:
            mi.SetChainId(chain)
        atom.SetMonomerInfo(mi)
        resid = ResidueId.from_atom(atom)
        name = name or ""
        number = number or 0
        segment = segment or 0
        chain = chain or ""
        assert resid.name == name
        assert resid.number == number
        assert resid.segment == segment
        assert resid.chain == chain

    def test_from_atom_no_mi(self) -> None:
        atom = Chem.Atom(1)
        resid = ResidueId.from_atom(atom)
        assert resid.name == ""
        assert resid.number == 0
        assert resid.segment == 0
        assert resid.chain == ""

    @pytest.mark.parametrize(
        ("resid_str", "expected"),
        [
            ("ALA", ("ALA",)),
            ("ALA1", ("ALA", 1)),
            ("ALA.B", ("ALA", 0, 0, "B")),
            ("ALA1.B", ("ALA", 1, 0, "B")),
            ("1.B", ("", 1, 0, "B")),
            (".B", ("", 0, 0, "B")),
            (".0", ("", 0, 0, "0")),
            ("1", ("", 1, 0, None)),
            ("", ("", 0, 0, None)),
            ("DA2.A", ("DA", 2, 0, "A")),
            ("DA2", ("DA", 2)),
            ("DA", ("DA",)),
            ("8NU", ("8NU",)),
            ("8NU1", ("8NU", 1)),
            ("8NU1.A", ("8NU", 1, 0, "A")),
            ("8NU.A", ("8NU", 0, 0, "A")),
            ("TIP3", ("TIP3",)),
            ("TIP3.A", ("TIP3", 0, 0, "A")),
            ("TIP31", ("TIP3", 1)),
            ("TIP31.A", ("TIP3", 1, 0, "A")),
            ("TIP31.1", ("TIP3", 1, 0, "1")),
            ("T3P1", ("T3P", 1)),
            ("HISE1", ("HISE", 1)),
            ("H2O1", ("H2O", 1)),
            ("K123.A", ("K", 123, 0, "A")),
            ("K  123.A", ("K", 123, 0, "A")),
            ("MN123.A", ("MN", 123, 0, "A")),
            ("MN 123.A", ("MN", 123, 0, "A")),
        ],
    )
    def test_from_string(self, resid_str: str, expected: tuple) -> None:
        resid = ResidueId.from_string(resid_str)
        assert resid == ResidueId(*expected)

    def test_eq(self) -> None:
        name, number, chain = "ALA", 1, "A"
        res1 = ResidueId(name, number, chain=chain)
        res2 = ResidueId(name, number, chain=chain)
        assert res1 == res2
        assert res1 != 42

    @pytest.mark.parametrize(
        ("res1", "res2"),
        [
            ("ALA1.A", "ALA1.B"),
            ("ALA2.A", "ALA3.A"),
            ("ALA4.A", "ALA1.B"),
            ("ALA1", "ALA2"),
            ("ALA1", "ALA1.A"),
        ],
    )
    def test_lt(self, res1: str, res2: str) -> None:
        resid1 = ResidueId.from_string(res1)
        resid2 = ResidueId.from_string(res2)
        assert resid1 < resid2

    @pytest.mark.parametrize(
        ("resid_str", "expected"),
        [
            ("ALA1.A", "ResidueId('ALA', 1, 0, 'A')"),
            ("DA2.B", "ResidueId('DA', 2, 0, 'B')"),
            ("HIS3", "ResidueId('HIS', 3, 0)"),
            ("ALA", "ResidueId('ALA', 0, 0)"),
            ("ALA.B", "ResidueId('ALA', 0, 0, 'B')"),
            ("1.B", "ResidueId('', 1, 0, 'B')"),
            ("1", "ResidueId('', 1, 0)"),
            (".B", "ResidueId('', 0, 0, 'B')"),
            ("", "ResidueId('', 0, 0)"),
            ("ALA1:2.A", "ResidueId('ALA', 1, 2, 'A')"),
            ("ALA1:2", "ResidueId('ALA', 1, 2)"),
            ("ALA:2", "ResidueId('ALA', 0, 2)"),
            ("1:2.A", "ResidueId('', 1, 2, 'A')"),
            ("1:2", "ResidueId('', 1, 2)"),
            (":2.A", "ResidueId('', 0, 2, 'A')"),
            (":2", "ResidueId('', 0, 2)"),
        ],
    )
    def test_repr(self, resid_str: str, expected: str) -> None:
        resid = ResidueId.from_string(resid_str)
        assert repr(resid) == expected

    @pytest.mark.parametrize(
        ("resid_str", "expected"),
        [
            ("ALA1.A", "ALA1:0.A"),
            ("DA2.B", "DA2:0.B"),
            ("HIS3", "HIS3:0"),
            ("ALA", "ALA0:0"),
            ("ALA.B", "ALA0:0.B"),
            ("1.B", "1:0.B"),
            ("1", "1:0"),
            (".B", "0:0.B"),
            ("", "0:0"),
            ("ALA1:2.A", "ALA1:2.A"),
            ("ALA1:2", "ALA1:2"),
            ("ALA:2", "ALA0:2"),
            ("1:2.A", "1:2.A"),
            ("1:2", "1:2"),
            (":2.A", "0:2.A"),
            (":2", "0:2"),
        ],
    )
    def test_str(self, resid_str: str, expected: str) -> None:
        resid = ResidueId.from_string(resid_str)
        assert str(resid) == expected


class TestResidue(pytest.BaseTestMixinRDKitMol):  # type: ignore[name-defined]
    @pytest.fixture(scope="class")
    def mol(self) -> Residue:
        mol = Chem.MolFromSequence("A")
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol)
        return Residue(mol)

    def test_init(self, mol: Residue) -> None:
        resid = ResidueId.from_atom(mol.GetAtomWithIdx(0))
        assert mol.resid == resid

    def test_str(self, mol: Residue) -> None:
        assert str(mol) == "ALA1:0.A"


class TestResidueGroup:
    @pytest.fixture(scope="class")
    def residues(self) -> list[Residue]:
        sequence = "ARNDCQEGHILKMFPSTWYV"
        protein = Chem.MolFromSequence(sequence)
        return [Residue(res) for res in Chem.SplitMolByPDBResidues(protein).values()]

    def test_init(self, residues: list[Residue]) -> None:
        rg = ResidueGroup(residues)
        for rg_res, res in zip(rg._residues, residues, strict=True):
            assert rg_res is res
        for (resid, rg_res), res in zip(rg.items(), residues, strict=True):
            assert rg_res is res
            assert resid is rg_res.resid
        resinfo = [(r.resid.name, r.resid.number, r.resid.chain) for r in residues]
        name, number, chain = zip(*resinfo, strict=True)
        assert_equal(rg.name, name)
        assert_equal(rg.number, number)
        assert_equal(rg.chain, chain)

    def test_init_empty(self) -> None:
        rg = ResidueGroup([])
        assert_equal(rg.name, [])
        assert_equal(rg.number, [])
        assert_equal(rg.chain, [])
        assert_equal(rg._residues, [])
        assert rg.data == {}

    def test_n_residues(self, residues: list[Residue]) -> None:
        rg = ResidueGroup(residues)
        assert rg.n_residues == len(rg)
        assert rg.n_residues == 20

    @pytest.mark.parametrize(
        ("ix", "resid", "resid_str"),
        [
            (0, ("ALA", 1, 0, "A"), "ALA1:0.A"),
            (4, ("CYS", 5, 0, "A"), "CYS5:0.A"),
            (6, ("GLU", 7, 0, "A"), "GLU7:0.A"),
            (9, ("ILE", 10, 0, "A"), "ILE10:0.A"),
            (19, ("VAL", 20, 0, "A"), "VAL20:0.A"),
            (-1, ("VAL", 20, 0, "A"), "VAL20:0.A"),
        ],
    )
    def test_getitem(
        self,
        residues: list[Residue],
        ix: int,
        resid: tuple,
        resid_str: str,
    ) -> None:
        rg = ResidueGroup(residues)
        resid_obj = ResidueId(*resid)
        assert rg[ix] == rg[resid_obj]
        assert rg[ix] == rg[resid_str]

    def test_getitem_keyerror(self) -> None:
        rg = ResidueGroup([])
        with pytest.raises(KeyError, match="Expected a ResidueId, int, or str"):
            rg[True]
        with pytest.raises(KeyError, match="Expected a ResidueId, int, or str"):
            rg[1.5]  # type: ignore[index]

    def test_select(self, protein_mol: "Molecule") -> None:
        rg = protein_mol.residues
        assert rg.select(rg.name == "TYR").n_residues == 6
        assert rg.select(rg.number == 109).n_residues == 1
        assert rg.select(rg.number == 1).n_residues == 0
        assert rg.select(rg.chain == "A").n_residues == 36
        # and
        assert rg.select((rg.chain == "A") & (rg.name == "TYR")).n_residues == 5
        # or
        assert rg.select((rg.chain == "A") | (rg.name == "TYR")).n_residues == 37
        # xor
        assert rg.select((rg.chain == "A") ^ (rg.name == "TYR")).n_residues == 32
        # not
        assert rg.select(~(rg.chain == "A")).n_residues == 20

    def test_select_sameas_getitem(self, protein_mol: "Molecule") -> None:
        rg = protein_mol.residues
        sel = rg.select((rg.name == "TYR") & (rg.number == 38))[0]
        assert sel.resid is rg["TYR38.A"].resid

    def test_resnumber_max_value(self) -> None:
        resnums = [0, 65535, 92805, 999999]
        residues = []
        for resnum in resnums:
            mi = Chem.AtomPDBResidueInfo()
            mi.SetResidueName("ALA")
            mi.SetResidueNumber(resnum)
            mi.SetChainId("A")
            mol = Chem.MolFromSequence("A")
            for atom in mol.GetAtoms():
                atom.SetMonomerInfo(mi)
            residues.append(Residue(mol))
        rg = ResidueGroup(residues)
        assert rg.number[-1] == 999999
