from contextlib import nullcontext
from unittest.mock import Mock

import gemmi
import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity

from prolif.datafiles import datapath
from prolif.io.cif import cif_template_reader
from prolif.io.molecule_standardizer import MoleculeStandardizer
from prolif.io.template_engine import (
    CIFTemplateEngine,
    RDKitMolTemplateEngine,
    _assign_intra_props_lone_H,
    strip_bonds,
)
from prolif.io.xml import parse_altnames
from prolif.molecule import Molecule
from prolif.residue import Residue


@pytest.fixture(scope="module")
def xml_test_data() -> str:
    """Fixture to load the XML test data for altnames parsing."""
    return (
        datapath / "molecule_standardizer/templates/standard_aa_name.xml"
    ).read_text()


def test_cif_template_reader() -> None:
    """Test the CIF template reader returns a gemmi Document with correct blocks."""

    result = cif_template_reader(datapath / "molecule_standardizer/templates/TPO.cif")

    assert isinstance(result, gemmi.cif.Document)
    assert len(result) == 1
    block = result.find_block("TPO")
    assert block is not None
    assert block.name == "TPO"

    # Check we can access bond data
    bond_table = block.find(
        "_chem_comp_bond.",
        ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"],
    )
    assert len(bond_table) > 0

    # Test with standard_aa.cif (multi-block)
    result_aa = cif_template_reader(
        datapath / "molecule_standardizer/templates/standard_aa.cif"
    )
    assert isinstance(result_aa, gemmi.cif.Document)
    assert len(result_aa) == 26  # 20 standard + variants
    assert result_aa.find_block("ALA") is not None


def test_xml_parse_altnames(xml_test_data: str) -> None:
    """Test the XML parsing of alternative names for residues and atoms."""

    resname_aliases, atomname_aliases = parse_altnames(xml_test_data)
    assert isinstance(resname_aliases, dict)
    assert isinstance(atomname_aliases, dict)


class TestMoleculeStandardizer:
    """Test class for MoleculeStandardizer functions."""

    @pytest.fixture(scope="class")
    def input_path(self) -> str:
        """Return the path to the input file."""
        return str(datapath / "molecule_standardizer/tpo.pdb")

    @pytest.fixture(scope="class")
    def input_rdmol(self, input_path: str, request: pytest.FixtureRequest) -> Chem.Mol:
        """Return the RDKit Molecule object for the input file."""
        input_mol = Chem.MolFromPDBFile(
            input_path, removeHs=getattr(request, "param", False)
        )
        DetermineConnectivity(input_mol, useHueckel=True)
        for atm in input_mol.GetAtoms():
            atm.SetNoImplicit(False)
        return input_mol

    @pytest.fixture(scope="class")
    def input_molecule(self, input_rdmol: Chem.Mol) -> Molecule:
        """Return the Molecule object for the input file."""
        return Molecule.from_rdkit(input_rdmol)

    @pytest.fixture(scope="class")
    def input_topology(
        self, request: pytest.FixtureRequest
    ) -> str | Chem.Mol | Molecule:
        """Return the input topology for the standardize method."""
        return request.getfixturevalue(request.param)  # type: ignore[no-any-return]

    @pytest.fixture(scope="class")
    def ben_path(self) -> str:
        """Return a Molecule object for the BEN path."""
        ben_path = datapath / "molecule_standardizer/ben_test.pdb"
        return str(ben_path)

    @pytest.fixture(scope="class")
    def cif_templates(self) -> list[gemmi.cif.Document]:
        """Return CIF templates for testing."""
        return [
            cif_template_reader(datapath / "molecule_standardizer/templates/TPO.cif"),
            cif_template_reader(datapath / "molecule_standardizer/templates/ACE.cif"),
            cif_template_reader(datapath / "molecule_standardizer/templates/NME.cif"),
            cif_template_reader(datapath / "molecule_standardizer/templates/BEN.cif"),
        ]

    @pytest.fixture(scope="class")
    def mol_templates(self) -> list[tuple[str, Chem.Mol]]:
        """Return RDKit Mol templates for testing."""
        return [
            ("XYZ", Chem.MolFromSmiles("C1=CC=CC=C1")),
            ("ABC", Chem.MolFromSmiles("C(C(=O)O)N")),
        ]

    @pytest.fixture(scope="class")
    def all_templates(
        self,
        cif_templates: list[gemmi.cif.Document],
        mol_templates: list[tuple[str, Chem.Mol]],
    ) -> list[gemmi.cif.Document | tuple[str, Chem.Mol]]:
        """Return combined templates for testing."""
        return [*mol_templates, *cif_templates]

    @pytest.fixture(scope="class")
    def any_templates(
        self, request: pytest.FixtureRequest
    ) -> list[gemmi.cif.Document] | list[tuple[str, Chem.Mol]]:
        return request.getfixturevalue(request.param)  # type: ignore[no-any-return]

    @pytest.fixture(scope="class")
    def hsd_residue(self) -> Residue:
        """Return a HID residue for testing."""
        protein_path = datapath / "implicitHbond/receptor_hsd.pdb"
        input_mol = Chem.MolFromPDBFile(str(protein_path))
        return Molecule.from_rdkit(input_mol).residues[106]

    @pytest.fixture(scope="class")
    def mol_missing_atom(self) -> Molecule:
        """Return a molecule missing side chain atoms for testing."""
        protein_path = datapath / "implicitHbond/receptor_hsd.pdb"
        input_mol = Chem.MolFromPDBFile(str(protein_path))
        return Molecule(Molecule.from_rdkit(input_mol).residues[0])

    @pytest.fixture(scope="class")
    def standardizer_default(self) -> MoleculeStandardizer:
        """Return a default instance of the MoleculeStandardizer class."""
        return MoleculeStandardizer()

    @pytest.mark.parametrize(
        ("templates", "expected_context"),
        [
            (None, nullcontext()),
            ([], nullcontext()),
            (
                [("ACE", Chem.MolFromSmiles("CC=O"))],
                nullcontext(),
            ),
            (
                12345,
                pytest.raises(TypeError),
            ),
            (
                ["invalid_format"],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_initialization(self, templates, expected_context) -> None:  # type: ignore
        """Test the initialization of the MoleculeStandardizer class."""

        with expected_context:
            standardizer = MoleculeStandardizer(templates)

        # Exclude the TypeError case
        if isinstance(expected_context, nullcontext):
            assert isinstance(standardizer.engines, dict)
            # should always have STANDARD_AA at minimum
            assert "ALA" in standardizer.engines

    def test_convert_to_standard_resname(self) -> None:
        """Test the conversion of residue names to standard names."""

        # warning with unknown forcefield name
        with pytest.warns(
            UserWarning,
            match=r"Could not guess the forcefield based on the residue names\. "
            r"CYS is assigned to neutral CYS \(charge = 0\)\.",
        ):
            MoleculeStandardizer.convert_to_standard_resname(
                resname="CYS", forcefield_name="unknown"
            )

        # gromos's CYS -> CYX
        resname = MoleculeStandardizer.convert_to_standard_resname(
            resname="CYS", forcefield_name="gromos"
        )
        assert resname == "CYX"

        # Test with an unknown residue name
        resname = MoleculeStandardizer.convert_to_standard_resname(
            resname="HSD", forcefield_name="unknown"
        )
        assert resname == "HID"

    def test_n_residue_heavy_atoms(self, input_molecule: Molecule) -> None:
        """Test the counting of heavy atoms in residues."""

        # Test with a Molecule object
        n_heavy_atoms = MoleculeStandardizer.n_residue_heavy_atoms(
            input_molecule.residues[1]
        )
        assert isinstance(n_heavy_atoms, int)
        assert n_heavy_atoms == 11

    def test_engine_n_heavy_atoms(
        self, cif_templates: list[gemmi.cif.Document]
    ) -> None:
        """Test the counting of heavy atoms in template engines."""
        # default engines
        standardizer = MoleculeStandardizer()
        assert standardizer.engines["ALA"].n_heavy_atoms() == 5

        # CIF engine
        tpo_doc = cif_templates[0]
        engine = CIFTemplateEngine("TPO", tpo_doc.find_block("TPO"))
        assert engine.n_heavy_atoms() == 11

        # RDKit Mol engine
        mol_engine = RDKitMolTemplateEngine("XYZ", Chem.MolFromSmiles("C1=CC=CC=C1"))
        assert mol_engine.n_heavy_atoms() == 6

    @pytest.mark.parametrize(
        "any_templates", ["mol_templates", "cif_templates"], indirect=True
    )
    def test_engine_exc_shows_residue_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        any_templates: list[gemmi.cif.Document] | list[tuple[str, Chem.Mol]],
    ) -> None:
        mock = Mock(side_effect=SystemError("test"))
        monkeypatch.setattr("prolif.io.template_engine.CIFTemplateEngine.apply", mock)
        monkeypatch.setattr(
            "prolif.io.template_engine.RDKitMolTemplateEngine.apply", mock
        )
        standardizer = MoleculeStandardizer(templates=any_templates)

        with pytest.raises(
            ValueError,
            match=r"Could not apply template for residue ALA1.A: test",
        ):
            standardizer(Chem.MolFromSequence("AA"))

    def test_fix_molecule_bond_orders_cif(
        self,
        input_molecule: Molecule,
        cif_templates: list[gemmi.cif.Document],
    ) -> None:
        """Test fixing bond orders using CIF template engine."""

        standardizer = MoleculeStandardizer(templates=cif_templates)
        engine = standardizer.engines["TPO"]
        fixed_mol = engine.apply(input_molecule.residues[1])
        assert isinstance(fixed_mol, Residue)

    def test_fix_molecule_bond_orders_rdkit(
        self,
        input_molecule: Molecule,
    ) -> None:
        """Test fixing bond orders using RDKit Mol template engine."""

        tpo_mol = Chem.MolFromSmiles("C[C@H]([C@@H](C(=O))N)OP(=O)(O)O")
        engine = RDKitMolTemplateEngine("TPO", tpo_mol)
        fixed_mol = engine.apply(input_molecule.residues[1])
        assert isinstance(fixed_mol, Residue)

    def test_fix_molecule_bond_orders_equivalence(
        self,
        input_molecule: Molecule,
        cif_templates: list[gemmi.cif.Document],
    ) -> None:
        """Test that CIF and RDKit engines produce equivalent results."""

        # CIF engine
        cif_standardizer = MoleculeStandardizer(templates=cif_templates)
        cif_engine = cif_standardizer.engines["TPO"]
        fixed_cif = cif_engine.apply(input_molecule.residues[1])

        # RDKit engine
        tpo_mol = Chem.MolFromSmiles("C[C@H]([C@@H](C(=O))N)OP(=O)(O)O")
        rdkit_engine = RDKitMolTemplateEngine("TPO", tpo_mol)
        fixed_rdkit = rdkit_engine.apply(input_molecule.residues[1])

        # check two residues are equal and bond orders are fixed
        for at1, at2 in zip(fixed_cif.GetAtoms(), fixed_rdkit.GetAtoms(), strict=True):
            assert at1.GetSymbol() == at2.GetSymbol()
            assert at1.GetIsAromatic() == at2.GetIsAromatic()
            assert at1.GetTotalDegree() == at2.GetTotalDegree()

        for bond1 in fixed_cif.GetBonds():
            bond2 = fixed_rdkit.GetBondBetweenAtoms(
                bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()
            )
            assert bond1.GetBondType() == bond2.GetBondType()

    def test_fix_molecule_bond_orders_no_template(
        self,
        input_molecule: Molecule,
    ) -> None:
        """Test: not found template for residue (ACE) in default templates."""
        standardizer = MoleculeStandardizer()
        with pytest.raises(
            ValueError,
            match=r"Residue \{'ACE'\} is not a standard residue or "
            r"not in the templates\. Please provide a custom template\.",
        ):
            standardizer(input_molecule)

    @pytest.mark.parametrize(
        ("resnames", "expected"),
        [
            ({"HSD"}, "charmm"),
            ({"NASP"}, "amber"),
            ({"ASN1"}, "gromos"),
            ({"HISD"}, "oplsaa"),
            ({"CYS"}, "unknown"),
        ],
    )
    def test_forcefield_guesser(self, resnames, expected) -> None:  # type: ignore
        """Test the forcefield guesser."""

        # Test with a known forcefield
        forcefield = MoleculeStandardizer.forcefield_guesser(resnames)
        assert forcefield == expected

    def test_standardize_no_valid_templates(
        self,
        input_molecule: Molecule,
        standardizer_default: MoleculeStandardizer,
    ) -> None:
        """Test the standardization of a molecule with no valid templates."""
        with pytest.raises(
            ValueError,
            match=r"Residue \{'ACE'\} is not a standard residue or "
            r"not in the templates\. Please provide a custom template\.",
        ):
            standardizer_default(input_molecule)

    def test_standardize_missing_heavy_atoms(
        self,
        mol_missing_atom: Molecule,
        standardizer_default: MoleculeStandardizer,
    ) -> None:
        """Test the standardization of a molecule with missing heavy atoms."""

        # Test standardizing a molecule with missing heavy atoms
        with pytest.warns(
            UserWarning,
            match=r"Residue MET1\.A has a different number of heavy atoms "
            r"than the standard residue\. This may affect H-bond detection\.",
        ):
            protein_mol = standardizer_default(mol_missing_atom)

        assert isinstance(protein_mol, Molecule)
        assert len(protein_mol.residues) == len(mol_missing_atom.residues)

    def test_standardize_wrong_format(
        self,
        standardizer_default: MoleculeStandardizer,
    ) -> None:
        """Test the standardization of a molecule with wrong format."""

        # Test with a wrong format (string)
        with pytest.raises(
            TypeError,
            match=r"input_topology must be a string \(path to a PDB file\) or "
            r"a prolif Molecule instance\.",
        ):
            standardizer_default("invalid_format")

    @pytest.mark.parametrize(
        "input_topology", ["input_path", "input_molecule", "input_rdmol"], indirect=True
    )
    def test_standardize(
        self,
        all_templates: list[gemmi.cif.Document | tuple[str, Chem.Mol]],
        input_topology: str | Chem.Mol | Molecule,
        input_molecule: Molecule,
        ben_path: str,
        hsd_residue: Residue,
    ) -> None:
        """Test the standardization of a molecule."""

        # Set up standardizer with templates
        standardizer = MoleculeStandardizer(templates=all_templates)

        # Test standardizing a molecule
        protein_mol = standardizer(input_topology)
        assert isinstance(protein_mol, Molecule)
        assert len(protein_mol.residues) == len(input_molecule.residues)

        # Test with a residue
        residue_mol = standardizer(Molecule(hsd_residue))
        assert len(residue_mol.residues) == 1
        assert str(residue_mol.residues[0].resid) == "HID109.A"

        # Test with a BENZAMIDINE residue (from path)
        ben_mol = standardizer(ben_path)
        assert isinstance(ben_mol, Molecule)
        all_bonds_info = []
        for bond in ben_mol.residues[0].GetBonds():
            if (bond.GetBeginAtom().GetAtomicNum() == 1) or (
                bond.GetEndAtom().GetAtomicNum() == 1
            ):
                # skip bonds involving hydrogen atoms
                continue

            all_bonds_info.append(
                (
                    {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()},
                    str(bond.GetBondType()),
                )
            )

        all_bonds_info.sort()
        assert_equal(
            [
                ({1, 3}, "AROMATIC"),
                ({1, 8}, "AROMATIC"),
                ({0, 1}, "UNSPECIFIED"),
                ({3, 5}, "AROMATIC"),
                ({5, 6}, "AROMATIC"),
                ({6, 7}, "AROMATIC"),
                ({7, 8}, "AROMATIC"),
                ({0, 2}, "DOUBLE"),
                ({0, 4}, "SINGLE"),
            ],
            all_bonds_info,
        )

    def test_assign_intra_props_lone_H(self) -> None:
        """
        Test the assignment of intra properties for a residue with a lone hydrogen.
        """

        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # type: ignore

        strip_mol = strip_bonds(mol)

        with Chem.RWMol(strip_mol) as em:
            # add a existing bond before testing the function
            em.AddBond(0, 1, order=Chem.BondType.SINGLE)
            em.AddBond(0, 2, order=Chem.BondType.SINGLE)

            # test the function
            em_fixed = _assign_intra_props_lone_H(em)

        for bond1, bond2 in zip(mol.GetBonds(), em_fixed.GetBonds(), strict=True):
            assert bond1.GetBondType() == bond2.GetBondType()
