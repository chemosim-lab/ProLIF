import pandas as pd
import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity

from prolif.datafiles import datapath
from prolif.io.cif import cif_parser_lite, cif_template_reader
from prolif.io.protein_helper import (
    ProteinHelper,
    _assign_intra_props_lone_H,
    strip_bonds,
)
from prolif.io.xml import parse_altnames
from prolif.molecule import Molecule
from prolif.residue import Residue


@pytest.fixture(scope="module")
def CIF() -> str:
    """Fixture to load the CIF file for testing."""
    return (datapath / "TPO.cif").read_text()


@pytest.fixture(scope="module")
def STANDARD_AA() -> str:
    """Fixture to load the standard amino acid CIF file for testing."""
    return (datapath / "standard_aa.cif").read_text()


def test_cif_parser_lite(CIF: str, STANDARD_AA: str) -> None:
    """Test the CIF parser for a valid CIF file and a standard amino acid CIF file."""
    # Test with a valid CIF file
    result = cif_parser_lite(CIF)

    assert isinstance(result, dict)
    assert isinstance(result["TPO"]["_pdbx_chem_comp_synonyms"], dict)
    assert result["TPO"]["_pdbx_chem_comp_synonyms"]["name"] == "PHOSPHONOTHREONINE"
    assert isinstance(result["TPO"]["_pdbx_chem_comp_descriptor"], pd.DataFrame)

    # Test with a standard amino acid CIF file
    result_2 = cif_parser_lite(STANDARD_AA)

    assert isinstance(result_2, dict)


def test_cif_template_reader() -> None:
    """Test the CIF template reader."""

    # Test reading the standard amino acid CIF template
    result = cif_template_reader(datapath / "TPO.cif")

    # Check the result structure
    assert isinstance(result["TPO"]["_pdbx_chem_comp_synonyms"], dict)
    assert result["TPO"]["_pdbx_chem_comp_synonyms"]["name"] == "PHOSPHONOTHREONINE"
    assert isinstance(result["TPO"]["_pdbx_chem_comp_descriptor"], pd.DataFrame)


@pytest.fixture(scope="module")
def XML_TEST_DATA() -> str:
    """Fixture to load the XML test data for altnames parsing."""
    return (datapath / "standard_aa_name.xml").read_text()


def test_xml_parse_altnames(XML_TEST_DATA: str) -> None:
    """Test the XML parsing of alternative names for residues and atoms."""

    resname_aliases, atomname_aliases = parse_altnames(XML_TEST_DATA)
    assert isinstance(resname_aliases, dict)
    assert isinstance(atomname_aliases, dict)


class TestProteinHelper:
    """Test class for protein helper functions."""

    @pytest.fixture(scope="class")
    def INPUT_PATH(self) -> str:
        """Return the path to the input file."""
        return str(datapath / "tpo.pdb")

    @pytest.fixture(scope="class", params=[True, False])
    def INPUT_MOL(self, INPUT_PATH: str, request) -> Molecule:  # type: ignore
        """Return the Molecule object for the input file."""
        input_mol = Chem.MolFromPDBFile(INPUT_PATH, removeHs=request.param)
        DetermineConnectivity(input_mol, useHueckel=True)
        for atm in input_mol.GetAtoms():
            atm.SetNoImplicit(False)  # set no implicit to False
        return Molecule.from_rdkit(input_mol)

    @pytest.fixture(scope="class")
    def BEN_PATH(self) -> str:
        """Return a Molecule object for the BEN path."""
        ben_path = datapath / "ben_test.pdb"
        return str(ben_path)

    @pytest.fixture(scope="class")
    def CUSTOM_TEMPLATE(self) -> dict:
        """Return a custom template for testing."""

        tpo_template = cif_template_reader(datapath / "TPO.cif")
        ace_template = cif_template_reader(datapath / "ACE.cif")
        nme_template = cif_template_reader(datapath / "NME.cif")
        ben_template = cif_template_reader(datapath / "BEN.cif")

        return {
            "XYZ": {
                "name": "XYZ",
                "SMILES": "C1=CC=CC=C1",
            },
            "ABC": {
                "name": "ABC",
                "SMILES": "C(C(=O)O)N",
            },
            "TPO": tpo_template["TPO"],
            "ACE": ace_template["ACE"],
            "NME": nme_template["NME"],
            "BEN": ben_template["BEN"],
        }

    @pytest.fixture(scope="class")
    def HSD_RESIDUE(self) -> Residue:
        """Return a HID residue for testing."""
        protein_path = datapath / "implicitHbond/1s2g__1__1.A_2.C__1.D/receptor_hsd.pdb"
        input_mol = Chem.MolFromPDBFile(str(protein_path))
        return Molecule.from_rdkit(input_mol).residues[106]

    @pytest.fixture(scope="class")
    def MOL_MISSING_ATOM(self) -> Molecule:
        """Return a molecule missing side chain atoms for testing."""
        protein_path = datapath / "implicitHbond/1s2g__1__1.A_2.C__1.D/receptor_hsd.pdb"
        input_mol = Chem.MolFromPDBFile(str(protein_path))
        return Molecule(Molecule.from_rdkit(input_mol).residues[0])

    @pytest.mark.parametrize(
        ("templates", "expected_context"),
        [
            (None, nullcontext()),
            ([], nullcontext()),
            ({"ACE": {"SMILES": "CC=O"}}, nullcontext()),
            (12345, pytest.raises(TypeError, match=r"Templates must be a dict, a list of dicts or None\.")),
            (["invalid_format"], pytest.raises(TypeError, match=r"Templates must be a dict, a list of dicts or None\.")),
            ({"RES": {"name": "BEN", "SMILES": "NC(=N)c1ccccc1"}}, pytest.warns(UserWarning, match=r"Align the template name \(BEN\) with \(RES\)\.")),
        ],
    )
    def test_initialization(self, templates, expected_context) -> None:  # type: ignore
        """Test the initialization of the TestProteinHelper class."""

        with expected_context:
            protein_helper = ProteinHelper(templates)
        assert isinstance(protein_helper.templates, list)

    def test_convert_to_standard_resname(self) -> None:
        """Test the conversion of residue names to standard names."""

        # warning with unknown forcefield name
        with pytest.warns(
            UserWarning,
            match=r"Could not guess the forcefield based on the residue names\. "
            r"CYS is assigned to neutral CYS \(charge = 0\)\.",
        ):
            ProteinHelper.convert_to_standard_resname(
                resname="CYS", forcefield_name="unknown"
            )

        # gromos's CYS -> CYX
        resname = ProteinHelper.convert_to_standard_resname(
            resname="CYS", forcefield_name="gromos"
        )
        assert resname == "CYX"

        # Test with an unknown residue name
        resname = ProteinHelper.convert_to_standard_resname(
            resname="HSD", forcefield_name="unknown"
        )
        assert resname == "HID"

    def test_check_resnames(self, CUSTOM_TEMPLATE: dict) -> None:
        """Test the checking of residue names."""

        # Test with resnames not within a default templates
        with pytest.raises(
            ValueError,
            match=r"Residue \{'XYZ'\} is not a standard residue or "
            r"not in the templates\. Please provide a custom template\.",
        ):
            ProteinHelper.check_resnames({"ALA", "CYS", "XYZ"})

        # Test with resnames not with custom template
        with pytest.raises(
            ValueError,
            match=r"Residue \{'ALA'\} is not a standard residue or "
            r"not in the templates\. Please provide a custom template\.",
        ):
            ProteinHelper.check_resnames(
                {"ALA", "ABC", "XYZ"}, templates=[CUSTOM_TEMPLATE]
            )

    def test_n_residue_heavy_atoms(self, INPUT_MOL: Molecule) -> None:
        """Test the counting of heavy atoms in residues."""

        # Test with a Molecule object
        n_heavy_atoms = ProteinHelper.n_residue_heavy_atoms(INPUT_MOL.residues[1])
        assert isinstance(n_heavy_atoms, int)
        assert n_heavy_atoms == 11

    def test_n_template_residue_heavy_atoms(self, CUSTOM_TEMPLATE: dict) -> None:
        """Test the counting of heavy atoms in template residues."""

        # Test with a standard amino acid template
        template_n_heavy_atoms = ProteinHelper.n_template_residue_heavy_atoms()
        assert isinstance(template_n_heavy_atoms, dict)
        assert template_n_heavy_atoms["ALA"] == 5

        # Test with a custom (SMILES) template
        custom_template_n_heavy_atoms = ProteinHelper.n_template_residue_heavy_atoms(
            templates=[
                CUSTOM_TEMPLATE,
                {"XYZ": {"name": "XYZ", "test": "duplicate XYZ will be skiped."}},
            ]
        )
        assert isinstance(custom_template_n_heavy_atoms, dict)
        assert custom_template_n_heavy_atoms["XYZ"] == 6

    def test_fix_molecule_bond_orders(
        self, INPUT_MOL: Molecule, CUSTOM_TEMPLATE: dict
    ) -> None:
        """Test the fixing of bond orders in a Molecule object."""

        # Test with a Molecule object (using CIF template)
        fixed_mol = ProteinHelper.fix_molecule_bond_orders(
            INPUT_MOL.residues[1], templates=[CUSTOM_TEMPLATE]
        )
        assert isinstance(fixed_mol, Residue)

        # Test with a Molecule object (using SMILES template)
        fixed_mol_custom = ProteinHelper.fix_molecule_bond_orders(
            INPUT_MOL.residues[1],
            templates=[
                {
                    "TPO": {
                        "name": "TPO",
                        "SMILES": "C[C@H]([C@@H](C(=O))N)OP(=O)(O)O",
                    }
                }
            ],
        )
        assert isinstance(fixed_mol_custom, Residue)

        # check two residue are equal and bond orders are fixed
        for at1, at2 in zip(
            fixed_mol.GetAtoms(), fixed_mol_custom.GetAtoms(), strict=True
        ):
            assert at1.GetSymbol() == at2.GetSymbol()

        for bond1 in fixed_mol.GetBonds():
            bond2 = fixed_mol_custom.GetBondBetweenAtoms(
                bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()
            )
            assert bond1.GetBondType() == bond2.GetBondType()

        # Test: not found template for residue (TPO) in default templates
        with pytest.raises(
            ValueError,
            match=r"Failed to find template for residue: \'TPO\'",
        ):
            ProteinHelper.fix_molecule_bond_orders(INPUT_MOL.residues[1])

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
        forcefield = ProteinHelper.forcefield_guesser(resnames)
        assert forcefield == expected

    def test_standardize_protein_no_valid_templates(self, INPUT_MOL: Molecule) -> None:
        """Test the standardization of a protein molecule with no valid templates."""
        # Test with no valid templates
        protein_helper = ProteinHelper()
        with pytest.raises(
            ValueError,
            match=r"Residue \{'ACE'\} is not a standard residue or "
            r"not in the templates\. Please provide a custom template\.",
        ):
            protein_helper.standardize_protein(INPUT_MOL)

    def test_standardize_protein_missing_heavy_atoms(
        self, MOL_MISSING_ATOM: Molecule
    ) -> None:
        """Test the standardization of a protein molecule with missing heavy atoms."""
        # Set up protein helper
        protein_helper = ProteinHelper()

        # Test standardizing a molecule with missing heavy atoms
        with pytest.warns(
            UserWarning,
            match=r"Residue MET1\.A has a different number of heavy atoms "
            r"than the standard residue\. This may affect H-bond detection\.",
        ):
            protein_mol = protein_helper.standardize_protein(MOL_MISSING_ATOM)

        assert isinstance(protein_mol, Molecule)
        assert len(protein_mol.residues) == len(MOL_MISSING_ATOM.residues)

    def test_standardize_protein_wrong_format(self) -> None:
        """Test the standardization of a protein molecule with wrong format."""
        # Set up protein helper
        protein_helper = ProteinHelper()

        # Test with a wrong format (string)
        with pytest.raises(
            TypeError,
            match=r"input_topology must be a string \(path to a PDB file\) or "
            r"a prolif Molecule instance\.",
        ):
            protein_helper.standardize_protein("invalid_format")

    def test_standardize_protein(
        self,
        CUSTOM_TEMPLATE: dict,
        INPUT_MOL: Molecule,
        BEN_PATH: str,
        HSD_RESIDUE: Residue,
    ) -> None:
        """Test the standardization of a protein molecule."""

        # Set up protein helper with a custom template
        protein_helper = ProteinHelper(templates=CUSTOM_TEMPLATE)

        # Test standardizing a protein molecule
        protein_mol = protein_helper.standardize_protein(input_topology=INPUT_MOL)
        assert isinstance(protein_mol, Molecule)
        assert len(protein_mol.residues) == len(INPUT_MOL.residues)

        # Test with a residue
        residue_mol = protein_helper.standardize_protein(Molecule(HSD_RESIDUE))
        assert len(residue_mol.residues) == 1
        assert str(residue_mol.residues[0].resid) == "HID109.A"

        # Test with a BENZAMIDINE residue (from path)
        ben_mol = protein_helper.standardize_protein(BEN_PATH)
        assert isinstance(ben_mol, Molecule)
        all_bonds_info = []
        for bond in ben_mol.residues[0].GetBonds():
            if bond.GetBeginAtom().GetAtomicNum() == 1:
                # skip bonds involving hydrogen atoms
                continue

            if bond.GetEndAtom().GetAtomicNum() == 1:
                # skip bonds involving hydrogen atoms
                continue

            if bond.GetBeginAtomIdx() < bond.GetEndAtomIdx():
                all_bonds_info.append(
                    f"{bond.GetBeginAtomIdx()}_{bond.GetEndAtomIdx()}_"
                    f"{bond.GetBondType()!s}"
                )
            else:
                all_bonds_info.append(
                    f"{bond.GetEndAtomIdx()}_{bond.GetBeginAtomIdx()}_"
                    f"{bond.GetBondType()!s}"
                )
        all_bonds_info = sorted(all_bonds_info)
        assert_equal(
            [
                "0_1_UNSPECIFIED",
                "0_2_DOUBLE",
                "0_4_SINGLE",
                "1_3_AROMATIC",
                "1_8_AROMATIC",
                "3_5_AROMATIC",
                "5_6_AROMATIC",
                "6_7_AROMATIC",
                "7_8_AROMATIC",
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
