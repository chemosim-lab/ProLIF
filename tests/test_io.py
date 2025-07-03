from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem

from prolif.datafiles import datapath
from prolif.molecule import Molecule
from prolif.residue import Residue


@pytest.fixture(scope="module")
def CIF() -> str:
    """Fixture to load the CIF file for testing."""
    return Path(str(datapath / "TPO.cif")).read_text()


@pytest.fixture(scope="module")
def STANDARD_AA() -> str:
    """Fixture to load the standard amino acid CIF file for testing."""
    return Path(str(datapath / "standard_aa.cif")).read_text()


def test_cif_parser_lite(CIF: str, STANDARD_AA: str) -> None:
    from prolif.io.cif import cif_parser_lite

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
    from prolif.io.cif import cif_template_reader

    # Test reading the standard amino acid CIF template
    result = cif_template_reader(datapath / "TPO.cif")

    # Check the result structure
    assert isinstance(result["TPO"]["_pdbx_chem_comp_synonyms"], dict)
    assert result["TPO"]["_pdbx_chem_comp_synonyms"]["name"] == "PHOSPHONOTHREONINE"
    assert isinstance(result["TPO"]["_pdbx_chem_comp_descriptor"], pd.DataFrame)


@pytest.fixture(scope="module")
def XML_TEST_DATA() -> str:
    """Fixture to load the XML test data for altnames parsing."""
    return Path(str(datapath / "standard_aa_name.xml")).read_text()


def test_xml_parse_altnames(XML_TEST_DATA: str) -> None:
    """Test the XML parsing of alternative names for residues and atoms."""
    from prolif.io.xml import parse_altnames

    resname_aliases, atomname_aliases = parse_altnames(XML_TEST_DATA)
    assert isinstance(resname_aliases, dict)
    assert isinstance(atomname_aliases, dict)


class TestProteinHelper:
    """Test class for protein helper functions."""

    @pytest.fixture(scope="class")
    def INPUT_PATH(self) -> str:
        """Return the path to the input file."""
        return str(datapath / "tpo.pdb")

    @pytest.fixture(scope="class")
    def INPUT_MOL(self, INPUT_PATH: str) -> Molecule:
        """Return the Molecule object for the input file."""
        input_mol = Chem.MolFromPDBFile(INPUT_PATH)
        return Molecule.from_rdkit(input_mol)

    @pytest.fixture(scope="class")
    def BEN_MOL(self) -> Molecule:
        """Return a Molecule object for the BEN residue."""
        ben_path = datapath / "ben_test.pdb"
        input_mol = Chem.MolFromPDBFile(str(ben_path))
        return Molecule.from_rdkit(input_mol)

    @pytest.fixture(scope="class")
    def CUSTOM_TEMPLATE(self) -> dict:
        """Return a custom template for testing."""
        from prolif.io.cif import cif_template_reader

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

    def test_initialization(self, INPUT_PATH: str, INPUT_MOL: Molecule) -> None:
        """Test the initialization of the TestProteinHelper class."""
        from prolif.io.protein_helper import ProteinHelper

        # test reading with a path
        protein_helper = ProteinHelper(INPUT_PATH)
        assert isinstance(protein_helper, ProteinHelper)
        assert isinstance(protein_helper.protein_mol, Molecule)

        # test reading with a Molecule object
        protein_helper2 = ProteinHelper(INPUT_MOL)
        assert isinstance(protein_helper2, ProteinHelper)
        assert isinstance(protein_helper2.protein_mol, Molecule)

        # test type error
        with pytest.raises(TypeError):
            ProteinHelper(12345)  # type: ignore

    def test_convert_to_standard_resname(self) -> None:
        """Test the conversion of residue names to standard names."""
        from prolif.io.protein_helper import ProteinHelper

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
        from prolif.io.protein_helper import ProteinHelper

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
        from prolif.io.protein_helper import ProteinHelper

        # Test with a Molecule object
        n_heavy_atoms = ProteinHelper.n_residue_heavy_atoms(INPUT_MOL.residues[1])
        assert isinstance(n_heavy_atoms, int)
        assert n_heavy_atoms == 11

    def test_n_template_residue_heavy_atoms(self, CUSTOM_TEMPLATE: dict) -> None:
        """Test the counting of heavy atoms in template residues."""
        from prolif.io.protein_helper import ProteinHelper

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
        from prolif.io.protein_helper import ProteinHelper

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

        bond1_info, bond2_info = [], []
        for bond in fixed_mol.GetBonds():
            if bond.GetBeginAtomIdx() < bond.GetEndAtomIdx():
                bond1_info.append(
                    f"{bond.GetBeginAtomIdx()}_{bond.GetEndAtomIdx()}_"
                    f"{int(bond.GetBondTypeAsDouble())}"
                )
            else:
                bond1_info.append(
                    f"{bond.GetEndAtomIdx()}_{bond.GetBeginAtomIdx()}_"
                    f"{int(bond.GetBondTypeAsDouble())}"
                )

        for bond in fixed_mol_custom.GetBonds():
            if bond.GetBeginAtomIdx() < bond.GetEndAtomIdx():
                bond2_info.append(
                    f"{bond.GetBeginAtomIdx()}_{bond.GetEndAtomIdx()}_"
                    f"{int(bond.GetBondTypeAsDouble())}"
                )
            else:
                bond2_info.append(
                    f"{bond.GetEndAtomIdx()}_{bond.GetBeginAtomIdx()}_"
                    f"{int(bond.GetBondTypeAsDouble())}"
                )
        bond1_info = sorted(bond1_info)
        bond2_info = sorted(bond2_info)
        assert_equal(bond1_info, bond2_info)

        # Test: not found template for residue (TPO) in default templates
        with pytest.raises(
            ValueError,
            match=r"Failed to find template for residue: \'TPO\'",
        ):
            ProteinHelper.fix_molecule_bond_orders(INPUT_MOL.residues[1])

    def test_forcefield_guesser(self) -> None:
        """Test the forcefield guesser."""
        from prolif.io.protein_helper import ProteinHelper

        # Test with a known forcefield
        forcefield = ProteinHelper.forcefield_guesser({"HSD"})
        assert forcefield == "charmm"

        forcefield = ProteinHelper.forcefield_guesser({"NASP"})
        assert forcefield == "amber"

        forcefield = ProteinHelper.forcefield_guesser({"ASN1"})
        assert forcefield == "gromos"

        forcefield = ProteinHelper.forcefield_guesser({"HISD"})
        assert forcefield == "oplsaa"

        forcefield = ProteinHelper.forcefield_guesser({"CYS"})
        assert forcefield == "unknown"

    def test_standardize_protein(
        self,
        INPUT_MOL: Molecule,
        BEN_MOL: Molecule,
        CUSTOM_TEMPLATE: dict,
        HSD_RESIDUE: Residue,
    ) -> None:
        """Test the standardization of a protein molecule."""
        from prolif.io.protein_helper import ProteinHelper

        # Test standardizing a protein molecule
        protein_helper = ProteinHelper(INPUT_MOL)
        with pytest.warns(
            UserWarning,
            match="Residue NME3 has a different number of heavy atoms "
            r"than the standard residue\. This may affect H-bond detection\.",
        ):
            protein_helper.standardize_protein(templates=CUSTOM_TEMPLATE)
        assert isinstance(protein_helper.protein_mol, Molecule)
        assert len(protein_helper.protein_mol.residues) == len(INPUT_MOL.residues)

        protein_helper2 = ProteinHelper(INPUT_MOL)
        with pytest.warns(
            UserWarning,
            match="Residue NME3 has a different number of heavy atoms "
            r"than the standard residue\. This may affect H-bond detection\.",
        ):
            protein_helper2.standardize_protein(
                templates=[CUSTOM_TEMPLATE, {"BEN": {"SMILES": "NC(=N)c1ccccc1"}}]
            )
        assert isinstance(protein_helper2.protein_mol, Molecule)
        assert len(protein_helper2.protein_mol.residues) == len(
            protein_helper.protein_mol.residues
        )

        # Test with no valid templates
        protein_helper = ProteinHelper(INPUT_MOL)
        with pytest.raises(
            ValueError,
            match=r"Residue \{'ACE'\} is not a standard residue or "
            r"not in the templates\. Please provide a custom template\.",
        ):
            protein_helper.standardize_protein()

        # Test with a format
        protein_helper = ProteinHelper(INPUT_MOL)
        with pytest.raises(
            TypeError, match=r"Templates must be a dict, a list of dicts or None\."
        ):
            protein_helper.standardize_protein(templates=["invalid_format"])  # type: ignore

        # Test with a residue
        protein_helper = ProteinHelper(Molecule(HSD_RESIDUE))
        protein_helper.standardize_protein()
        assert len(protein_helper.protein_mol.residues) == 1
        assert str(protein_helper.protein_mol.residues[0].resid) == "HID109.A"

        # Test with the template having different names
        protein_helper = ProteinHelper(Molecule(HSD_RESIDUE))
        with pytest.warns(
            UserWarning, match=r"Align the template name \(BEN\) with \(RESNAME\)\."
        ):
            protein_helper.standardize_protein(
                templates=[
                    {"RESNAME": {"name": "BEN", "SMILES": "NC(=N)c1ccccc1"}},
                ]
            )

        # Test with a BENZAMIDINE residue
        protein_helper = ProteinHelper(BEN_MOL)
        protein_helper.standardize_protein(templates=CUSTOM_TEMPLATE)
        assert isinstance(protein_helper.protein_mol, Molecule)
        all_bonds_info = []
        for bond in protein_helper.protein_mol.residues[0].GetBonds():
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
        from prolif.io.protein_helper import _assign_intra_props_lone_H, strip_bonds

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
