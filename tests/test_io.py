from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from prolif.datafiles import datapath
from prolif.molecule import Molecule


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
    def INPUT_MOL(self, INPUT_PATH) -> Molecule:
        """Return the Molecule object for the input file."""
        input_mol = Chem.MolFromPDBFile(INPUT_PATH)
        return Molecule.from_rdkit(input_mol)

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
            ProteinHelper(12345)
