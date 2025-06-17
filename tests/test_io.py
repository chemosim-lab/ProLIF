from pathlib import Path

import pandas as pd
import pytest

from prolif.datafiles import datapath


@pytest.fixture(scope="module")
def CIF():
    """Fixture to load the CIF file for testing."""
    return Path(str(datapath / "TPO.cif")).read_text()


@pytest.fixture(scope="module")
def STANDARD_AA():
    """Fixture to load the standard amino acid CIF file for testing."""
    return Path(str(datapath / "standard_aa.cif")).read_text()


def test_cif_parser_lite(CIF, STANDARD_AA):
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


@pytest.fixture(scope="module")
def XML_TEST_DATA():
    """Fixture to load the XML test data for altnames parsing."""
    return Path(str(datapath / "standard_aa_name.xml")).read_text()


def test_xml_parse_altnames(XML_TEST_DATA):
    """Test the XML parsing of alternative names for residues and atoms."""
    from prolif.io.xml import parse_altnames

    resname_aliases, atomname_aliases = parse_altnames(XML_TEST_DATA)
    assert isinstance(resname_aliases, dict)
    assert isinstance(atomname_aliases, dict)
