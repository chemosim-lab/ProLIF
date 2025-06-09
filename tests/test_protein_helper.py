import pandas as pd

from prolif.datafiles import datapath


def test_cif_parser_lite():
    from prolif.protein_helper import cif_parser_lite

    # Test with a valid CIF file

    CIF = str(datapath / "TPO.cif")
    with open(CIF) as f:
        tpo_cif = f.read()
    result = cif_parser_lite(tpo_cif)

    assert isinstance(result, dict)
    assert isinstance(result["TPO"]["_pdbx_chem_comp_synonyms"], dict)
    assert result["TPO"]["_pdbx_chem_comp_synonyms"]["name"] == "PHOSPHONOTHREONINE"
    assert isinstance(result["TPO"]["_pdbx_chem_comp_descriptor"], pd.DataFrame)
