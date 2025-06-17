import pandas as pd


def test_standard_aa():
    """Test that STANDARD_AA contains the expected amino acids."""

    from prolif.constants import STANDARD_AA

    assert len(STANDARD_AA) == 26
    assert len(STANDARD_AA["ALA"]) == 3
    assert isinstance(STANDARD_AA["ALA"]["_chem_comp_atom"], pd.DataFrame)


def test_resname_aliases():
    """Test that RESNAME_ALIASES contains the expected aliases."""

    from prolif.constants import RESNAME_ALIASES

    assert len(RESNAME_ALIASES) == 42
    assert RESNAME_ALIASES["HSH"] == "HIS"


def test_atomname_aliases():
    """Test that ATOMNAME_ALIASES contains the expected aliases."""

    from prolif.constants import ATOMNAME_ALIASES

    assert len(ATOMNAME_ALIASES) == 40
    assert len(ATOMNAME_ALIASES["Protein"]) == 20
    assert ATOMNAME_ALIASES["Protein"]["OT"] == "OXT"


def test_standard_resname_map():
    """Test that STANDARD_RESNAME_MAP contains the expected mappings."""

    from prolif.constants import STANDARD_RESNAME_MAP

    assert len(STANDARD_RESNAME_MAP) == 100
    assert STANDARD_RESNAME_MAP["ASN1"] == "ASN"
    assert STANDARD_RESNAME_MAP["CYS"] == "CYS"
    assert STANDARD_RESNAME_MAP["CTRP"] == "TRP"
    assert STANDARD_RESNAME_MAP["NMET"] == "MET"


def test_force_field_pool():
    """Test the number of contents in force field pool."""

    from prolif.constants import AMBER_POOL, CHARMM_POOL, GROMOS_POOL, OPLS_AA_POOL

    assert len(AMBER_POOL) == 55
    assert len(CHARMM_POOL) == 10
    assert len(GROMOS_POOL) == 7
    assert len(OPLS_AA_POOL) == 3


def test_formal_charge_aliases():
    """Test that FORMAL_CHARGE_ALIASES contains the expected aliases."""

    from prolif.constants import FORMAL_CHARGE_ALIASES

    assert len(FORMAL_CHARGE_ALIASES) == 11
    assert FORMAL_CHARGE_ALIASES["ARG"]["NH2"] == 1
    assert FORMAL_CHARGE_ALIASES["ASP"]["OD2"] == -1
    assert FORMAL_CHARGE_ALIASES["CYS"]["SG"] == 0
    assert FORMAL_CHARGE_ALIASES["HIS"]["ND1"] == 1


def test_other_constants():
    """Test that other constants are defined correctly."""

    from prolif.constants import (
        MAX_AMIDE_LENGTH,
        MAX_DISULPHIDE_LENGTH,
        N_STANDARD_RESIDUE_HEAVY_ATOMS,
        TERMINAL_OXYGEN_NAMES,
    )

    assert MAX_AMIDE_LENGTH == 2
    assert MAX_DISULPHIDE_LENGTH == 2.5
    assert len(N_STANDARD_RESIDUE_HEAVY_ATOMS) == 26
    assert len(TERMINAL_OXYGEN_NAMES) == 6
    assert "OXT" in TERMINAL_OXYGEN_NAMES
    assert "OT" in TERMINAL_OXYGEN_NAMES
