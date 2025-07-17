import pandas as pd

from prolif.constants import (
    AMBER_POOL,
    ATOMNAME_ALIASES,
    CHARMM_POOL,
    FORMAL_CHARGE_ALIASES,
    GROMOS_POOL,
    MAX_AMIDE_LENGTH,
    MAX_DISULPHIDE_LENGTH,
    OPLS_AA_POOL,
    RESNAME_ALIASES,
    STANDARD_AA,
    STANDARD_RESNAME_MAP,
    TERMINAL_OXYGEN_NAMES,
)


def test_standard_aa() -> None:
    """Test that STANDARD_AA contains the expected amino acids."""

    assert len(STANDARD_AA) == 26
    assert len(STANDARD_AA["ALA"]) == 4
    assert isinstance(STANDARD_AA["ALA"]["_chem_comp_atom"], pd.DataFrame)


def test_resname_aliases() -> None:
    """Test that RESNAME_ALIASES contains the expected aliases."""

    assert len(RESNAME_ALIASES) == 42
    assert RESNAME_ALIASES["HSH"] == "HIS"


def test_atomname_aliases() -> None:
    """Test that ATOMNAME_ALIASES contains the expected aliases."""

    assert len(ATOMNAME_ALIASES) == 40
    assert len(ATOMNAME_ALIASES["Protein"]) == 20
    assert ATOMNAME_ALIASES["Protein"]["OT"] == "OXT"


def test_standard_resname_map() -> None:
    """Test that STANDARD_RESNAME_MAP contains the expected mappings."""

    assert len(STANDARD_RESNAME_MAP) == 100
    assert STANDARD_RESNAME_MAP["ASN1"] == "ASN"
    assert STANDARD_RESNAME_MAP["CYS"] == "CYS"
    assert STANDARD_RESNAME_MAP["CTRP"] == "TRP"
    assert STANDARD_RESNAME_MAP["NMET"] == "MET"


def test_force_field_pool() -> None:
    """Test the number of contents in force field pool."""

    assert len(AMBER_POOL) == 55
    assert len(CHARMM_POOL) == 10
    assert len(GROMOS_POOL) == 7
    assert len(OPLS_AA_POOL) == 3


def test_formal_charge_aliases() -> None:
    """Test that FORMAL_CHARGE_ALIASES contains the expected aliases."""

    assert len(FORMAL_CHARGE_ALIASES) == 11
    assert FORMAL_CHARGE_ALIASES["ARG"]["NH2"] == 1
    assert FORMAL_CHARGE_ALIASES["ASP"]["OD2"] == -1
    assert FORMAL_CHARGE_ALIASES["CYS"]["SG"] == 0
    assert FORMAL_CHARGE_ALIASES["HIS"]["ND1"] == 1


def test_other_constants() -> None:
    """Test that other constants are defined correctly."""

    assert MAX_AMIDE_LENGTH == 2
    assert MAX_DISULPHIDE_LENGTH == 2.5
    assert len(TERMINAL_OXYGEN_NAMES) == 6
    assert "OXT" in TERMINAL_OXYGEN_NAMES
    assert "OT" in TERMINAL_OXYGEN_NAMES
