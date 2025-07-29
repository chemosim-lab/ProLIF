"""Name replacements for residues and atoms

Some constants are originally from OpenMM and pdbinf.

Other constants are organized by Yu-Yuan (Stuart) Yang, 2025
"""

from pathlib import Path

from prolif.datafiles import datapath
from prolif.io.cif import cif_parser_lite
from prolif.io.xml import parse_altnames

# --------- the below is originally from openmm and pdbinf ---------
# RESNAME_ALIASES: dict mapping residue names to canonical resnames
# ATOMNAME_ALIASES: dict of dicts (keyed with canonical resnames)
#                   mapping atomnames to canonical
# Source:
# https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_aliases.py#L323
_PDB_NAMES = Path(
    str(datapath / "protein_helper/templates/standard_aa_name.xml")
).read_text()
RESNAME_ALIASES, ATOMNAME_ALIASES = parse_altnames(_PDB_NAMES)

# Source:
# https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_pdbinf.py#L65
MAX_AMIDE_LENGTH = 2.0
MAX_DISULPHIDE_LENGTH = 2.5

# amino acid template
# Source:
# https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_standard_AAs.py#L6
_STANDARD_AA = Path(
    str(datapath / "protein_helper/templates/standard_aa.cif")
).read_text()
STANDARD_AA = cif_parser_lite(_STANDARD_AA)
# --------- the above is originally from openmm and pdbinf ---------

# Other constants are defined as follows:
#
# Different from the pdbinf, the residue names here are seperated
# by using side chain H-bond donor status. See below for details.
# ALA, ARG (Protonated (+1), default), ASN,
# ASP (Deprotonated (-1), default), ASH (Neutral (0)),
# CYS (Neutral (0), default), CYX (Deprotonated (-1) or forming an S-S bridge),
# GLN, GLY,
# GLU (Deprotonated (-1), default), GLH (Neutral (0)),
# HIS (Both NE2 and ND1 protonated, default),
# HIE (neutral, hydrogen at NE2, might couple to HEME at ND1),
# HID (neutral, hydrogen at ND1, might couple to HEME at NE2),
# ILE, LEU, LYS (Protonated (+1), default), MET, PRO, PHE, SER, TYR, THR, TRP, VAL
# HOH (water)

# STANDARD_RESNAME_MAP: dict mapping various residue names to standardised names based
# on their H-bond donor
STANDARD_RESNAME_MAP = {
    # ALA
    "ALA": "ALA",  # Common name
    "ALAD": "ALA",  # CHARMM
    "DALA": "ALA",  # GROMOS
    "NALA": "ALA",  # AMBER
    "CALA": "ALA",  # AMBER
    # ARG (Protonated (+1) or Neutral (0), both can have a possible donor)
    # with H-bond donor (sp2 N)
    "ARG": "ARG",  # Common name
    "NARG": "ARG",  # AMBER
    "CARG": "ARG",  # AMBER
    "ARGN": "ARG",  # OPLS-AA, GROMOS
    # ASN
    # with H-bond donor (sp2 N)
    "ASN": "ASN",  # Common name
    "NASN": "ASN",  # AMBER
    "CASN": "ASN",  # AMBER
    "ASN1": "ASN",  # GROMOS
    "CASF": "ASN",  # AMBER
    "ASF": "ASN",  # AMBER
    # ASP (Deprotonated (-1))
    "ASP": "ASP",  # Common name
    "NASP": "ASP",  # AMBER
    "CASP": "ASP",  # AMBER
    # ASH (Neutral (0))
    # with H-bond donor (sp3 O)
    "ASPH": "ASH",  # OPLS-AA, GROMOS
    "ASPP": "ASH",  # CHARMM
    "ASH": "ASH",  # AMBER
    # CYS (Neutral (0))
    # with H-bond donor (sp3 S)
    "CYS": "CYS",  # AMBER and CHARMM
    "CYSH": "CYS",  # OPLS-AA
    "NCYS": "CYS",  # AMBER
    "CCYS": "CYS",  # AMBER
    # CYX (Deprotonated (-1) or forming an S-S bridge,
    # both cannot have a possible donor)
    "CYS1": "CYX",  # GROMOS
    "CYS2": "CYX",  # GROMOS, CHARMM, OPLS-AA
    "CYN": "CYX",  # CHARMM
    "CYM": "CYX",  # CHARMM
    "CYX": "CYX",  # AMBER
    "NCYX": "CYX",  # AMBER
    "CCYX": "CYX",  # AMBER
    # GLN
    # with H-bond donor (sp2 N)
    "GLN": "GLN",  # Common name
    "NGLN": "GLN",  # AMBER
    "CGLN": "GLN",  # AMBER
    # GLY
    "GLY": "GLY",  # Common name
    "NGLY": "GLY",  # AMBER
    "CGLY": "GLY",  # AMBER
    # GLU (Deprotonated (-1))
    "GLU": "GLU",  # Common name
    "NGLU": "GLU",  # AMBER
    "CGLU": "GLU",  # AMBER
    # GLH (Neutral (0))
    # with H-bond donor (sp3 O)
    "GLUH": "GLH",  # OPLS-AA, GROMOS
    "GLUP": "GLH",  # CHARMM
    "GLH": "GLH",  # AMBER
    # HIS (Both NE2 and ND1 protonated)
    # with H-bond donor (sp2 N)
    "HIS": "HIS",  # Common name
    "HISH": "HIS",  # OPLS-AA, GROMOS
    "HIP": "HIS",  # AMBER
    "NHIP": "HIS",  # AMBER
    "CHIP": "HIS",  # AMBER
    "HSP": "HIS",  # CHARMM
    # HIE (neutral, hydrogen at NE2, might couple to HEME at ND1)
    # with H-bond donor (sp2 N)
    "HIS1": "HIE",  # GROMOS
    "HISB": "HIE",  # GROMOS
    "HISE": "HIE",  # OPLS-AA
    "HIE": "HIE",  # AMBER
    "NHIE": "HIE",  # AMBER
    "CHIE": "HIE",  # AMBER
    "HSE": "HIE",  # CHARMM
    # HID (neutral, hydrogen at ND1, might couple to HEME at NE2)
    # with H-bond donor (sp2 N)
    "HIS2": "HID",  # GROMOS
    "HISA": "HID",  # GROMOS
    "HISD": "HID",  # OPLS-AA
    "HID": "HID",  # AMBER
    "NHID": "HID",  # AMBER
    "CHID": "HID",  # AMBER
    "HSD": "HID",  # CHARMM
    # ILE
    "ILE": "ILE",  # Common name
    "NILE": "ILE",  # AMBER
    "CILE": "ILE",  # AMBER
    # LEU
    "LEU": "LEU",  # Common name
    "NLEU": "LEU",  # AMBER
    "CLEU": "LEU",  # AMBER
    # LYS (Protonated (+1) or Neutral (0), both can have a possible donor)
    # with H-bond donor (sp3 N)
    "LYS": "LYS",  # Common name
    "LYN": "LYS",  # AMBER
    "LSN": "LYS",  # CHARMM
    "LYSH": "LYS",  # OPLS-AA, GROMOS
    "NLYS": "LYS",  # AMBER
    "CLYS": "LYS",  # AMBER
    # MET
    "MET": "MET",  # Common name
    "NMET": "MET",  # AMBER
    "CMET": "MET",  # AMBER
    # PRO
    "PRO": "PRO",  # Common name
    "NPRO": "PRO",  # AMBER
    "CPRO": "PRO",  # AMBER
    # PHE
    "PHE": "PHE",  # Common name
    "NPHE": "PHE",  # AMBER
    "CPHE": "PHE",  # AMBER
    # SER
    # with H-bond donor (sp3 O)
    "SER": "SER",  # Common name
    "NSER": "SER",  # AMBER
    "CSER": "SER",  # AMBER
    # TYR
    # with H-bond donor (sp3 O)
    "TYR": "TYR",  # Common name
    "NTYR": "TYR",  # AMBER
    "CTYR": "TYR",  # AMBER
    # THR
    # with H-bond donor (sp3 O)
    "THR": "THR",  # Common name
    "NTHR": "THR",  # AMBER
    "CTHR": "THR",  # AMBER
    # TRP
    # with H-bond donor (sp2 N)
    "TRP": "TRP",  # Common name
    "NTRP": "TRP",  # AMBER
    "CTRP": "TRP",  # AMBER
    # VAL
    "VAL": "VAL",  # Common name
    "NVAL": "VAL",  # AMBER
    "CVAL": "VAL",  # AMBER
    # HOH
    **{k: v for k, v in RESNAME_ALIASES.items() if v == "HOH"}
}

# FORMAL_CHARGE_ALIASES: dict mapping residue names to a dict of atom names
# (the below map is relevant to the above atom name/topology file in STANDARD_AA)
FORMAL_CHARGE_ALIASES = {
    "ARG": {"NH2": 1},
    "ASP": {"OD2": -1},
    "ASH": {"OD2": 0},
    "CYS": {"SG": 0, "SG1": 0},  # alternative name for SG
    "CYX": {"SG": -1, "SG1": -1},  # alternative name for SG
    "GLU": {"OE2": -1},
    "GLH": {"OE2": 0},
    "HIS": {"ND1": 1, "NE2": 0},
    "HIE": {"ND1": 0, "NE2": 0},
    "HID": {"ND1": 0, "NE2": 0},
    "LYS": {"NZ": 1},
}

# Pools of residue names for different force fields: OPLS-AA, GROMOS, CHARMM, AMBER
# Once the residue is in the pool, it suggests the user
# used the corresponding force field for the simulations.
OPLS_AA_POOL = {"HISD", "HISE", "PGLU"}
GROMOS_POOL = {"DALA", "ASN1", "CYS1", "HIS1", "HIS2", "HISA", "HISB"}
CHARMM_POOL = {
    "ALAD",
    "ASPP",
    "CYM",
    "CYN",
    "GLUP",
    "HSD",
    "HSE",
    "HSP",
    "LSN",
    "CME",
}
AMBER_POOL = {
    "NALA",
    "CALA",
    "NARG",
    "CARG",
    "NASN",
    "CASN",
    "CASF",
    "ASF",
    "NASP",
    "CASP",
    "ASH",
    "CYX",
    "NCYS",
    "CCYS",
    "NCYX",
    "CCYX",
    "NGLN",
    "CGLN",
    "NGLY",
    "CGLY",
    "GLH",
    "NGLU",
    "CGLU",
    "HID",
    "HIE",
    "HIP",
    "NHID",
    "NHIE",
    "NHIP",
    "CHID",
    "CHIE",
    "CHIP",
    "NILE",
    "CILE",
    "NLEU",
    "CLEU",
    "LYN",
    "NLYS",
    "CLYS",
    "NMET",
    "CMET",
    "NPRO",
    "CPRO",
    "NPHE",
    "CPHE",
    "NSER",
    "CSER",
    "NTYR",
    "CTYR",
    "NTHR",
    "CTHR",
    "NTRP",
    "CTRP",
    "NVAL",
    "CVAL",
}

TERMINAL_OXYGEN_NAMES = {
    key for key, value in ATOMNAME_ALIASES["Protein"].items() if value.strip() == "OXT"
}
TERMINAL_OXYGEN_NAMES.add("OXT")
