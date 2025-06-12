"""Name replacements for residues and atoms

The XML below is shamelessly stolen from openmm, (pdbNames.xml) and
should get patched there if additions are found

The function to parser the altnames is stolen from pdbinf.

Other constants are organized by Yu-Yuan (Stuart) Yang, 2025
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from prolif.datafiles import datapath

PDB_NAMES = """\
<Residues>
 <Residue name="All">
 </Residue>
 <Residue name="Protein">
  <Atom name="H" alt1="HN" alt2="H1" alt3="1H" alt4="HN1" alt5="HT1"/>
  <Atom name="H2" alt1="2H" alt2="HN2" alt3="HT2"/>
  <Atom name="H3" alt1="3H" alt2="HN3" alt3="HT3"/>
  <Atom name="O" alt1="O1" alt2="OT1" alt3="OCT1" alt4="OC1"/>
  <Atom name="OXT" alt1="O2" alt2="OT2" alt3="OCT2" alt4="OC2" alt5="OT"/>
 </Residue>
 <Residue name="Nucleic">
  <Atom name="H" alt1="HN"/>
  <Atom name="OP1" alt1="O1P"/>
  <Atom name="OP2" alt1="O2P"/>
  <Atom name="OP3" alt1="O3P"/>
  <Atom name="HOP2" alt1="2HOP"/>
  <Atom name="HOP3" alt1="3HOP"/>
  <Atom name="C1'" alt1="C1*"/>
  <Atom name="C2'" alt1="C2*"/>
  <Atom name="C3'" alt1="C3*"/>
  <Atom name="C4'" alt1="C4*"/>
  <Atom name="C5'" alt1="C5*"/>
  <Atom name="O2'" alt1="O2*"/>
  <Atom name="O3'" alt1="O3*"/>
  <Atom name="O4'" alt1="O4*"/>
  <Atom name="O5'" alt1="O5*"/>
  <Atom name="H1'" alt1="H1*"/>
  <Atom name="H2'" alt1="H2*" alt2="1H2*" alt3="1H2'" alt4="H2'1"/>
  <Atom name="H2''" alt1="2H2*" alt2="2H2'" alt3="H2'2"/>
  <Atom name="H3'" alt1="H3*"/>
  <Atom name="H4'" alt1="H4*"/>
  <Atom name="H5'" alt1="1H5*" alt2="1H5'" alt3="H5'1"/>
  <Atom name="H5''" alt1="2H5*" atl2="2H5'" alt3="H5'2"/>
  <Atom name="HO2'" alt1="2HO*"/>
  <Atom name="HO3'" alt1="H3T"/>
  <Atom name="HO5'" alt1="H5T"/>
 </Residue>
 <Residue name="GLY" type="Protein">
  <Atom name="HA2" alt1="2HA"/>
  <Atom name="HA3" alt1="HA1" alt2="1HA" alt3="3HA"/>
 </Residue>
 <Residue name="ALA" type="Protein">
  <Atom name="HB1" alt1="1HB"/>
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="3HB"/>
 </Residue>
 <Residue name="VAL" type="Protein">
  <Atom name="HG11" alt1="1HG1"/>
  <Atom name="HG12" alt1="2HG1"/>
  <Atom name="HG13" alt1="3HG1"/>
  <Atom name="HG21" alt1="1HG2"/>
  <Atom name="HG22" alt1="2HG2"/>
  <Atom name="HG23" alt1="3HG2"/>
 </Residue>
 <Residue name="LEU" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HD11" alt1="1HD1"/>
  <Atom name="HD12" alt1="2HD1"/>
  <Atom name="HD13" alt1="3HD1"/>
  <Atom name="HD21" alt1="1HD2"/>
  <Atom name="HD22" alt1="2HD2"/>
  <Atom name="HD23" alt1="3HD2"/>
 </Residue>
 <Residue name="ILE" type="Protein">
  <Atom name="CD1" alt1="CD"/>
  <Atom name="HG12" alt1="2HG1"/>
  <Atom name="HG13" alt1="HG11" alt2="1HG1" alt3="3HG1"/>
  <Atom name="HG21" alt1="1HG2"/>
  <Atom name="HG22" alt1="2HG2"/>
  <Atom name="HG23" alt1="3HG2"/>
  <Atom name="HD11" alt1="1HD1" alt2="HD1"/>
  <Atom name="HD12" alt1="2HD1" alt2="HD2"/>
  <Atom name="HD13" alt1="3HD1" alt2="HD3"/>
 </Residue>
 <Residue name="SER" type="Protein">
  <Atom name="OG" alt1="OG1"/>
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG" alt1="HG1" alt2="HOG"/>
 </Residue>
 <Residue name="THR" type="Protein">
  <Atom name="OG1" alt1="OG"/>
  <Atom name="CG2" alt1="CG"/>
  <Atom name="HG1" alt1="HOG" alt2="HOG1" alt3="1HG"/>
  <Atom name="HG21" alt1="1HG2"/>
  <Atom name="HG22" alt1="2HG2"/>
  <Atom name="HG23" alt1="3HG2"/>
 </Residue>
 <Residue name="CYS" type="Protein" alt1="CYX">
  <Atom name="SG" alt1="SG1"/>
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG" alt1="HG1" alt2="HSG"/>
 </Residue>
 <Residue name="PRO" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
  <Atom name="HD2" alt1="2HD"/>
  <Atom name="HD3" alt1="HD1" alt2="1HD" alt3="3HD"/>
  <Atom name="H2" alt1="HT1"/>
  <Atom name="H3" alt1="HT2"/>
 </Residue>
 <Residue name="PHE" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HD1" alt1="1HD"/>
  <Atom name="HD2" alt1="2HD"/>
  <Atom name="HE1" alt1="1HE"/>
  <Atom name="HE2" alt1="2HE"/>
 </Residue>
 <Residue name="TYR" type="Protein">
  <Atom name="HH" alt1="HOH"/>
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
 </Residue>
 <Residue name="TRP" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HE1" alt1="HNE"/>
 </Residue>
 <Residue name="HIS" type="Protein" \
    alt1="HSP" alt2="HSH" alt3="HIP" alt4="HIH" alt5="HID" alt6="HIE" alt7="HSD">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HD1" alt1="HND" alt2="HND1"/>
  <Atom name="HD2" alt1="HD"/>
  <Atom name="HE1" alt1="HE"/>
  <Atom name="HE2" alt1="HNE" alt2="HNE2"/>
 </Residue>
 <Residue name="ASP" type="Protein" alt1="ASH">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
 </Residue>
 <Residue name="ASN" type="Protein">
  <Atom name="OD1" alt1="OD"/>
  <Atom name="ND2" alt1="ND"/>
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HD21" alt1="1HD2" alt2="HND1"/>
  <Atom name="HD22" alt1="2HD2" alt2="HND2"/>
 </Residue>
 <Residue name="GLU" type="Protein" alt1="GLH">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
 </Residue>
 <Residue name="GLN" type="Protein">
  <Atom name="OE1" alt1="OE"/>
  <Atom name="NE2" alt1="NE"/>
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
  <Atom name="HE21" alt1="1HE2" alt2="HNE1"/>
  <Atom name="HE22" alt1="2HE2" alt2="HNE2"/>
 </Residue>
 <Residue name="MET" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
  <Atom name="HE1" alt1="1HE"/>
  <Atom name="HE2" alt1="2HE"/>
  <Atom name="HE3" alt1="3HE"/>
 </Residue>
 <Residue name="LYS" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
  <Atom name="HD2" alt1="2HD"/>
  <Atom name="HD3" alt1="HD1" alt2="1HD" alt3="3HD"/>
  <Atom name="HE2" alt1="2HE"/>
  <Atom name="HE3" alt1="HE1" alt2="1HE" alt3="3HE"/>
  <Atom name="HZ1" alt1="1HZ" alt2="HNZ1"/>
  <Atom name="HZ2" alt1="2HZ" alt2="HNZ2"/>
  <Atom name="HZ3" alt1="3HZ" alt2="HNZ3"/>
 </Residue>
 <Residue name="ARG" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
  <Atom name="HD2" alt1="2HD"/>
  <Atom name="HD3" alt1="HD1" alt2="1HD" alt3="3HD"/>
  <Atom name="HH11" alt1="1HH1" alt2="HN11"/>
  <Atom name="HH12" alt1="2HH1" alt2="HN12"/>
  <Atom name="HH21" alt1="1HH2" alt2="HN21"/>
  <Atom name="HH22" alt1="2HH2" alt2="HN22"/>
 </Residue>
 <Residue name="ORN" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
  <Atom name="HD2" alt1="2HD"/>
  <Atom name="HD3" alt1="HD1" alt2="1HD" alt3="3HD"/>
  <Atom name="HE1" alt1="1HE" alt2="HNE1"/>
  <Atom name="HE2" alt1="2HE" alt2="HNE2"/>
  <Atom name="HE3" alt1="3HE" alt2="HNE3"/>
 </Residue>
 <Residue name="AIB" type="Protein">
  <Atom name="HB11" alt1="1HB1"/>
  <Atom name="HB12" alt1="2HB1"/>
  <Atom name="HB13" alt1="3HB1"/>
  <Atom name="HB21" alt1="1HB2"/>
  <Atom name="HB22" alt1="2HB2"/>
  <Atom name="HB23" alt1="3HB2"/>
 </Residue>
 <Residue name="PCA" type="Protein">
  <Atom name="HB2" alt1="2HB"/>
  <Atom name="HB3" alt1="HB1" alt2="1HB" alt3="3HB"/>
  <Atom name="HG2" alt1="2HG"/>
  <Atom name="HG3" alt1="HG1" alt2="1HG" alt3="3HG"/>
 </Residue>
 <Residue name="FOR" type="Protein">
  <Atom name="C" alt1="CY"/>
  <Atom name="O" alt1="OY"/>
  <Atom name="H" alt1="HY"/>
 </Residue>
 <Residue name="ACE" type="Protein">
  <Atom name="C" alt1="CY"/>
  <Atom name="CH3" alt1="CAY" alt2="CA"/>
  <Atom name="O" alt1="OY"/>
  <Atom name="H1" alt1="1H" alt2="HY1" alt3="HH31" alt4="1HH3"/>
  <Atom name="H2" alt1="2H" alt2="HY2" alt3="HH32" alt4="2HH3"/>
  <Atom name="H3" alt1="3H" alt2="HY3" alt3="HH33" alt4="3HH3"/>
 </Residue>
 <Residue name="NME" alt1="NMA" type="Protein">
  <Atom name="N" alt1="NT"/>
  <Atom name="C" alt1="CH3" alt2="CT" alt3="CAT" alt4="CA"/>
  <Atom name="H" alt1="HNT" alt2="HN"/>
  <Atom name="H1" alt1="1H" alt2="1HA" alt3="HT1" alt4="HH31" alt5="1HH3"/>
  <Atom name="H2" alt1="2H" alt2="2HA" alt3="HT2" alt4="HH32" alt5="2HH3"/>
  <Atom name="H3" alt1="3H" alt2="3HA" alt3="HT3" alt4="HH33" alt5="3HH3"/>
 </Residue>
 <Residue name="NH2" alt1="NHE" type="Protein">
  <Atom name="N" alt1="NT"/>
  <Atom name="HN1" alt1="1H" alt2="HT1" alt3="H1"/>
  <Atom name="HN2" alt1="2H" alt2="HT2" alt3="H2"/>
 </Residue>
 <Residue name="UNK" type="Protein"/>
 <Residue name="A" alt1="ADE" alt2="A3" alt3="A5" type="Nucleic">
  <Atom name="H61" alt1="1H6"/>
  <Atom name="H62" alt1="2H6"/>
 </Residue>
 <Residue name="G" alt1="GUA" alt2="G3" alt3="G5" type="Nucleic">
  <Atom name="H21" alt1="1H2"/>
  <Atom name="H22" alt1="2H2"/>
 </Residue>
 <Residue name="C" alt1="CYT" alt2="C3" alt3="C5" type="Nucleic">
  <Atom name="H41" alt1="1H4"/>
  <Atom name="H42" alt1="2H4"/>
 </Residue>
 <Residue name="U" alt1="URA" alt2="U3" alt3="U5" type="Nucleic"/>
 <Residue name="DA" alt1="DAD" alt2="DA3" alt3="DA5" type="Nucleic">
  <Atom name="H61" alt1="1H6"/>
  <Atom name="H62" alt1="2H6"/>
 </Residue>
 <Residue name="DG" alt1="DGU" alt2="DG3" alt3="DG5" type="Nucleic">
  <Atom name="H21" alt1="1H2"/>
  <Atom name="H22" alt1="2H2"/>
 </Residue>
 <Residue name="DC" alt1="DCY" alt2="DC3" alt3="DC5" type="Nucleic">
  <Atom name="H41" alt1="1H4"/>
  <Atom name="H42" alt1="2H4"/>
 </Residue>
 <Residue name="DT" alt1="THY" alt2="DT3" alt3="DT5" type="Nucleic">
  <Atom name="C7" alt1="C5M"/>
  <Atom name="H71" alt1="1H5M"/>
  <Atom name="H72" alt1="2H5M"/>
  <Atom name="H73" alt1="3H5M"/>
 </Residue>
 <Residue name="HOH" alt1="H20" alt2="WAT" alt3="SOL" alt4="TIP3" alt5="TP3" alt6="TIP">
  <Atom name="O" alt1="OW" alt2="OH2"/>
  <Atom name="H1" alt1="HW1"/>
  <Atom name="H2" alt1="HW2"/>
 </Residue>
</Residues>
"""


def _parse_altnames():
    residues = ET.fromstring(PDB_NAMES)

    resname_aliases = {}
    atomname_aliases = {}

    for res in residues:
        resname = res.attrib["name"]

        for k, alt_resname in res.attrib.items():
            if k.startswith("alt"):
                resname_aliases[alt_resname] = resname

        aliases = {}
        for atom in res:
            atomname = atom.attrib["name"]

            for k, alt_atomname in atom.attrib.items():
                if k.startswith("alt"):
                    aliases[alt_atomname] = atomname

        atomname_aliases[resname] = aliases

    return resname_aliases, atomname_aliases


# RESNAME_ALIASES: dict mapping resnames to canonical names
# ATOMNAME_ALIASES: dict of dicts (keyed with canonical resnames)
#                   mapping atomnames to canonical
RESNAME_ALIASES, ATOMNAME_ALIASES = _parse_altnames()

MAX_AMIDE_LENGTH = 2.0
MAX_DISULPHIDE_LENGTH = 2.5

# amino acid template
_STANDARD_AA = Path(str(datapath / "standard_aa.cif")).read_text()
# --- the above is originally from openmm and pdbinf ---
# (the below formal charge map is relevant to the above _STANDARD_AA)
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
formal_charge_special_assign_map = {
    "ARG": {"NH2": 1},
    "ASP": {"OD2": -1},
    "ASH": {"OD2": 0},
    "CYS": {"SG": 0},
    "CYX": {"SG": -1},
    "GLU": {"OE2": -1},
    "GLH": {"OE2": 0},
    "HIS": {"ND1": 1, "NE2": 0},
    "HIE": {"ND1": 0, "NE2": 0},
    "HID": {"ND1": 0, "NE2": 0},
    "LYS": {"NZ": 1},
}

# resnames_map: dict mapping various residue names to their canonical names
resnames_map = {
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
}

# Pools of residue names for different force fields
# OPLS-AA, GROMOS, CHARMM, AMBER
# Once the residue is in the pool, it suggests the user
# used the corresponding force field for the simulations.
opls_aa_pool = {"HISD", "HISE", "PGLU"}
gromos_pool = {"DALA", "ASN1", "CYS1", "HIS1", "HIS2", "HISA", "HISB"}
charmm_pool = {
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
amber_pool = {
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
