import shlex
import warnings

import MDAnalysis as mda
import pandas as pd

from prolif.molecule import Molecule

all_resnames_dict = {
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


class ProteinHelper:
    def __init__(self, input_topolody: str | Molecule):
        """Initialize the ProteinHelper class.

        Parameters
        ----------
        input_topolody : str or Molecule
            The path of input protein molecule or a ProLIF Molecule.
        """

        # read as prolif molecule
        if isinstance(input_topolody, Molecule):
            self.protein_mol = input_topolody

        elif isinstance(input_topolody, str):
            input_protein_top = mda.Universe(input_topolody)
            if not hasattr(input_protein_top.atoms, "elements"):
                # MDAnalysis 3.0.0 deprecates mda.topology.guessers.guess_types
                try:
                    # For MDAnalysis < 3.0.0
                    guessed_elements = mda.topology.guessers.guess_types(
                        input_protein_top.atoms.names
                    )
                    input_protein_top.add_TopologyAttr("elements", guessed_elements)
                except AttributeError:
                    # For MDAnalysis >= 3.0.0, use the new guessers API
                    input_protein_top.guess_TopologyAttrs(force_guess=["elements"])

            self.protein_mol = Molecule.from_mda(input_protein_top)

        else:
            raise TypeError(
                "input_topology must be a string (path to a PDB file) or "
                "a prolif Molecule instance."
            )

    def standardize_protein(self) -> None:
        """Standardize the protein molecule."""

        # guess forcefield
        conv_resnames = set(self.protein_mol.residues.name)
        forcefield_name = self.forcefield_guesser(conv_resnames)

        # standardize the protein molecule
        standard_resnames = []
        for res in self.protein_mol.residues.name:
            standard_resname = self.convert_to_standard_resname(
                resname=res, forcefield_name=forcefield_name
            )
            standard_resnames.append(standard_resname)
        self.protein_mol.residues.name = standard_resnames
        stand_resnames = set(standard_resnames)

        # [TODO] fix the bond orders for non-standard residues
        if self.check_resnames(stand_resnames):
            self.fix_molecule_bond_orders()

    @staticmethod
    def forcefield_guesser(
        conv_resnames: set[str],
    ) -> str:
        """Guesses the forcefield based on the residue names.

        Parameters
        ----------
        conv_resnames : set[str]
            Set of residue names in the protein molecule.

        Returns
        -------
        str
            The guessed forcefield name.
        """
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
        if len(amber_pool.intersection(conv_resnames)) != 0:
            return "amber"
        if len(charmm_pool.intersection(conv_resnames)) != 0:
            return "charmm"
        if len(gromos_pool.intersection(conv_resnames)) != 0:
            return "gromos"
        if len(opls_aa_pool.intersection(conv_resnames)) != 0:
            return "oplsaa"

        return "unknown"

    @staticmethod
    def convert_to_standard_resname(
        resname: str, forcefield_name: str = "unknown"
    ) -> str:
        """Convert a residue name to its standard form based on the forcefield.

        Note that this conversion is designed to distinguish residues with
        different possible H-bond donors at side chains, instead of the
        actual protonated states of residues.
        For example, neutral and protonated arginine are both assigned to ARG,
        while neutral and deprotonated arginine in GROMOS force field
        are assigned to ARGN and ARG, respectively.


        Parameters
        ----------
        resname : str
            The residue name to convert.
        forcefield_name : str
            The name of the forcefield to use for conversion. Default is "unknown".

        Returns
        -------
        str
            The standard residue name.
        """

        if forcefield_name == "unknown":
            warnings.warn(
                "Could not guess the forcefield based on the residue names. "
                "CYS is assigned to neutral CYS (charge = 0).",
                stacklevel=2,
            )

        if forcefield_name == "gromos" and resname == "CYS":
            return "CYX"

        return all_resnames_dict.get(resname, resname)

    @staticmethod
    def check_resnames(resnames_to_check: set[str]) -> bool:
        """Check if the residue names are standard and raise a warning if not.

        Parameters
        ----------
        resnames_to_check : set[str]
            Set of residue names to check.

        Raises
        ------
        UserWarning
            If any residue name is not standard.
        """
        all_resnames = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "ASH",
            "CYS",
            "CYX",
            "GLN",
            "GLY",
            "GLU",
            "GLH",
            "HIS",
            "HIE",
            "HID",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PRO",
            "PHE",
            "SER",
            "TYR",
            "THR",
            "TRP",
            "VAL",
        }
        non_standard_resnames = resnames_to_check - all_resnames
        if non_standard_resnames:
            warnings.warn(
                "Non-standard residue (or ligand/solvent) names found: "
                f"{non_standard_resnames}",
                stacklevel=2,
            )
            return True
        return False

    def fix_molecule_bond_orders(self, templates=None) -> None:
        """Fix the bond orders of a molecule.

        Parameters
        ----------
        templates : str or list of str, optional
            The templates to use for fixing the bond orders.
            If `None`, the standard amino acid template is used.
            Default is `None`.
        """
        templates = [STANDARD_AA] if templates is None else [templates, STANDARD_AA]
        raise NotImplementedError()
        # [TODO] fix the bond orders for non-standard residues

        # check number of heavy atoms with templates

        # read the templates with RDKit

        # fix molecule bond orders with templates


# The below code contains functions for fixing molecule bond orders.
# The idea is based on pdbinf: https://github.com/OpenFreeEnergy/pdbinf/tree/main
# A user can provide a custom CIF file with the standard amino acid.


def _block_decompose(data_block: list) -> tuple:
    """
    Decomposes a CIF data block into decriptive information and tables.
    """
    descriptions = []
    data_tables = []
    data_table = None

    for idx, block_line in enumerate(data_block):
        if block_line.startswith("#"):
            if data_table is not None:
                # save the current table
                data_tables.append(data_table)
            # reset the table
            data_table = None
        elif block_line.startswith("loop_"):
            # table format
            data_table = []
        elif data_table is not None:
            # add data to the current table
            data_table.append(block_line)
            if idx == len(data_block) - 1:  # last line of the block
                # save the final table
                data_tables.append(data_table)
        else:
            descriptions.append(block_line)

    return descriptions, data_tables


def cif_parser_lite(cif_string: str) -> dict:
    """
    Parses a CIF string and returns a dictionary of data blocks.

    Parameters
    ----------
    cif_string : str
        The CIF string to parse.

    """
    # Split the CIF string into blocks based on 'data_' lines
    data_blocks = {}
    current_block = None
    all_lines = cif_string.strip().split("\n")
    for idx, line in enumerate(all_lines):
        if line.startswith("data_"):
            current_block = line.split("data_")[1]
            data_block = []
        elif line.startswith("##") or idx == len(all_lines) - 1:
            # end of a data block
            data_blocks[current_block] = data_block
        else:
            data_block.append(line.strip())

    # create a dictionary to hold the parsed data
    cif_dict = {}
    for block_name, data_block in data_blocks.items():
        descriptions, data_tables = _block_decompose(data_block)
        cif_dict[block_name] = {}

        # descriptive information
        for each in descriptions:
            content = shlex.split(each)
            info_name = content[0].split(".")
            info = content[1]
            if info_name[0] not in cif_dict[block_name]:
                cif_dict[block_name][info_name[0]] = {}
            cif_dict[block_name][info_name[0]][info_name[1]] = info

        # data tables
        for data_table in data_tables:
            header = []
            content = []
            table_name = data_table[0].split(".")[0]
            for each_line in data_table:
                if each_line.startswith("_"):
                    # header line
                    header.append(each_line.split(".")[1].strip())
                else:
                    # content line
                    # Use shlex.split to respect quoted strings
                    content.append(shlex.split(each_line))

            table = pd.DataFrame(content, columns=header)
            cif_dict[block_name][table_name] = table

    return cif_dict


# amino acid template
# ALA, ARG, ASN,
# ASP (Deprotonated (-1), default), ASH (Neutral (0)),
# CYS (Neutral (0), default), CYX,
# GLN, GLY,
# GLU (Deprotonated (-1), default), GLH (Neutral (0)),
# HIS (Both NE2 and ND1 protonated, default),
# HIE (neutral, hydrogen at NE2, might couple to HEME at ND1),
# HID (neutral, hydrogen at ND1, might couple to HEME at NE2),
# ILE, LEU, LYS, MET, PRO, PHE, SER, TYR, THR, TRP, VAL
# HOH (water)
_STANDARD_AA = """\
data_ALA
#
_chem_comp.id ALA
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
ALA  N    N    N  N  N  N
ALA  CA   CA   C  N  N  S
ALA  C    C    C  N  N  N
ALA  O    O    O  N  N  N
ALA  CB   CB   C  N  N  N
ALA  OXT  OXT  O  N  Y  N
ALA  HA   HA   H  N  N  N
ALA  HB1  1HB  H  N  N  N
ALA  HB2  2HB  H  N  N  N
ALA  HB3  3HB  H  N  N  N
ALA  H    H    H  N  N  N
ALA  H2   H2   H  N  N  N
ALA  H3   H3   H  N  Y  N
ALA  HXT  HXT  H  N  Y  N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
N   H3  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  HB1 SING N N
CB  HB2 SING N N
CB  HB3 SING N N
OXT HXT SING N N
##
data_ARG
#
_chem_comp.id ARG
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
ARG N    N    N N N N
ARG CA   CA   C N N S
ARG C    C    C N N N
ARG O    O    O N N N
ARG CB   CB   C N N N
ARG CG   CG   C N N N
ARG CD   CD   C N N N
ARG NE   NE   N N N N
ARG CZ   CZ   C N N N
ARG NH1  NH1  N N N N
ARG NH2  NH2  N N N N
ARG OXT  OXT  O N Y N
ARG H    H    H N N N
ARG H2   HN2  H N Y N
ARG HA   HA   H N N N
ARG HB2  1HB  H N N N
ARG HB3  2HB  H N N N
ARG HG2  1HG  H N N N
ARG HG3  2HG  H N N N
ARG HD2  1HD  H N N N
ARG HD3  2HD  H N N N
ARG HE   HE   H N N N
ARG HH11 1HH1 H N N N
ARG HH12 2HH1 H N N N
ARG HH21 1HH2 H N N N
ARG HH22 2HH2 H N N N
ARG HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
N   H3   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  CG   SING N N
CB  HB2  SING N N
CB  HB3  SING N N
CG  CD   SING N N
CG  HG2  SING N N
CG  HG3  SING N N
CD  NE   SING N N
CD  HD2  SING N N
CD  HD3  SING N N
NE  CZ   SING N N
NE  HE   SING N N
CZ  NH1  SING N N
CZ  NH2  DOUB N N
NH1 HH11 SING N N
NH1 HH12 SING N N
NH2 HH21 SING N N
NH2 HH22 SING N N
OXT HXT  SING N N
##
data_ASN
#
_comp_chem.id ASN
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N    N    N N N N
CA   CA   C N N S
C    C    C N N N
O    O    O N N N
CB   CB   C N N N
CG   CG   C N N N
OD1  OD1  O N N N
ND2  ND2  N N N N
OXT  OXT  O N Y N
H    H    H N N N
H2   HN2  H N Y N
HA   HA   H N N N
HB2  1HB  H N N N
HB3  2HB  H N N N
HD21 1HD2 H N N N
HD22 2HD2 H N N N
HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  CG   SING N N
CB  HB2  SING N N
CB  HB3  SING N N
CG  OD1  DOUB N N
CG  ND2  SING N N
ND2 HD21 SING N N
ND2 HD22 SING N N
OXT HXT  SING N N
##
data_ASP
#
_chem_comp.id ASP
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
OD1 OD1 O N N N
OD2 OD2 O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 HB1 H N N N
HB3 HB2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  OD1 DOUB N N
CG  OD2 SING N N
OXT HXT SING N N
##
data_ASH
#
_chem_comp.id ASH
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
OD1 OD1 O N N N
OD2 OD2 O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 HB1 H N N N
HB3 HB2 H N N N
HD2 HD2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  OD1 DOUB N N
CG  OD2 SING N N
OD2 HD2 SING N N
OXT HXT SING N N
##
data_CYS
#
_chem_comp.id CYS
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N R
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
SG  SG  S N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HG  HG  H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  SG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
SG  HG  SING N N
OXT HXT SING N N
##
data_CYX
#
_chem_comp.id CYX
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N R
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
SG  SG  S N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  SG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
OXT HXT SING N N
##
data_GLN
#
_comp_chem.id GLN
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N    N    N N N N
CA   CA   C N N S
C    C    C N N N
O    O    O N N N
CB   CB   C N N N
CG   CG   C N N N
CD   CD   C N N N
OE1  OE1  O N N N
NE2  NE2  N N N N
OXT  OXT  O N Y N
H    H    H N N N
H2   HN2  H N Y N
HA   HA   H N N N
HB2  1HB  H N N N
HB3  2HB  H N N N
HG2  1HG  H N N N
HG3  2HG  H N N N
HE21 1HE2 H N N N
HE22 2HE2 H N N N
HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  CG   SING N N
CB  HB2  SING N N
CB  HB3  SING N N
CG  CD   SING N N
CG  HG2  SING N N
CG  HG3  SING N N
CD  OE1  DOUB N N
CD  NE2  SING N N
NE2 HE21 SING N N
NE2 HE22 SING N N
OXT HXT  SING N N
##
data_GLY
#
_comp_chem.id GLY
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N N
C   C   C N N N
O   O   O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA2 HA1 H N N N
HA3 HA2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  HA2 SING N N
CA  HA3 SING N N
C   O   DOUB N N
C   OXT SING N N
OXT HXT SING N N
##
data_GLU
#
_chem_comp.id GLU
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
CD  CD  C N N N
OE1 OE1 O N N N
OE2 OE2 O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 HB1 H N N N
HB3 HB2 H N N N
HG2 HG1 H N N N
HG3 HG2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD  SING N N
CG  HG2 SING N N
CG  HG3 SING N N
CD  OE1 DOUB N N
CD  OE2 SING N N
OXT HXT SING N N
##
data_GLH
#
_chem_comp.id GLH
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
CD  CD  C N N N
OE1 OE1 O N N N
OE2 OE2 O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 HB1 H N N N
HB3 HB2 H N N N
HG2 HG1 H N N N
HG3 HG2 H N N N
HE2 HE2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD  SING N N
CG  HG2 SING N N
CG  HG3 SING N N
CD  OE1 DOUB N N
CD  OE2 SING N N
OE2 HE2 SING N N
OXT HXT SING N N
##
data_HIS
#
_chem_comp.id HIS
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C Y N N
ND1 ND1 N Y N N
CD2 CD2 C Y N N
CE1 CE1 C Y N N
NE2 NE2 N Y N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HD1 HD1 H N N N
HD2 HD2 H N N N
HE1 HE1 H N N N
HE2 HE2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  ND1 SING Y N
CG  CD2 DOUB Y N
ND1 CE1 DOUB Y N
ND1 HD1 SING N N
CD2 NE2 SING Y N
CD2 HD2 SING N N
CE1 NE2 SING Y N
CE1 HE1 SING N N
NE2 HE2 SING N N
OXT HXT SING N N
##
data_HIE
#
_chem_comp.id HIE
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C Y N N
ND1 ND1 N Y N N
CD2 CD2 C Y N N
CE1 CE1 C Y N N
NE2 NE2 N Y N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HD2 HD2 H N N N
HE1 HE1 H N N N
HE2 HE2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  ND1 SING Y N
CG  CD2 DOUB Y N
ND1 CE1 DOUB Y N
CD2 NE2 SING Y N
CD2 HD2 SING N N
CE1 NE2 SING Y N
CE1 HE1 SING N N
NE2 HE2 SING N N
OXT HXT SING N N
##
data_HID
#
_chem_comp.id HID
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C Y N N
ND1 ND1 N Y N N
CD2 CD2 C Y N N
CE1 CE1 C Y N N
NE2 NE2 N Y N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HD1 HD1 H N N N
HD2 HD2 H N N N
HE1 HE1 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  ND1 SING Y N
CG  CD2 DOUB Y N
ND1 CE1 SING Y N
ND1 HD1 SING N N
CD2 NE2 SING Y N
CD2 HD2 SING N N
CE1 NE2 DOUB Y N
CE1 HE1 SING N N
OXT HXT SING N N
##
data_ILE
#
_comp_chem.id ILE
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N    N    N N N N
CA   CA   C N N S
C    C    C N N N
O    O    O N N N
CB   CB   C N N S
CG1  CG1  C N N N
CG2  CG2  C N N N
CD1  CD1  C N N N
OXT  OXT  O N Y N
H    H    H N N N
H2   HN2  H N Y N
HA   HA   H N N N
HB   HB   H N N N
HG12 1HG1 H N N N
HG13 2HG1 H N N N
HG21 1HG2 H N N N
HG22 2HG2 H N N N
HG23 3HG2 H N N N
HD11 1HD1 H N N N
HD12 2HD1 H N N N
HD13 3HD1 H N N N
HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  CG1  SING N N
CB  CG2  SING N N
CB  HB   SING N N
CG1 CD1  SING N N
CG1 HG12 SING N N
CG1 HG13 SING N N
CG2 HG21 SING N N
CG2 HG22 SING N N
CG2 HG23 SING N N
CD1 HD11 SING N N
CD1 HD12 SING N N
CD1 HD13 SING N N
OXT HXT  SING N N
##
data_LEU
#
_chem_comp.id LEU
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N    N    N N N N
CA   CA   C N N S
C    C    C N N N
O    O    O N N N
CB   CB   C N N N
CG   CG   C N N N
CD1  CD1  C N N N
CD2  CD2  C N N N
OXT  OXT  O N Y N
H    H    H N N N
H2   HN2  H N Y N
HA   HA   H N N N
HB2  1HB  H N N N
HB3  2HB  H N N N
HG   HG   H N N N
HD11 1HD1 H N N N
HD12 2HD1 H N N N
HD13 3HD1 H N N N
HD21 1HD2 H N N N
HD22 2HD2 H N N N
HD23 3HD2 H N N N
HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  CG   SING N N
CB  HB2  SING N N
CB  HB3  SING N N
CG  CD1  SING N N
CG  CD2  SING N N
CG  HG   SING N N
CD1 HD11 SING N N
CD1 HD12 SING N N
CD1 HD13 SING N N
CD2 HD21 SING N N
CD2 HD22 SING N N
CD2 HD23 SING N N
OXT HXT  SING N N
##
data_LYS
#
_chem_comp.id LYS
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
CD  CD  C N N N
CE  CE  C N N N
NZ  NZ  N N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HG2 1HG H N N N
HG3 2HG H N N N
HD2 1HD H N N N
HD3 2HD H N N N
HE2 1HE H N N N
HE3 2HE H N N N
HZ1 1HZ H N N N
HZ2 2HZ H N N N
HZ3 3HZ H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD  SING N N
CG  HG2 SING N N
CG  HG3 SING N N
CD  CE  SING N N
CD  HD2 SING N N
CD  HD3 SING N N
CE  NZ  SING N N
CE  HE2 SING N N
CE  HE3 SING N N
NZ  HZ1 SING N N
NZ  HZ2 SING N N
NZ  HZ3 SING N N
OXT HXT SING N N
##
data_MET
#
_chem_comp.id MET
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
SD  SD  S N N N
CE  CE  C N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HG2 1HG H N N N
HG3 2HG H N N N
HE1 1HE H N N N
HE2 2HE H N N N
HE3 3HE H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  SD  SING N N
CG  HG2 SING N N
CG  HG3 SING N N
SD  CE  SING N N
CE  HE1 SING N N
CE  HE2 SING N N
CE  HE3 SING N N
OXT HXT SING N N
##
data_PRO
#
_chem_comp.id PRO
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C N N N
CD  CD  C N N N
OXT OXT O N Y N
H   HT1 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HG2 1HG H N N N
HG3 2HG H N N N
HD2 1HD H N N N
HD3 2HD H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   CD  SING N N
N   H   SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD  SING N N
CG  HG2 SING N N
CG  HG3 SING N N
CD  HD2 SING N N
CD  HD3 SING N N
OXT HXT SING N N
##
data_PHE
#
_chem_comp.id PHE
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C Y N N
CD1 CD1 C Y N N
CD2 CD2 C Y N N
CE1 CE1 C Y N N
CE2 CE2 C Y N N
CZ  CZ  C Y N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HD1 HD1 H N N N
HD2 HD2 H N N N
HE1 HE1 H N N N
HE2 HE2 H N N N
HZ  HZ  H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD1 DOUB Y N
CG  CD2 SING Y N
CD1 CE1 SING Y N
CD1 HD1 SING N N
CD2 CE2 DOUB Y N
CD2 HD2 SING N N
CE1 CZ  DOUB Y N
CE1 HE1 SING N N
CE2 CZ  SING Y N
CE2 HE2 SING N N
CZ  HZ  SING N N
OXT HXT SING N N
##
data_SER
#
_chem_comp.id SER
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
OG  OG  O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HG  HG  H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  OG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
OG  HG  SING N N
OXT HXT SING N N
##
data_TYR
#
_chem_comp.id TYR
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C Y N N
CD1 CD1 C Y N N
CD2 CD2 C Y N N
CE1 CE1 C Y N N
CE2 CE2 C Y N N
CZ  CZ  C Y N N
OH  OH  O N N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HD1 HD1 H N N N
HD2 HD2 H N N N
HE1 HE1 H N N N
HE2 HE2 H N N N
HH  HH  H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD1 DOUB Y N
CG  CD2 SING Y N
CD1 CE1 SING Y N
CD1 HD1 SING N N
CD2 CE2 DOUB Y N
CD2 HD2 SING N N
CE1 CZ  DOUB Y N
CE1 HE1 SING N N
CE2 CZ  SING Y N
CE2 HE2 SING N N
CZ  OH  SING N N
OH  HH  SING N N
OXT HXT SING N N
##
data_THR
_chem_comp.id THR
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N    N    N N N N
CA   CA   C N N S
C    C    C N N N
O    O    O N N N
CB   CB   C N N R
OG1  OG1  O N N N
CG2  CG2  C N N N
OXT  OXT  O N Y N
H    H    H N N N
H2   HN2  H N Y N
HA   HA   H N N N
HB   HB   H N N N
HG1  HG1  H N N N
HG21 1HG2 H N N N
HG22 2HG2 H N N N
HG23 3HG2 H N N N
HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  OG1  SING N N
CB  CG2  SING N N
CB  HB   SING N N
OG1 HG1  SING N N
CG2 HG21 SING N N
CG2 HG22 SING N N
CG2 HG23 SING N N
OXT HXT  SING N N
##
data_TRP
#
_comp_chem.id TRP
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N   N   N N N N
CA  CA  C N N S
C   C   C N N N
O   O   O N N N
CB  CB  C N N N
CG  CG  C Y N N
CD1 CD1 C Y N N
CD2 CD2 C Y N N
NE1 NE1 N Y N N
CE2 CE2 C Y N N
CE3 CE3 C Y N N
CZ2 CZ2 C Y N N
CZ3 CZ3 C Y N N
CH2 CH2 C Y N N
OXT OXT O N Y N
H   H   H N N N
H2  HN2 H N Y N
HA  HA  H N N N
HB2 1HB H N N N
HB3 2HB H N N N
HD1 HD1 H N N N
HE1 HE1 H N N N
HE3 HE3 H N N N
HZ2 HZ2 H N N N
HZ3 HZ3 H N N N
HH2 HH2 H N N N
HXT HXT H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA  SING N N
N   H   SING N N
N   H2  SING N N
CA  C   SING N N
CA  CB  SING N N
CA  HA  SING N N
C   O   DOUB N N
C   OXT SING N N
CB  CG  SING N N
CB  HB2 SING N N
CB  HB3 SING N N
CG  CD1 DOUB Y N
CG  CD2 SING Y N
CD1 NE1 SING Y N
CD1 HD1 SING N N
CD2 CE2 DOUB Y N
CD2 CE3 SING Y N
NE1 CE2 SING Y N
NE1 HE1 SING N N
CE2 CZ2 SING Y N
CE3 CZ3 DOUB Y N
CE3 HE3 SING N N
CZ2 CH2 DOUB Y N
CZ2 HZ2 SING N N
CZ3 CH2 SING Y N
CZ3 HZ3 SING N N
CH2 HH2 SING N N
OXT HXT SING N N
##
data_VAL
#
_chem_comp.id VAL
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
N    N    N N N N
CA   CA   C N N S
C    C    C N N N
O    O    O N N N
CB   CB   C N N N
CG1  CG1  C N N N
CG2  CG2  C N N N
OXT  OXT  O N Y N
H    H    H N N N
H2   HN2  H N Y N
HA   HA   H N N N
HB   HB   H N N N
HG11 1HG1 H N N N
HG12 2HG1 H N N N
HG13 3HG1 H N N N
HG21 1HG2 H N N N
HG22 2HG2 H N N N
HG23 3HG2 H N N N
HXT  HXT  H N Y N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
N   CA   SING N N
N   H    SING N N
N   H2   SING N N
CA  C    SING N N
CA  CB   SING N N
CA  HA   SING N N
C   O    DOUB N N
C   OXT  SING N N
CB  CG1  SING N N
CB  CG2  SING N N
CB  HB   SING N N
CG1 HG11 SING N N
CG1 HG12 SING N N
CG1 HG13 SING N N
CG2 HG21 SING N N
CG2 HG22 SING N N
CG2 HG23 SING N N
OXT HXT  SING N N
##
data_HOH
#
_chem_comp.id HOH
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
O  O  O N N N
H1 H1 H N N N
H2 H2 H N N N
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
O  H1 SING N N
O  H2 SING N N
##
"""
STANDARD_AA = cif_parser_lite(_STANDARD_AA)
