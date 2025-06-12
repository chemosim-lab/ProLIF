import shlex
import warnings

import pandas as pd
from rdkit import Chem

from prolif.datafiles import datapath
from prolif.molecule import Molecule
from prolif.residue import Residue

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
    def __init__(self, input_topology: Molecule):
        """Initialize the ProteinHelper class.
        This class is designed for implicit H-bond detection.

        Parameters
        ----------
        input_topology : str or Molecule
            The path of input protein molecule (.pdb) or a ProLIF Molecule.
        """

        # read as prolif molecule
        if isinstance(input_topology, Molecule):
            self.protein_mol = input_topology

        elif isinstance(input_topology, str) and input_topology.endswith(".pdb"):
            input_protein_top = Chem.MolFromPDBFile(input_topology)
            self.protein_mol = Molecule.from_rdkit(input_protein_top)

        else:
            raise TypeError(
                "input_topology must be a string (path to a PDB file) or "
                "a prolif Molecule instance."
            )

    def standardize_protein(self, **kwargs) -> None:
        """Standardize the protein molecule."""

        # guess forcefield
        conv_resnames = set(self.protein_mol.residues.name)
        forcefield_name = self.forcefield_guesser(conv_resnames)

        # standardize the protein molecule
        for residue in self.protein_mol.residues.values():
            standard_resname = self.convert_to_standard_resname(
                resname=residue.name, forcefield_name=forcefield_name
            )

            # [TODO] fix the bond orders for non-standard residues
            # if self.check_resnames({standard_resname}):
            #     raise UserWarning()
            # self.fix_molecule_bond_orders()

            for atom in residue.GetAtoms():
                # Set the standard residue name
                atom.GetPDBResidueInfo().SetResidueName(standard_resname)

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

        elif forcefield_name == "gromos" and resname == "CYS":
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

    @staticmethod
    def fix_molecule_bond_orders(
        residue: Residue, templates: list | None = None
    ) -> None:
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
with open(str(datapath / "standard_aa.cif")) as f:
    _STANDARD_AA = f.read()

STANDARD_AA = cif_parser_lite(_STANDARD_AA)
