import warnings

import MDAnalysis as mda

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
    # HID Neutral (neutral, hydrogen at ND1, might couple to HEME at NE2)
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
    def __init__(self, input_pdb_path: str):
        """Initialize the ProteinHelper class.

        Parameters
        ----------
        input_pdb_path : str
            The path of input protein molecule.
        """

        from prolif.molecule import Molecule

        # read as prolif molecule
        input_protein_top = mda.Universe(input_pdb_path)
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

    def standardize_protein(self):
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
    ):
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
    def check_resnames(resnames_to_check: set[str]) -> None:
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

    def fix_molecule_bond_orders(self):
        """Fix the bond orders of a molecule."""
        raise NotImplementedError()
        # [TODO] fix the bond orders for non-standard residues
