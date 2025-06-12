import logging
import shlex
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from prolif.constants import (
    _STANDARD_AA,
    ATOMNAME_ALIASES,
    MAX_AMIDE_LENGTH,
    amber_pool,
    charmm_pool,
    formal_charge_special_assign_map,
    gromos_pool,
    opls_aa_pool,
    resnames_map,
)
from prolif.molecule import Molecule
from prolif.residue import Residue, ResidueGroup

logger = logging.getLogger(__name__)


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
        new_mol = []
        for residue in self.protein_mol.residues.values():
            standardized_resname = self.convert_to_standard_resname(
                resname=residue.resid.name.upper(), forcefield_name=forcefield_name
            )

            # set the standard residue name
            for atom in residue.GetAtoms():
                atom.GetPDBResidueInfo().SetResidueName(standardized_resname)
            residue.resid.name = standardized_resname

            # before fixing the bond orders: strict check with non-standard residues
            assert self.check_resnames({standardized_resname}, **kwargs), (
                f"Residue {residue.resid} is not a standard residue or "
                "not in the templates. Please provide a custom template."
            )
            # soft check the heavy atoms in the residue compared to the standard one
            if self.n_residue_heavy_atoms(
                residue
            ) != self.n_standard_residue_heavy_atoms(standardized_resname):
                warnings.warn(
                    f"Residue {residue.resid} has a different number of "
                    "heavy atoms than the standard residue. "
                    "This may affect H-bond detection.",
                    stacklevel=2,
                )
            # fix the bond orders
            new_mol.append(self.fix_molecule_bond_orders(residue, **kwargs))

        self.protein_mol.residues = ResidueGroup(new_mol)

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

        return resnames_map.get(resname, resname)

    @staticmethod
    def check_resnames(
        resnames_to_check: set[str], templates: list[dict] | None = None
    ) -> bool:
        """Check if the residue names are standard or in templates and
        raise a warning if not.

        Parameters
        ----------
        resnames_to_check : set[str]
            Set of residue names to check.

        Raises
        ------
        UserWarning
            If any residue name is not standard or not in the templates.

        Returns
        -------
        bool
            True if all residue names are standard or in the templates,
            False otherwise.
        """

        if templates is None:
            templates = [STANDARD_AA]

        all_available_resnames = set()
        for t in templates:
            all_available_resnames.update(t.keys())

        non_standard_resnames = resnames_to_check - all_available_resnames
        if non_standard_resnames:
            warnings.warn(
                "Non-standard residue (or ligand/solvent) names found "
                f"not in templates: {non_standard_resnames}",
                stacklevel=2,
            )
            return False
        return True

    @staticmethod
    def n_residue_heavy_atoms(residue: Residue) -> int:
        """Count the number of heavy atoms in a residue.

        Parameters
        ----------
        residue : Residue
            The residue to count the heavy atoms in.

        Returns
        -------
        int
            The number of heavy atoms in the residue.
        """
        terminal_oxygen = {
            key
            for key, value in ATOMNAME_ALIASES["Protein"].items()
            if value.strip() == "OXT"
        }
        terminal_oxygen.add("OXT")
        return len(
            [
                atom
                for atom in residue.GetAtoms()
                if atom.GetSymbol().strip() != "H"
                and atom.GetPDBResidueInfo().GetName().strip() not in terminal_oxygen
            ]
        )

    @staticmethod
    def n_standard_residue_heavy_atoms(resname: str) -> int:
        """Count the number of heavy atoms in a standard residue.

        Parameters
        ----------
        resname : str
            The residue name to count the heavy atoms in.

        Returns
        -------
        int
            The number of heavy atoms in the standard residue.
        """
        if resname not in STANDARD_AA:
            raise ValueError(f"Residue {resname} is not a standard residue.")
        residue_atom_df = STANDARD_AA[resname]["_chem_comp_atom"]
        residue_atom_df = residue_atom_df[residue_atom_df["alt_atom_id"] != "OXT"]

        return sum(residue_atom_df["type_symbol"] != "H")

    @staticmethod
    def fix_molecule_bond_orders(
        residue: Residue, templates: list | None = None
    ) -> Residue:
        """Fix the bond orders of a molecule.

        Parameters
        ----------
        residue : Residue
            The residue to fix the bond orders for.
        templates : str or list of str, optional
            The templates to use for fixing the bond orders.
            If `None`, the standard amino acid template is used.
            Default is `None`. If the residue is not a standard amino acid,
            the user should provide a custom template.

            Also, note that the order of templates is relevant,
            as the first template that matches the residue name will be used.

        Note
        ----
        Any bonds and chiral designation on the input molecule will be removed
        at the start of the process.
        This function is adapted from the pdbinf/_pdbinf.py module's assign_pdb_bonds
        function, which is used to assign bonds and aromaticity based on
        the standard amino acid templates.
        """
        templates = [STANDARD_AA] if templates is None else [templates, STANDARD_AA]
        resname = residue.resid.name.upper()

        # strip bonds and chiral tags
        new_res = strip_bonds(residue)

        # assign properties inside each Residue
        valence = np.zeros(new_res.GetNumAtoms(), dtype=int)

        for t in templates:
            # check if resname in template doc
            if resname not in t:
                continue
            t[resname]["name"] = resname  # add name for the reference block
            new_res, v = assign_intra_props(new_res, t[resname])
            valence += v
            break  # avoid double assignment, ordering of templates is relevant
        else:
            raise ValueError(f"Failed to find template for residue: '{resname}'")

        return Residue(new_res)


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


def strip_bonds(m: Chem.Mol) -> Chem.Mol:
    em = AllChem.EditableMol(m)

    for b in m.GetBonds():
        em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    # rdkit, perhaps rightfully, gets upset at chiral tags w/ no bonds
    m = em.GetMol()
    for at in m.GetAtoms():
        at.SetChiralTag(Chem.CHI_UNSPECIFIED)
    return m


def assign_intra_props(
    mol: Chem.Mol, reference_block: dict
) -> tuple[Chem.Mol, np.ndarray]:
    """assign bonds and aromaticity based on NAMES

    Parameters
    ----------
    mol : rdkit.Chem.Mol
      the input molecule, this is modified in place and returned
    reference_block : dict
      the template molecule from which to assign bonds

    Returns
    -------
    tuple[rdkit.Chem.Mol, np.ndarray]
      the modified molecule and an array of valences for each atom

    Notes
    -----
    This function is adapted from the pdbinf/_pdbinf.py module's assign_intra_props
    """
    nm_2_idx = {}

    # convert indices to names
    fc_special_assign_nm_idx_fc_pair = {}
    aliases = ATOMNAME_ALIASES.get(reference_block["name"], {})
    for atom in mol.GetAtoms():
        nm = atom.GetMonomerInfo().GetName().strip()
        nm = aliases.get(nm, nm)
        if (
            reference_block["name"] in formal_charge_special_assign_map
            and nm in formal_charge_special_assign_map[reference_block["name"]]
        ):
            fc_special_assign_nm_idx_fc_pair[atom.GetIdx()] = (
                formal_charge_special_assign_map[reference_block["name"]][nm]
            )
        nm_2_idx[nm] = atom.GetIdx()

    logger.debug(f"assigning intra props for {reference_block['name']}")

    em = AllChem.EditableMol(mol)

    # we'll assign rdkit SINGLE, DOUBLE or AROMATIC bonds
    # but we'll also want to know the original *valence*
    valence = np.zeros(mol.GetNumAtoms(), dtype=int)

    # grab bond data from cif Block
    for _, row in reference_block["_chem_comp_bond"].iterrows():
        nm1, nm2 = row["atom_id_1"], row["atom_id_2"]
        order, arom = row["value_order"], row["pdbx_aromatic_flag"]
        try:
            idx1, idx2 = nm_2_idx[nm1], nm_2_idx[nm2]
        except KeyError:
            continue

        v = 1 if order == "SING" else 2  # 'DOUB'
        valence[idx1] += v
        valence[idx2] += v

        if order == "SING":
            order = Chem.BondType.SINGLE
        elif order == "DOUB":
            order = Chem.BondType.DOUBLE
        elif arom == "Y":
            order = Chem.BondType.AROMATIC
        else:
            order = Chem.BondType.UNSPECIFIED

        logger.debug(f"adding bond: {nm1}-{nm2} at {idx1} {idx2} {order}")

        em.AddBond(idx1, idx2, order=order)

    mol = em.GetMol()

    # find lone hydrogens, then attach to the nearest heavy atom
    _assign_intra_props_lone_H(mol)

    for _, row in reference_block["_chem_comp_atom"].iterrows():
        nm, arom = row["atom_id"].strip(), row["pdbx_aromatic_flag"].strip()
        try:
            idx = nm_2_idx[nm]
        except KeyError:
            # todo: could check atom is marked as leaving atom
            continue

        atom = mol.GetAtomWithIdx(idx)
        atom.SetIsAromatic(arom == "Y")

    # assign formal charge for a specific atom
    if fc_special_assign_nm_idx_fc_pair != {}:
        for (
            fc_special_assign_nm_idx,
            fc_to_assign,
        ) in fc_special_assign_nm_idx_fc_pair.items():
            atom = mol.GetAtomWithIdx(fc_special_assign_nm_idx)
            atom.SetFormalCharge(fc_to_assign)
            logger.debug(
                f"Assigned {reference_block['name']}'s formal charge {fc_to_assign} "
                f"on {atom.GetPDBResidueInfo().GetName()}."
            )

    # sanitize the molecule
    mol.UpdatePropertyCache()
    Chem.SanitizeMol(mol)

    return mol, valence


def _assign_intra_props_lone_H(mol):
    additional_bonds = []
    heavy_atoms = []
    lone_H = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            heavy_atoms.append(atom.GetIdx())
            continue
        if atom.GetBonds():  # if any bonds, we're ok
            continue
        lone_H.append(atom.GetIdx())
    if lone_H:
        logger.debug(f"found lone hydrogens: {lone_H}")
        conf = mol.GetConformer()
        for idx in lone_H:
            pos = conf.GetAtomPosition(idx)
            minidx, mindist = -1, float("inf")
            for idx2 in heavy_atoms:
                pos2 = conf.GetAtomPosition(idx2)
                d = pos.Distance(pos2)
                if d > mindist:
                    continue
                minidx, mindist = idx2, d

            if mindist < MAX_AMIDE_LENGTH:
                logger.debug(
                    f"attached hydrogen {idx} to heavy atom {minidx} d={mindist}"
                )
                additional_bonds.append((idx, minidx))
    if additional_bonds:
        em = Chem.EditableMol(mol)
        for i, j in additional_bonds:
            em.AddBond(i, j, order=Chem.BondType.SINGLE)
        mol = em.GetMol()

    return mol


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
STANDARD_AA = cif_parser_lite(_STANDARD_AA)
