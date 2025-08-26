"""
Protein helper functions --- :mod:`prolif.io.protein_helper`
============================================================
This module provides helper functions for working with
protein structures (but not limited to protein structures).

Yu-Yuan (Stuart) Yang, 2025
"""

import logging
import warnings
from pathlib import Path

from rdkit import Chem

from prolif.io.constants import (
    AMBER_POOL,
    ATOMNAME_ALIASES,
    CHARMM_POOL,
    FORMAL_CHARGE_ALIASES,
    GROMOS_POOL,
    MAX_AMIDE_LENGTH,
    OPLS_AA_POOL,
    RESNAME_ALIASES,
    STANDARD_AA,
    STANDARD_RESNAME_MAP,
    TERMINAL_OXYGEN_NAMES,
)
from prolif.molecule import Molecule, pdbqt_supplier
from prolif.residue import Residue, ResidueGroup

logger = logging.getLogger(__name__)


class ProteinHelper:
    """:class:`ProteinHelper` is a class to standardize the residue names and fix the
    bond orders when reading the non-standard residues with RDKit for a molecule.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    templates : list[dict] or dict or None, optional
        The templates to use for standardizing the protein molecule.
        If `None`, the standard amino acid template is used.
        If a dict is provided, it should contain the templates for residues.
        If a list is provided, it should contain dictionaries for each template.
        Default is `None`.

    Attributes
    ----------
    templates : list[dict]
        The templates used for standardizing the protein molecule.

    Note
    ----
    This class is designed to work with ProLIF Molecule instances or PDB files.
    It reads the input topology, standardizes the residue names, and fixes the bond
    orders based on the provided templates or standard amino acid template.

    Example
    -------
    >>> import prolif as plf
    >>> from prolif.io import ProteinHelper
    >>> protein_helper = ProteinHelper(templates=[{"ALA": {"SMILES": "CC(C(=O)O)N"}}])
    >>> mol = protein_helper.standardize_protein(input_topology="path/to/protein.pdb")
    >>> plf.display_residues(mol)

    """

    def __init__(self, templates: list[dict] | dict | None = None):
        # read the templates
        if templates is None:
            self.templates = [STANDARD_AA]
        elif isinstance(templates, dict):
            # if templates is a dict, convert it to a list of dicts
            self.templates = [templates, STANDARD_AA]
        elif isinstance(templates, list):
            # if templates is a list, check if it contains dicts
            if not all(isinstance(t, dict) for t in templates):
                raise TypeError("Templates must be a dict, a list of dicts or None.")
            self.templates = [*templates, STANDARD_AA]
        else:
            raise TypeError("Templates must be a dict, a list of dicts or None.")

        # check the templates with "name" for each residue
        for template in self.templates:
            for t in template:
                # if not, set the residue name as the template name
                if "name" not in template[t]:
                    template[t]["name"] = t  # use the residue name as the template name

                elif template[t]["name"] != t:
                    warnings.warn(
                        f"Align the template name ({template[t]['name']}) with ({t}).",
                        stacklevel=2,
                    )
                    template[t]["name"] = t

        # get a dict of the number of the heavy atoms in the template residues
        self.n_template_res_hatms = self.n_template_residue_heavy_atoms(self.templates)

    def standardize_protein(self, input_topology: Molecule | str | Path) -> Molecule:
        """Standardize the protein molecule.

        This function will standardize the residue names, fix the bond orders,
        and check the residue names against the templates.

        Parameters
        ----------
        input_topology : Molecule | str | Path
            The input topology to standardize.
            It can be a ProLIF Molecule or a path to a PDB file.

        Returns
        -------
        Molecule
            The standardized protein molecule.

        Example
        -------
        >>> from prolif.io import ProteinHelper
        >>> protein_helper = ProteinHelper(
                templates=[{"ALA": {"SMILES": "CC(C(=O)O)N"}}]
            )
        >>> mol = protein_helper.standardize_protein(
               "path/to/protein.pdb"
            )

        .. important::
            If your input for `standardize_protein` is a :class:`prolif.Molecule`, it
            will modify your original molecule in place. Your residue names will be
            updated to the standardized names and residue's bond orders will be fixed
            to the corresponding protonated states.

        """

        # read as prolif molecule
        if isinstance(input_topology, Molecule):
            protein_mol = input_topology

        elif isinstance(input_topology, str | Path) and str(input_topology).endswith(
            ".pdb"
        ):
            input_protein_top = Chem.MolFromPDBFile(str(input_topology), removeHs=False)
            protein_mol = Molecule.from_rdkit(input_protein_top)

        else:
            raise TypeError(
                "input_topology must be a string (path to a PDB file) or "
                "a prolif Molecule instance."
            )

        # guess forcefield
        conv_resnames = set(protein_mol.residues.name)
        forcefield_name = self.forcefield_guesser(conv_resnames)

        # standardize the protein molecule
        new_residues = []
        for residue in protein_mol.residues.values():
            standardized_resname = self.convert_to_standard_resname(
                resname=residue.resid.name.upper(), forcefield_name=forcefield_name
            )

            # set the standard residue name
            for atom in residue.GetAtoms():
                # set new residue name for each atom (at residue level)
                atom.GetPDBResidueInfo().SetResidueName(standardized_resname)
                # set the new residue name for each atom at the molecule level
                protein_mol.GetAtomWithIdx(
                    atom.GetUnsignedProp("mapindex")
                ).GetPDBResidueInfo().SetResidueName(standardized_resname)
            # update the residue name in the Residue object
            residue.resid.name = standardized_resname

            # before fixing the bond orders: strict check with non-standard residues
            self.check_resnames({standardized_resname}, self.templates)

            # soft check the heavy atoms in the residue compared to the standard one
            if self.n_residue_heavy_atoms(residue) != self.n_template_res_hatms.get(
                standardized_resname, 0
            ):
                warnings.warn(
                    f"Residue {residue.resid} has a different number of "
                    "heavy atoms than the standard residue. "
                    "This may affect H-bond detection.",
                    stacklevel=2,
                )
            # fix the bond orders
            new_residues.append(self.fix_molecule_bond_orders(residue, self.templates))

        # update the protein molecule with the new residues
        protein_mol.residues = ResidueGroup(new_residues)

        return protein_mol

    @staticmethod
    def forcefield_guesser(
        conv_resnames: set[str],
    ) -> str:
        """Guess the forcefield based on the residue names.

        Parameters
        ----------
        conv_resnames : set[str]
            Set of residue names in the protein molecule.

        Returns
        -------
        str
            The guessed forcefield name.
        """
        if AMBER_POOL.intersection(conv_resnames):
            return "amber"
        if CHARMM_POOL.intersection(conv_resnames):
            return "charmm"
        if GROMOS_POOL.intersection(conv_resnames):
            return "gromos"
        if OPLS_AA_POOL.intersection(conv_resnames):
            return "oplsaa"

        return "unknown"

    @staticmethod
    def convert_to_standard_resname(
        resname: str, forcefield_name: str = "unknown"
    ) -> str:
        """Convert a residue name to its standard form based on the forcefield.

        Parameters
        ----------
        resname : str
            The residue name to convert.
        forcefield_name : str
            The name of the forcefield for assigning the correct standard name for CYS.
            Default is "unknown".

        Returns
        -------
        str
            The standard residue name.

        Note
        ----
        This conversion is designed to distinguish residues with
        different possible H-bond donors at side chains, instead of the
        actual protonated states of residues.

        For example, neutral and protonated arginine are both assigned to ARG,
        while neutral and deprotonated arginine in GROMOS force field
        are assigned to ARGN and ARG, respectively.
        """

        if forcefield_name == "unknown" and resname == "CYS":
            warnings.warn(
                "Could not guess the forcefield based on the residue names. "
                "CYS is assigned to neutral CYS (charge = 0).",
                stacklevel=2,
            )

        elif forcefield_name == "gromos" and resname == "CYS":
            return "CYX"

        return STANDARD_RESNAME_MAP.get(resname, resname)

    @staticmethod
    def check_resnames(
        resnames_to_check: set[str], templates: list[dict] | None = None
    ) -> None:
        """Check if the residue names are standard or in templates and
        raise a warning if not.

        Parameters
        ----------
        resnames_to_check : set[str]
            Set of residue names to check.
        templates : list[dict] or None, optional
            The templates to use for checking the residue names.
            If `None`, the standard amino acid template is used.
            Default is `None`.

        Raises
        ------
        ValueError
            If any residue name is not standard or not in the templates.

        """

        if templates is None:
            templates = [STANDARD_AA]

        all_available_resnames = set().union(*list(templates))

        non_standard_resnames = resnames_to_check - all_available_resnames
        if non_standard_resnames:
            raise ValueError(
                f"Residue {non_standard_resnames} is not a standard residue or "
                "not in the templates. Please provide a custom template."
            )

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

        return len(
            [
                atom
                for atom in residue.GetAtoms()
                if atom.GetAtomicNum() != 1
                and atom.GetPDBResidueInfo().GetName().strip()
                not in TERMINAL_OXYGEN_NAMES
            ]
        )

    @staticmethod
    def n_template_residue_heavy_atoms(
        templates: list[dict] | None = None,
    ) -> dict[str, int]:
        """Count the number of heavy atoms in a residue based on the templates.

        Parameters
        ----------
        templates : list or None, optional
            The templates to use for counting the heavy atoms.
            If `None`, the standard amino acid template is used.
            Default is `None`.

        Returns
        -------
        dict[str, int]
            The dictionary with residue names as keys and
            the number of heavy atoms as values.
        """
        # Check if templates are provided,
        # otherwise use the standard amino acid template
        if templates is None:
            templates = [STANDARD_AA]

        n_template_residue_heavy_atoms = {}
        for t in templates:
            for resname in t:
                if resname in n_template_residue_heavy_atoms:
                    continue

                # SMILES template
                if "SMILES" in t[resname]:
                    t_mol = Chem.MolFromSmiles(t[resname]["SMILES"])
                    n_template_residue_heavy_atoms[resname] = len(
                        [at for at in t_mol.GetAtoms() if at.GetAtomicNum() != 1]
                    )
                    continue

                # CIF template
                residue_atom_df = t[resname]["_chem_comp_atom"]
                residue_atom_df = residue_atom_df[
                    residue_atom_df["alt_atom_id"] != "OXT"
                ]

                n_template_residue_heavy_atoms[resname] = sum(
                    residue_atom_df["type_symbol"] != "H"
                )

        return n_template_residue_heavy_atoms

    @staticmethod
    def fix_molecule_bond_orders(
        residue: Residue, templates: list[dict] | None = None
    ) -> Residue:
        r"""Fix the bond orders of a molecule.

        Parameters
        ----------
        residue : Residue
            The residue to fix the bond orders for.
        templates : list[dict] or None, optional
            The templates to use for fixing the bond orders.
            If `None`, the standard amino acid template is used.
            Default is `None`. If the residue is not a standard amino acid,
            the user should provide a custom template.

            Also, note that the order of templates is relevant,
            as the first template that matches the residue name will be used.

        Note
        ----
        1\. If the user provides a SMILES template, it will be converted to an RDKit
        molecule, and the bond orders will be assigned from the template.

        2\. SMILES templates are prioritized over CIF templates.

        3\. For CIF templates, any bonds and chiral designation on the input molecule
        will be removed at the start of the process. This function is adapted from the
        `pdbinf/_pdbinf.py` module's `assign_pdb_bonds` function, which is used to
        assign bonds and aromaticity based on the standard amino acid templates.

        Source: https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_pdbinf.py#L482
        """
        if templates is None:
            templates = [STANDARD_AA]
        resname = residue.resid.name.upper()

        # read the templates and assign correct bond orders
        for t in templates:
            # check if resname in template doc
            if resname not in t:
                continue

            # SMILES template
            if "SMILES" in t[resname]:
                new_res = assign_bond_orders_from_smiles(
                    template_smiles=t[resname]["SMILES"], mol=residue
                )
                break

            # cif template
            new_res = strip_bonds(residue)  # strip bonds and chiral tags
            new_res = assign_intra_props(new_res, t[resname])
            break  # avoid double assignment, ordering of templates is relevant

        else:
            raise ValueError(f"Failed to find template for residue: '{resname}'")

        return Residue(new_res)


def assign_bond_orders_from_smiles(
    template_smiles: str, mol: Residue | Chem.Mol
) -> Chem.Mol:
    """Assign bond orders from a SMILES template to a residue.

    Parameters
    ----------
    template_smiles : str
        The SMILES string of the template to assign bond orders from.
    residue : Residue | Chem.Mol
        The residue or molecule to assign bond orders to.

    Returns
    -------
    Chem.Mol
        The residue or molecule with assigned bond orders from the template.
    """
    for atm in mol.GetAtoms():
        # Set the necessary property for _adjust_hydrogens function (index of the atom)
        atm.SetIntProp("_MDAnalysis_index", atm.GetIdx())

    # Call template from SMILES
    mol_template = Chem.MolFromSmiles(template_smiles)

    # use the available function to assign bond orders (adjust hydrogens)
    return pdbqt_supplier._adjust_hydrogens(mol_template, mol)


# The below code contains functions for fixing molecule bond orders.
# The code is adapted from pdbinf: https://github.com/OpenFreeEnergy/pdbinf/tree/main
# Accessed on: 16 June, 2025 under MIT License.
def strip_bonds(m: Chem.Mol) -> Chem.Mol:
    """Strip all bonds and chiral tags from a molecule.

    Parameters
    ----------
    m : rdkit.Chem.Mol
        The input molecule to strip bonds from.

    Returns
    -------
    rdkit.Chem.Mol
        The modified molecule with all bonds and chiral tags removed.

    Note
    ----
    This function is adapted from the `pdbinf/_pdbinf.py` module's `strip_bonds`
    function.

    Source: https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_pdbinf.py#L71
    """
    with Chem.RWMol(m) as em:
        for b in m.GetBonds():
            em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        # rdkit, perhaps rightfully, gets upset at chiral tags w/ no bonds

        for at in em.GetAtoms():
            at.SetChiralTag(Chem.CHI_UNSPECIFIED)

    return em.GetMol()


def assign_intra_props(mol: Chem.Mol, reference_block: dict) -> Chem.Mol:
    """Assign bonds and aromaticity based on residue and atom names
    from a reference block (cif template molecule).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The input molecule, this is modified in place and returned.
    reference_block : dict
        The cif template molecule from which to assign bonds.

    Returns
    -------
    rdkit.Chem.Mol
        The modified molecule with assigned bonds and aromaticity.

    Note
    ----
    This function is adapted from the `pdbinf/_pdbinf.py` module's `assign_intra_props`
    function, which is used to assign bonds and aromaticity based on
    the standard amino acid templates.

    Source: https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_pdbinf.py#L117
    """
    nm_2_idx = {}
    fc_special_assign_nm_idx_fc_pair = {}

    # convert indices to names
    res_alias = RESNAME_ALIASES.get(reference_block["name"], reference_block["name"])
    aliases = ATOMNAME_ALIASES.get(res_alias, {})
    for atom in mol.GetAtoms():
        nm = atom.GetMonomerInfo().GetName().strip()
        nm = aliases.get(nm, nm)
        if (
            reference_block["name"] in FORMAL_CHARGE_ALIASES
            and nm in FORMAL_CHARGE_ALIASES[reference_block["name"]]
        ):
            fc_special_assign_nm_idx_fc_pair[atom.GetIdx()] = FORMAL_CHARGE_ALIASES[
                reference_block["name"]
            ][nm]
        nm_2_idx[nm] = atom.GetIdx()

    logger.debug(f"assigning intra props for {reference_block['name']}")

    with Chem.RWMol(mol) as em:
        # grab bond data from cif Block
        for _, row in reference_block["_chem_comp_bond"].iterrows():
            nm1, nm2 = row["atom_id_1"], row["atom_id_2"]
            order, arom = row["value_order"], row["pdbx_aromatic_flag"]
            try:
                idx1, idx2 = nm_2_idx[nm1], nm_2_idx[nm2]
            except KeyError:
                continue

            # [different from pdbinf] we priorituze SINGLE and DOUBLE bonds
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

    # find lone hydrogens, then attach to the nearest heavy atom
    em = _assign_intra_props_lone_H(em)

    # assign aromaticity for atoms based on the template
    for _, row in reference_block["_chem_comp_atom"].iterrows():
        nm, arom = row["atom_id"].strip(), row["pdbx_aromatic_flag"].strip()
        try:
            idx = nm_2_idx[nm]
        except KeyError:
            # todo: could check atom is marked as leaving atom
            continue

        atom = em.GetAtomWithIdx(idx)
        atom.SetIsAromatic(arom == "Y")

    # [different from pdbinf] assign formal charge for a specific atom
    if fc_special_assign_nm_idx_fc_pair != {}:
        for (
            fc_special_assign_nm_idx,
            fc_to_assign,
        ) in fc_special_assign_nm_idx_fc_pair.items():
            atom = em.GetAtomWithIdx(fc_special_assign_nm_idx)
            atom.SetFormalCharge(fc_to_assign)
            logger.debug(
                f"Assigned {reference_block['name']}'s formal charge {fc_to_assign} "
                f"on {atom.GetPDBResidueInfo().GetName()}."
            )

    # [different from pdbinf] sanitize the molecule
    mol = em.GetMol()
    mol.UpdatePropertyCache()
    Chem.SanitizeMol(mol)

    return mol


def _assign_intra_props_lone_H(em: Chem.RWMol) -> Chem.RWMol:
    """Assign lone hydrogens to the nearest heavy atom in the molecule.
    This is a part function for assign_intra_props.

    Parameters
    ----------
    em : rdkit.Chem.RWMol
        The input editable molecule to assign lone hydrogens to the nearest heavy atom.

    Returns
    -------
    rdkit.Chem.RWMol
        The modified molecule with lone hydrogens assigned to the nearest heavy atom.

    Note
    ----
    This function is adapted from the `pdbinf/_pdbinf.py` module's
    `assign_intra_props` function.

    Source: https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_pdbinf.py#L167
    """
    additional_bonds = []
    heavy_atoms = []
    lone_H = []
    for atom in em.GetAtoms():
        if atom.GetAtomicNum() != 1:
            heavy_atoms.append(atom.GetIdx())
            continue
        if atom.GetBonds():  # if any bonds, we're ok
            continue
        lone_H.append(atom.GetIdx())
    if lone_H:
        logger.debug(f"found lone hydrogens: {lone_H}")
        conf = em.GetConformer()
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
        for i, j in additional_bonds:
            em.AddBond(i, j, order=Chem.BondType.SINGLE)

    return em
