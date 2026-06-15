"""
Template engines --- :mod:`prolif.io.template_engine`
=====================================================
This module defines the :class:`TemplateEngine` protocol and its concrete
implementations for fixing bond orders on residues.

.. versionadded:: 2.2.0
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import gemmi
from rdkit import Chem

from prolif.io.constants import (
    ATOMNAME_ALIASES,
    FORMAL_CHARGE_ALIASES,
    MAX_AMIDE_LENGTH,
    RESNAME_ALIASES,
)
from prolif.molecule import pdbqt_supplier
from prolif.residue import Residue

logger = logging.getLogger(__name__)


@runtime_checkable
class TemplateEngine(Protocol):
    """Protocol for template engines that fix bond orders on a residue."""

    @property
    def name(self) -> str:
        """The residue name this template matches."""
        ...

    def n_heavy_atoms(self) -> int:
        """Number of heavy atoms in this template (excluding OXT)."""
        ...

    def apply(self, residue: Residue) -> Residue:
        """Fix the bond orders of the residue using this template."""
        ...


class RDKitMolTemplateEngine:
    """Template engine using an RDKit Mol object.

    Parameters
    ----------
    name : str
        The residue name this template matches.
    mol : Chem.Mol
        The RDKit molecule to use as a template.

    """

    def __init__(self, name: str, mol: Chem.Mol) -> None:
        self._name = name
        self._mol = mol

    @property
    def name(self) -> str:
        return self._name

    def n_heavy_atoms(self) -> int:
        return sum(1 for at in self._mol.GetAtoms() if at.GetAtomicNum() != 1)

    def apply(self, residue: Residue) -> Residue:
        new_res = assign_bond_orders_from_template(template_mol=self._mol, mol=residue)
        return Residue(new_res)


class CIFTemplateEngine:
    """Template engine using a gemmi CIF block.

    Parameters
    ----------
    name : str
        The residue name this template matches.
    block : gemmi.cif.Block
        The gemmi CIF block containing the bond and atom data.

    """

    def __init__(self, name: str, block: gemmi.cif.Block) -> None:
        self._name = name
        self._block = block

    @property
    def name(self) -> str:
        return self._name

    def n_heavy_atoms(self) -> int:
        atom_table = self._block.find(
            "_chem_comp_atom.", ["alt_atom_id", "type_symbol"]
        )
        return sum(
            1 for row in atom_table if row[0].strip() != "OXT" and row[1].strip() != "H"
        )

    def apply(self, residue: Residue) -> Residue:
        new_res = strip_bonds(residue)
        new_res = assign_intra_props(new_res, self._name, self._block)
        return Residue(new_res)


def assign_bond_orders_from_template(
    template_mol: Chem.Mol, mol: Residue | Chem.Mol
) -> Chem.Mol:
    """Assign bond orders from an RDKit template molecule to a residue.

    Parameters
    ----------
    template_mol : Chem.Mol
        The template molecule with correct bond orders.
    mol : Residue | Chem.Mol
        The residue or molecule to assign bond orders to.

    Returns
    -------
    Chem.Mol
        The molecule with assigned bond orders from the template.

    .. versionchanged:: 2.2.0
        Renamed from ``assign_bond_orders_from_smiles`` and now takes an RDKit
        ``Mol`` object instead of a SMILES string.
    """

    for atm in mol.GetAtoms():
        # Set the necessary property for _adjust_hydrogens function
        atm.SetIntProp("_MDAnalysis_index", atm.GetIdx())

    return pdbqt_supplier._adjust_hydrogens(template_mol, mol)


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


_BOND_ORDER_MAP = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
}


def assign_intra_props(
    mol: Chem.Mol, residue_name: str, block: gemmi.cif.Block
) -> Chem.Mol:
    """Assign bonds and aromaticity based on residue and atom names
    from a gemmi CIF block.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The input molecule, this is modified in place and returned.
    residue_name : str
        The name of the residue being processed.
    block : gemmi.cif.Block
        The gemmi CIF block from which to assign bonds.

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

    .. versionchanged:: 2.2.0
        Now takes a ``gemmi.cif.Block`` instead of a custom dict, and added
        support for triple bonds (``TRIP``).
    """
    nm_2_idx: dict[str, int] = {}
    fc_special_assign_nm_idx_fc_pair: dict[int, int] = {}

    # convert indices to names
    res_alias = RESNAME_ALIASES.get(residue_name, residue_name)
    aliases = ATOMNAME_ALIASES.get(res_alias, {})
    for atom in mol.GetAtoms():
        nm = atom.GetMonomerInfo().GetName().strip()
        nm = aliases.get(nm, nm)
        if (
            residue_name in FORMAL_CHARGE_ALIASES
            and nm in FORMAL_CHARGE_ALIASES[residue_name]
        ):
            fc_special_assign_nm_idx_fc_pair[atom.GetIdx()] = FORMAL_CHARGE_ALIASES[
                residue_name
            ][nm]
        nm_2_idx[nm] = atom.GetIdx()

    logger.debug(f"assigning intra props for {residue_name}")

    bond_table = block.find(
        "_chem_comp_bond.",
        ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"],
    )

    with Chem.RWMol(mol) as em:
        for row in bond_table:
            nm1, nm2 = row[0], row[1]
            order_str, arom = row[2], row[3]
            try:
                idx1, idx2 = nm_2_idx[nm1], nm_2_idx[nm2]
            except KeyError:
                continue

            # [different from pdbinf] we prioritize SINGLE, DOUBLE, TRIPLE bonds
            order = _BOND_ORDER_MAP.get(order_str)
            if order is None:
                order = (
                    Chem.BondType.AROMATIC if arom == "Y" else Chem.BondType.UNSPECIFIED
                )

            logger.debug(f"adding bond: {nm1}-{nm2} at {idx1} {idx2} {order}")

            em.AddBond(idx1, idx2, order=order)

    # find lone hydrogens, then attach to the nearest heavy atom
    em = _assign_intra_props_lone_H(em)

    # assign aromaticity for atoms based on the template
    atom_table = block.find("_chem_comp_atom.", ["atom_id", "pdbx_aromatic_flag"])
    for row in atom_table:
        nm, arom = row[0].strip(), row[1].strip()
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
                f"Assigned {residue_name}'s formal charge {fc_to_assign} "
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
