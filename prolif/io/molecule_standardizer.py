"""
Molecule standardizer --- :mod:`prolif.io.molecule_standardizer`
===============================================================
This module provides the :class:`MoleculeStandardizer` class for standardizing
residue names and fixing bond orders in molecular structures.

Yu-Yuan (Stuart) Yang, 2025
"""

import warnings
from collections.abc import Sequence
from pathlib import Path

import gemmi
from rdkit import Chem

from prolif.io.constants import (
    AMBER_POOL,
    CHARMM_POOL,
    GROMOS_POOL,
    OPLS_AA_POOL,
    STANDARD_AA,
    STANDARD_RESNAME_MAP,
    TERMINAL_OXYGEN_NAMES,
)
from prolif.io.template_engine import (
    CIFTemplateEngine,
    RDKitMolTemplateEngine,
    TemplateEngine,
)
from prolif.molecule import Molecule
from prolif.residue import Residue, ResidueGroup

#: Type alias for a template given as a ``(residue_name, rdkit_mol)`` pair.
MolTemplate = tuple[str, Chem.Mol]


def _build_engines(
    templates: Sequence[gemmi.cif.Document | MolTemplate] | None,
) -> dict[str, TemplateEngine]:
    """Build a flattened dict of template engines from user-supplied templates
    and the built-in STANDARD_AA.

    User-supplied templates take priority (first match wins).
    STANDARD_AA is always appended as a fallback.
    """
    engines: dict[str, TemplateEngine] = {}

    all_templates: list[gemmi.cif.Document | MolTemplate] = []
    if templates is not None:
        all_templates.extend(templates)
    # always append STANDARD_AA as fallback
    all_templates.append(STANDARD_AA)

    for template in all_templates:
        if isinstance(template, gemmi.cif.Document):
            for block in template:
                name = block.name
                if name not in engines:
                    engines[name] = CIFTemplateEngine(name, block)
        elif isinstance(template, tuple) and len(template) == 2:
            name, mol = template
            if not isinstance(name, str) or not isinstance(mol, Chem.Mol):
                raise TypeError(
                    "Mol templates must be tuples of (str, rdkit.Chem.Mol). "
                    f"Got ({type(name).__name__}, {type(mol).__name__})."
                )
            if name not in engines:
                engines[name] = RDKitMolTemplateEngine(name, mol)
        else:
            raise TypeError(
                "Templates must be gemmi.cif.Document objects or "
                f"(str, Chem.Mol) tuples. Got {type(template).__name__}."
            )

    return engines


class MoleculeStandardizer:
    """:class:`MoleculeStandardizer` standardizes residue names and fixes bond orders
    when reading molecules with RDKit.

    .. versionadded:: 2.2.0

    Parameters
    ----------
    templates : Sequence[gemmi.cif.Document | tuple[str, Chem.Mol]] | None, optional
        The templates to use for standardizing the molecule.
        Each element can be:

        - A ``gemmi.cif.Document`` (e.g. from
          :func:`~prolif.io.cif.cif_template_reader`),
          where each block in the document provides a CIF-based template.
        - A ``(residue_name, rdkit.Chem.Mol)`` tuple for an RDKit-molecule-based
          template.

        If ``None``, only the standard amino acid templates are used.
        User-supplied templates take priority over the built-in ones (first match wins).
        Default is ``None``.

    Attributes
    ----------
    engines : dict[str, TemplateEngine]
        Maps each residue name to its corresponding template engine.

    Note
    ----
    This class works with ProLIF Molecule instances or PDB files.
    It reads the input topology, standardizes the residue names, and fixes the bond
    orders based on the provided templates or the standard amino acid template.

    Example
    -------
    >>> import prolif as plf
    >>> from prolif.io import MoleculeStandardizer, cif_template_reader
    >>> from rdkit import Chem
    >>> standardizer = MoleculeStandardizer(
    ...     templates=[
    ...         cif_template_reader("path/to/ligand.cif"),
    ...         ("EDO", Chem.MolFromSmiles("OCCO")),
    ...     ]
    ... )
    >>> mol = standardizer("path/to/protein.pdb")
    >>> plf.display_residues(mol)

    """

    def __init__(
        self,
        templates: Sequence[gemmi.cif.Document | MolTemplate] | None = None,
    ) -> None:
        self.engines = _build_engines(templates)

    def __call__(self, input_topology: Molecule | Chem.Mol | str | Path) -> Molecule:
        """Standardize a molecule.

        This function will standardize the residue names, fix the bond orders,
        and check the residue names against the templates.

        Parameters
        ----------
        input_topology : Molecule | Chem.Mol | str | Path
            The input topology to standardize.
            It can be a ProLIF Molecule, an RDKit Molecule, or a path to a PDB file.

        Returns
        -------
        Molecule
            The standardized molecule.

        Example
        -------
        >>> from prolif.io import MoleculeStandardizer
        >>> standardizer = MoleculeStandardizer()
        >>> mol = standardizer("path/to/protein.pdb")

        .. important::
            If your input for `standardize` is a :class:`prolif.Molecule`, it
            will modify your original molecule in place. Your residue names will be
            updated to the standardized names and residue's bond orders will be fixed
            to the corresponding protonated states.

        """

        # read as prolif molecule
        if isinstance(input_topology, Molecule):
            protein_mol = input_topology

        elif isinstance(input_topology, Chem.Mol):
            protein_mol = Molecule.from_rdkit(input_topology)

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

            # check if we have a template for this residue
            engine = self.engines.get(standardized_resname)
            if engine is None:
                raise ValueError(
                    f"Residue {{'{standardized_resname}'}} is not a standard residue "
                    "or not in the templates. Please provide a custom template."
                )

            # soft check the heavy atoms in the residue compared to the template
            if self.n_residue_heavy_atoms(residue) != engine.n_heavy_atoms():
                warnings.warn(
                    f"Residue {residue.resid} has a different number of "
                    "heavy atoms than the standard residue. "
                    "This may affect H-bond detection.",
                    stacklevel=2,
                )

            # fix the bond orders via the engine
            try:
                fixed = engine.apply(residue)
            except Exception as e:
                raise ValueError(
                    f"Could not apply template for residue {residue.resid}: {e}"
                ) from e
            new_residues.append(fixed)

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
