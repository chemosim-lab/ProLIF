"""
Plot residues --- :mod:`prolif.plotting.residues`
=================================================

.. versionadded:: 2.0.0

.. autofunction:: display_residues

"""

from typing import Any

from rdkit import Chem
from rdkit.Chem import Draw

from prolif.molecule import Molecule


def display_residues(
    mol: Molecule,
    residues_slice: slice | None = None,
    *,
    size: tuple[int, int] = (200, 140),
    mols_per_row: int = 4,
    use_svg: bool = True,
    sanitize: bool = True,
) -> Any:
    """Display a grid image of the residues in the molecule. The hydrogens are stripped
    and the 3D coordinates removed for a clearer visualisation.

    Parameters
    ----------
    mol: prolif.Molecule
        The molecule to show residues from.
    residues_slice: slice | None = None
        Optionally, a slice of residues to display, e.g. ``slice(20)`` for the first 20
        residues, or ``slice(<start>, <stop>, <step>)`` for a more complex selection.
    size: tuple[int, int] = (200, 140)
        Size of each residue image.
    mols_per_row: int = 4
        Number of residues displayed per row.
    use_svg: bool = True
        Generate an SVG or PNG image.
    sanitize: bool = True
        Sanitize the residues before displaying.

    .. versionchanged:: 2.1.0
        Added ``sanitize`` parameter that defaults to False for easier debugging.
    """
    frags = []
    residues_iterable = (
        mol if residues_slice is None else mol.residues.select(residues_slice).values()
    )
    ipython_kwargs = (
        {"maxMols": mol.n_residues} if hasattr(Chem.Mol, "_repr_svg_") else {}
    )

    for residue in residues_iterable:
        resmol = Chem.RemoveHs(residue, sanitize=sanitize)
        resmol.RemoveAllConformers()
        resmol.SetProp("_Name", str(residue.resid))
        frags.append(resmol)

    return Draw.MolsToGridImage(
        frags,
        legends=[mol.GetProp("_Name") for mol in frags],
        subImgSize=size,
        molsPerRow=mols_per_row,
        useSVG=use_svg,
        **ipython_kwargs,
    )
