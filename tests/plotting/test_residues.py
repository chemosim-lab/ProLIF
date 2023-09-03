import pytest
from PIL.PngImagePlugin import PngImageFile

import prolif as plf


@pytest.mark.parametrize(
    "kwargs, expected_size",
    [
        ({"residues_slice": slice(10)}, "width='800px' height='420px'"),
        ({"residues_slice": slice(10, 20)}, "width='800px' height='420px'"),
        ({"residues_slice": slice(0, 20, 2)}, "width='800px' height='420px'"),
        ({"size": (100, 100)}, "width='400px' height='1400px'"),
        ({"mols_per_row": 2}, "width='400px' height='3920px'"),
    ],
)
def test_display_residues_svg(
    protein_mol: plf.Molecule, kwargs: dict, expected_size: str
) -> None:
    img = plf.display_residues(protein_mol, **kwargs)
    assert "<svg" in img
    assert expected_size in img


def test_display_residues_png(protein_mol: plf.Molecule) -> None:
    img = plf.display_residues(protein_mol, slice(10), use_svg=False)
    assert isinstance(img, PngImageFile)
    assert img.size == (800, 420)
