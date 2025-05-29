import builtins
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prolif.typeshed import InteractionMetadata

IS_NOTEBOOK = hasattr(builtins, "__IPYTHON__")

separated_interaction_colors = {
    "Hydrophobic": "#59e382",
    "VdWContact": "#dfab43",
    "HBAcceptor": "#59bee3",
    "HBDonor": "#239fcd",
    "XBAcceptor": "#ff9f02",
    "XBDonor": "#ce8000",
    "Cationic": "#e35959",
    "Anionic": "#5979e3",
    "CationPi": "#e359d8",
    "PiCation": "#ea85e2",
    "PiStacking": "#b559e3",
    "EdgeToFace": "#c885ea",
    "FaceToFace": "#a22ddc",
    "MetalAcceptor": "#7da982",
    "MetalDonor": "#609267",
    "WaterBridge": "#323aa8",
}

grouped_interaction_colors = {
    "Hydrophobic": "#59e382",
    "VdWContact": "#dfab43",
    "HBAcceptor": "#59bee3",
    "HBDonor": "#59bee3",
    "XBAcceptor": "#ff9f02",
    "XBDonor": "#ff9f02",
    "Cationic": "#e35959",
    "Anionic": "#5979e3",
    "CationPi": "#e359d8",
    "PiCation": "#e359d8",
    "PiStacking": "#b559e3",
    "EdgeToFace": "#b559e3",
    "FaceToFace": "#b559e3",
    "MetalAcceptor": "#7da982",
    "MetalDonor": "#7da982",
    "WaterBridge": "#323aa8",
}


def metadata_iterator(
    metadata_tuple: Sequence["InteractionMetadata"], display_all: bool
) -> Iterator["InteractionMetadata"]:
    """Iterate over the metadata tuple, yielding either all or the one with the
    shortest distance.
    """
    if display_all:
        yield from metadata_tuple
    else:
        yield min(
            metadata_tuple,
            key=lambda m: m.get("distance", float("nan")),
        )
