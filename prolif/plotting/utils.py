import builtins
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prolif.ifp import IFP
    from prolif.typeshed import InteractionMetadata

IS_NOTEBOOK = hasattr(builtins, "__IPYTHON__")

separated_interaction_colors = {
    "Hydrophobic": "#59e382",
    "VdWContact": "#dfab43",
    "HBAcceptor": "#59bee3",
    "ImplicitHBAcceptor": "#59bee3",
    "HBDonor": "#239fcd",
    "ImplicitHBDonor": "#239fcd",
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
    "ImplicitHBAcceptor": "#59bee3",
    "ImplicitHBDonor": "#59bee3",
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


BRIDGED_INTERACTIONS: dict[str, str] = {"WaterBridge": "water_residues"}
"""Interactions that are mediated by a third component (e.g. water molecules), mapped
to the metadata key listing the intermediate residues."""


def get_interaction_records(ifp: "IFP", all_metadata: bool) -> list[dict[str, Any]]:
    """Flatten the interactions of a single frame into a list of records.

    Each record describes an edge between two components with the keys ``ligand``
    (source), ``protein`` (target), ``interaction``, ``components``, ``atoms`` (ligand
    atom indices, empty for edges that don't involve the ligand) and ``distance``.
    Bridged interactions (see :data:`BRIDGED_INTERACTIONS`) are split into one record
    per segment of the bridge: ``ligand_water``, ``water_water`` and ``water_protein``,
    all other interactions produce a single ``ligand_protein`` record.
    """
    records: list[dict[str, Any]] = []
    for (lig_resid, prot_resid), int_data in ifp.items():
        for int_name, metadata_tuple in int_data.items():
            is_bridged_interaction = BRIDGED_INTERACTIONS.get(int_name)
            for metadata in metadata_iterator(metadata_tuple, all_metadata):
                if is_bridged_interaction:
                    distances = [d for d in metadata if d.startswith("distance_")]
                    for distlabel in distances:
                        _, src, dest = distlabel.split("_")
                        if src == "ligand":
                            components = "ligand_water"
                            src = str(lig_resid)
                            atoms = metadata["parent_indices"]["ligand"]
                        elif dest == "protein":
                            components = "water_protein"
                            dest = str(prot_resid)
                            atoms = ()
                        else:
                            components = "water_water"
                            atoms = ()
                        records.append(
                            {
                                "ligand": src,
                                "protein": dest,
                                "interaction": int_name,
                                "components": components,
                                "atoms": atoms,
                                "distance": metadata[distlabel],
                            }
                        )
                else:
                    records.append(
                        {
                            "ligand": str(lig_resid),
                            "protein": str(prot_resid),
                            "interaction": int_name,
                            "components": "ligand_protein",
                            "atoms": metadata["parent_indices"]["ligand"],
                            "distance": metadata.get("distance", 0),
                        }
                    )
    return records


def spring_layout_weight(distance: float) -> float:
    """Edge weight for :func:`networkx.spring_layout`: shorter interactions pull the
    nodes closer together. Falls back to a weak attraction when the distance is
    unknown (zero or negative) to avoid dividing by zero."""
    return 10 / distance if distance > 0 else 1.0
