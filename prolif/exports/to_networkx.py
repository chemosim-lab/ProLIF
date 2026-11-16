"""
Export fingerprints as NetworkX graphs --- :mod:`prolif.exports.to_networkx`
=============================================================================

.. versionadded:: 3.0.0

.. autofunction:: to_networkx

.. autofunction:: build_graph

.. autofunction:: calculate_coordinates

"""

from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from operator import itemgetter
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast

import networkx as nx
import numpy as np
from rdkit import Chem

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import metadata_iterator
from prolif.residue import ResidueId

if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP
    from prolif.typeshed import InteractionMetadata

_BRIDGED_INTERACTIONS: dict[str, str] = {"WaterBridge": "water_residues"}


class InteractionRecord(TypedDict):
    """A single interaction record extracted from an IFP frame."""

    source: str
    target: str
    interaction: str
    components: str
    atoms: tuple[int, ...]
    distance: float
    weight: float


_AggKey: TypeAlias = tuple[str, str, str, tuple[int, ...], str]
"""5-tuple key used in the aggregated interaction dicts:
(source, target, interaction_name, atom_indices, components)."""

_GroupKey: TypeAlias = tuple[str, str, str]
"""(source, target, interaction_name): identifies an interaction between two residues
regardless of the ligand atoms involved."""


_record_key: Callable[[InteractionRecord], _AggKey] = itemgetter(
    "source", "target", "interaction", "atoms", "components"
)


def build_graph(
    fp: "Fingerprint",
    ligand_mol: Chem.Mol,
    mol_scale: int = 1,
    kind: Literal["aggregate", "frame"] = "aggregate",
    frame: int = 0,
    display_all: bool = False,
    threshold: float = 0.3,
) -> "nx.MultiGraph[int | str]":
    """Build the interaction graph (nodes + edges only, no layout coordinates).

    Separating graph construction from coordinate calculation allows callers
    such as :class:`~prolif.plotting.network.LigNetwork` to override ligand
    atom coordinates with their own 2-D layout before the protein/water node
    positions are resolved.
    """
    if not hasattr(fp, "ifp"):
        raise RunRequiredError(
            "Please run the fingerprint analysis before attempting to display results."
        )
    if kind == "frame":
        return _make_frame_graph_from_fp(fp, ligand_mol, mol_scale, frame, display_all)
    if kind == "aggregate":
        return _make_agg_graph_from_fp(fp, ligand_mol, mol_scale, threshold)
    raise ValueError(f'{kind!r} must be "aggregate" or "frame"')


def to_networkx(
    fp: "Fingerprint",
    ligand_mol: Chem.Mol,
    kind: Literal["aggregate", "frame"] = "aggregate",
    frame: int = 0,
    display_all: bool = False,
    threshold: float = 0.3,
    mol_scale: int = 1,
) -> "nx.MultiGraph[int | str]":
    """Convert a ProLIF fingerprint to a NetworkX graph with layout coordinates.

    Parameters
    ----------
    fp : Fingerprint
        Fingerprint object that has been run.
    ligand_mol : Chem.Mol
        The ligand molecule (must have a conformer).
    mol_scale : int
        Multiplier applied to atom coordinates to scale the depiction.
    kind : Literal["aggregate", "frame"]
        ``"aggregate"`` merges all frames; ``"frame"`` uses a single frame.
    frame : int
        Frame index; only used when ``kind="frame"``.
    display_all : bool
        Show all atom-level occurrences; only used when ``kind="frame"``.
    threshold : float
        Minimum frequency (0-1) for aggregate graphs. The frequency of an interaction
        between two residues is summed over the different sets of ligand atoms
        involved before being compared to the threshold, and only the most frequent
        set of atoms is kept.

    Returns
    -------
    nx.MultiGraph
        Graph with ``x``/``y`` coordinates on every node.

    .. versionchanged:: 3.0.0
        Removed the ``use_segid`` parameter.
    """
    graph = build_graph(fp, ligand_mol, mol_scale, kind, frame, display_all, threshold)
    return calculate_coordinates(graph, ligand_mol, mol_scale)


def _get_records(ifp: "IFP", all_metadata: bool) -> tuple[list[InteractionRecord], int]:
    """Extract interaction records and the maximum water-bridge order from one IFP frame

    Returns
    -------
    records :
        One entry per individual interaction (one per ligand atom for standard
        interactions; one per distance segment for water bridges).
    max_order :
        Highest water-bridge order seen in this frame. Used by the layout
        algorithm to determine the number of intermediate node layers.
    """
    records: dict[_AggKey, InteractionRecord] = {}
    max_order = 0
    for (lig_resid, prot_resid), int_data in ifp.items():
        if wb := int_data.get("WaterBridge"):
            max_order = max(max_order, wb[0]["order"])

        for int_name, metadata_tuple in int_data.items():
            is_bridged = _BRIDGED_INTERACTIONS.get(int_name)
            for metadata in metadata_iterator(metadata_tuple, all_metadata):
                if is_bridged:
                    new_records = _process_bridged_interaction(
                        metadata, int_name, lig_resid, prot_resid
                    )
                else:
                    new_records = [
                        _process_standard_interaction(
                            metadata, int_name, lig_resid, prot_resid
                        )
                    ]
                # the same edge can appear several times in a frame, e.g. a water
                # segment shared by several water bridges: keep the shortest one
                for record in new_records:
                    key = _record_key(record)
                    if (
                        key not in records
                        or record["distance"] < records[key]["distance"]
                    ):
                        records[key] = record
    return list(records.values()), max_order


def _process_bridged_interaction(
    metadata: "InteractionMetadata",
    int_name: str,
    lig_resid: ResidueId,
    prot_resid: ResidueId,
) -> list[InteractionRecord]:
    """Decompose one water-bridge metadata entry into per-segment records.

    A water-bridge spans ligand → water(s) → protein.  Each ``distance_*``
    key in *metadata* represents one segment of that chain, so this function
    emits one record per segment.
    """
    records: list[InteractionRecord] = []
    for distlabel in (d for d in metadata if d.startswith("distance_")):
        _, src, dest = distlabel.split("_")
        if src == "ligand":
            records.append(
                InteractionRecord(
                    source=str(lig_resid),
                    target=dest,
                    interaction=int_name,
                    components="ligand_water",
                    atoms=metadata["parent_indices"]["ligand"],
                    distance=metadata[distlabel],
                    weight=1.0,
                )
            )
        elif dest == "protein":
            records.append(
                InteractionRecord(
                    source=src,
                    target=str(prot_resid),
                    interaction=int_name,
                    components="water_protein",
                    atoms=(),
                    distance=metadata[distlabel],
                    weight=1.0,
                )
            )
        else:
            records.append(
                InteractionRecord(
                    source=src,
                    target=dest,
                    interaction=int_name,
                    components="water_water",
                    atoms=(),
                    distance=metadata[distlabel],
                    weight=1.0,
                )
            )
    return records


def _process_standard_interaction(
    metadata: "InteractionMetadata",
    int_name: str,
    lig_resid: ResidueId,
    prot_resid: ResidueId,
) -> InteractionRecord:
    """Build one record for a direct ligand-protein interaction."""
    return InteractionRecord(
        source=str(lig_resid),
        target=str(prot_resid),
        interaction=int_name,
        components="ligand_protein",
        atoms=metadata["parent_indices"]["ligand"],
        distance=metadata.get("distance", 0),
        weight=1.0,
    )


def _make_frame_graph_from_fp(
    fp: "Fingerprint",
    ligand_mol: Chem.Mol,
    mol_scale: int = 1,
    frame: int = 0,
    display_all: bool = False,
) -> "nx.MultiGraph[int | str]":
    """Build an interaction graph for a single trajectory frame."""
    if frame not in fp.ifp:
        raise ValueError(f"Frame {frame} not found in fingerprint data")
    graph: "nx.MultiGraph[int | str]" = nx.MultiGraph()
    records, max_order = _get_records(fp.ifp[frame], all_metadata=display_all)
    if not records:
        return graph
    _add_ligand_nodes_and_bonds(graph, ligand_mol, mol_scale=mol_scale)
    _add_records_to_graph(graph, records)
    graph.graph["max_bridge_order"] = max_order
    return graph


def _add_ligand_nodes_and_bonds(
    graph: "nx.MultiGraph[int | str]",
    ligand_mol: Chem.Mol,
    mol_scale: int = 1,
) -> None:
    """Add ligand atom nodes (with 3-D coordinates) and bond edges to *graph*."""
    conformer = ligand_mol.GetConformer()
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx) * mol_scale
        graph.add_node(
            idx,
            node_type="ligand",
            symbol=atom.GetSymbol(),
            charge=atom.GetFormalCharge(),
            x=float(pos.x),
            y=float(pos.y),
            z=float(pos.z),
            ligand=str(ResidueId.from_atom(atom)),
        )
    for bond in ligand_mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            edge_type="bond",
            bond_type=bond.GetBondType(),
            bond_order=bond.GetBondTypeAsDouble(),
        )


def _spring_layout_weight(distance: float) -> float:
    """Edge weight for :func:`networkx.spring_layout`: shorter interactions pull the
    nodes closer together. Falls back to a weak attraction when the distance is
    unknown (zero) to avoid dividing by zero."""
    return 10.0 / distance if distance > 0 else 1.0


_COMPONENT_NODE_ATTRS: dict[str, tuple[str, str]] = {
    "ligand_protein": ("protein", "residue"),
    "ligand_water": ("water", "water_id"),
    "water_protein": ("protein", "residue"),
    "water_water": ("water", "water_id"),
}


def _add_records_to_graph(
    graph: "nx.MultiGraph[int | str]", records: list[InteractionRecord]
) -> None:
    """Add protein/water nodes and interaction edges to *graph* from *records*.

    Ligand nodes must already be present (added by :func:`_add_ligand_nodes_and_bonds`).
    The ``weight_spring_layout`` edge attribute is precomputed here so that
    :func:`calculate_coordinates` can use it directly without a second pass.
    """
    for record in records:
        components = record["components"]
        source = record["source"]
        target = record["target"]
        weight = record["weight"]
        distance = record["distance"]
        spring_weight = _spring_layout_weight(distance)

        if components in {"ligand_protein", "ligand_water"}:
            node_type, id_attr = _COMPONENT_NODE_ATTRS[components]
            if not graph.has_node(target):
                graph.add_node(target, node_type=node_type, **{id_attr: target})
            for atom_idx in record["atoms"]:
                graph.add_edge(
                    atom_idx,
                    target,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=weight,
                    distance=distance,
                    components=components,
                    atoms=record["atoms"],
                    weight_spring_layout=spring_weight,
                )

        elif components in {"water_protein", "water_water"}:
            node_type, id_attr = _COMPONENT_NODE_ATTRS[components]
            if not graph.has_node(source):
                graph.add_node(source, node_type="water", water_id=source)
            if not graph.has_node(target):
                graph.add_node(target, node_type=node_type, **{id_attr: target})
            graph.add_edge(
                source,
                target,
                edge_type="interaction",
                interaction=record["interaction"],
                weight=weight,
                distance=distance,
                components=components,
                atoms=record["atoms"],
                weight_spring_layout=spring_weight,
            )


class _AggValue(TypedDict):
    weight: float
    distance: float


def _filtered_data_to_records(
    filtered_data: dict[_AggKey, _AggValue],
) -> list[InteractionRecord]:
    """Convert the output of :func:`_filter_by_threshold` into
    :class:`InteractionRecord` list."""
    return [
        InteractionRecord(
            source=key[0],
            target=key[1],
            interaction=key[2],
            atoms=key[3],
            components=key[4],
            weight=data_item["weight"],
            distance=data_item["distance"],
        )
        for key, data_item in filtered_data.items()
    ]


def _make_agg_graph_from_fp(
    fp: "Fingerprint",
    ligand_mol: Chem.Mol,
    mol_scale: int = 1,
    threshold: float = 0.3,
) -> "nx.MultiGraph[int | str]":
    """Build an interaction graph aggregated over all frames."""
    graph: "nx.MultiGraph[int | str]" = nx.MultiGraph()
    all_records: list[InteractionRecord] = []
    max_order = 0
    n_frames = len(fp.ifp)
    for ifp in fp.ifp.values():
        frame_records, frame_max_order = _get_records(ifp, all_metadata=False)
        all_records.extend(frame_records)
        max_order = max(max_order, frame_max_order)
    if not all_records:
        return graph
    _add_ligand_nodes_and_bonds(graph, ligand_mol, mol_scale=mol_scale)
    grouped = _aggregate_interaction_data(all_records, n_frames)
    filtered = _filter_by_threshold(grouped, threshold)
    _add_records_to_graph(graph, _filtered_data_to_records(filtered))
    graph.graph["max_bridge_order"] = max_order
    return graph


def _aggregate_interaction_data(
    records: list[InteractionRecord], n_frames: int
) -> dict[_AggKey, _AggValue]:
    """Aggregate interaction records across frames, normalising counts to frequencies.

    The returned dict maps a 5-tuple key
    ``(source, target, interaction, atoms, components)`` to
    ``{"weight": float, "distance": float}`` where *weight* is the fraction of
    frames in which the interaction was observed (0-1).
    """
    raw: defaultdict[_AggKey, dict] = defaultdict(lambda: {"count": 0, "distances": []})
    for record in records:
        key = _record_key(record)
        raw[key]["count"] += 1
        raw[key]["distances"].append(record["distance"])

    return {
        key: _AggValue(
            weight=v["count"] / n_frames,
            distance=sum(v["distances"]) / len(v["distances"]),
        )
        for key, v in raw.items()
    }


def _filter_by_threshold(
    grouped_data: dict[_AggKey, _AggValue], threshold: float
) -> dict[_AggKey, _AggValue]:
    """Keep only interactions whose frequency meets *threshold*.

    The frequency of an interaction between two residues is summed over the
    different sets of ligand atoms involved before being compared to the threshold,
    and only the most frequent set of atoms is kept for each interaction. Water
    molecules that no longer bridge the ligand to a protein residue after this
    filtering are removed as well, along with their edges.
    """
    totals: defaultdict[_GroupKey, float] = defaultdict(float)
    for key, data_item in grouped_data.items():
        totals[key[:3]] += data_item["weight"]

    # most frequent set of atoms for each interaction above the threshold
    best: dict[_GroupKey, _AggKey] = {}
    for key in sorted(grouped_data, key=lambda k: (-grouped_data[k]["weight"], k)):
        if totals[key[:3]] >= threshold:
            best.setdefault(key[:3], key)
    kept = set(best.values())
    filtered_data = {key: v for key, v in grouped_data.items() if key in kept}
    return _prune_dangling_waters(filtered_data)


def _prune_dangling_waters(
    filtered_data: dict[_AggKey, _AggValue],
) -> dict[_AggKey, _AggValue]:
    """Remove the water molecules (and their edges) that are not on a path between
    the ligand and a protein residue, which can happen after thresholding."""
    graph: "nx.Graph[str]" = nx.Graph()
    node_types: dict[str, str] = {}
    for source, target, _, _, components in filtered_data:
        node_types[source], node_types[target] = components.split("_")
        graph.add_edge(source, target)

    # a water molecule with a single neighbor cannot bridge anything
    while leaves := [
        node
        for node, degree in graph.degree
        if node_types[node] == "water" and degree <= 1
    ]:
        graph.remove_nodes_from(leaves)

    valid: set[str] = set()
    for component in nx.connected_components(graph):
        if {"ligand", "protein"} <= {node_types[node] for node in component}:
            valid.update(component)
    return {
        key: v
        for key, v in filtered_data.items()
        if key[0] in valid and key[1] in valid
    }


def calculate_coordinates(
    graph: "nx.MultiGraph[int | str]", ligand_mol: Chem.Mol, mol_scale: int = 1
) -> "nx.MultiGraph[int | str]":
    """
    Calculate coordinates for graph nodes.

    1. Keep ligand nodes fixed at their existing positions
    2. Calculate initial positions for water and protein nodes based on ligand
       interactions
    3. Apply spring layout to optimize positions while keeping ligand nodes fixed
    4. Resolve overlaps between nodes

    Parameters
    ----------
    graph : nx.MultiGraph
        NetworkX graph with nodes and edges
    ligand_mol : Chem.Mol
        The ligand molecule for coordinate reference
    mol_scale : int
        Scaling factor for coordinates

    Returns
    -------
    nx.MultiGraph
        Graph with updated node coordinates
    """
    # Get conformer for ligand coordinates
    conformer = ligand_mol.GetConformer()

    # Separate nodes by type
    ligand_nodes: list[int] = []
    water_nodes: list[str] = []
    protein_nodes: list[str] = []
    for n, d in graph.nodes(data=True):
        if d.get("node_type") == "ligand":
            ligand_nodes.append(cast(int, n))
        elif d.get("node_type") == "water":
            water_nodes.append(cast(str, n))
        elif d.get("node_type") == "protein":
            protein_nodes.append(cast(str, n))
        else:
            continue
    if not ligand_nodes:
        return graph

    # Calculate ligand center and dimensions
    ligand_coords = np.array(
        [
            [graph.nodes[n]["x"], graph.nodes[n]["y"]]
            for n in ligand_nodes
            if "x" in graph.nodes[n] and "y" in graph.nodes[n]
        ]
    )

    if len(ligand_coords) == 0:
        # Fallback: use conformer coordinates if not already set
        ligand_coords = np.array(
            [
                [
                    conformer.GetAtomPosition(n).x * mol_scale,
                    conformer.GetAtomPosition(n).y * mol_scale,
                ]
                for n in ligand_nodes
            ]
        )
        # Update the graph with these coordinates
        for i, node in enumerate(ligand_nodes):
            graph.nodes[node]["x"] = float(ligand_coords[i][0])
            graph.nodes[node]["y"] = float(ligand_coords[i][1])

    center = np.mean(ligand_coords, axis=0)
    ligand_bounds = np.max(ligand_coords, axis=0) - np.min(ligand_coords, axis=0)
    width, height = ligand_bounds

    # Calculate initial positions for water and protein nodes
    pos = _calculate_initial_positions(
        graph, ligand_nodes, water_nodes, protein_nodes, center, max(width, height)
    )

    # Apply spring layout with fixed ligand positions
    if len(water_nodes) + len(protein_nodes) > 0 and graph.number_of_edges() > 0:
        with suppress(Exception):
            pos = nx.spring_layout(
                graph,
                pos=pos,
                fixed=ligand_nodes,
                scale=2,
                iterations=100,
                center=center,
                weight="weight_spring_layout",
            )

    # Resolve overlaps
    pos = _resolve_overlaps(pos, water_nodes, protein_nodes, ligand_coords, mol_scale)

    # Update node positions in graph
    for n, (x, y) in pos.items():
        graph.nodes[n]["x"] = float(x)
        graph.nodes[n]["y"] = float(y)

    return graph


def _calculate_initial_positions(
    graph: "nx.MultiGraph[int | str]",
    ligand_nodes: list[int],
    water_nodes: list[str],
    protein_nodes: list[str],
    center: np.ndarray,
    size: float,
) -> dict[int | str, np.ndarray]:
    """
    Calculate initial positions for water and protein nodes based on ligand interactions

    approach:
    1. Find maximum order (number of water molecules in interaction chain)
    2. Create layers containing nodes at each level
    3. Calculate initial positions using direction vectors from ligand center
    """
    pos: dict[int | str, np.ndarray] = {
        node: np.array([graph.nodes[node]["x"], graph.nodes[node]["y"]])
        for node in ligand_nodes
    }

    max_order: int = graph.graph.get("max_bridge_order", 1)
    distance_scaler = {1: 0.3, 2: 0.25}
    layers = _create_layers(graph, ligand_nodes, water_nodes, protein_nodes, max_order)

    # Position nodes layer by layer
    for layer_num, layer_nodes in layers.items():
        if layer_num == 0:  # Ligand layer - already positioned
            continue

        for i, node in enumerate(cast("list[int | str]", layer_nodes)):
            # Get connected nodes from previous layer
            connected_atoms: list[int] = []
            for neighbor in graph.neighbors(node):
                if neighbor in pos:
                    connected_atoms.append(cast(int, neighbor))

            if connected_atoms:
                # Calculate position based on average of connected atoms
                connected_coords = np.array([pos[atom] for atom in connected_atoms])
                avg_pos = np.mean(connected_coords, axis=0)

                # Create direction vector from center towards average interaction point
                direction = avg_pos - center
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 0:
                    direction_unit = direction / direction_norm
                    # Position at distance based on layer number
                    distance = size * distance_scaler.get(layer_num, 0.2)
                    pos[node] = avg_pos + direction_unit * distance
                else:
                    # Spread around the center if direction is zero
                    angle = 2 * np.pi * i / len(layer_nodes)
                    distance = size * distance_scaler.get(layer_num, 0.2) * 1.2
                    pos[node] = center + distance * np.array(
                        [np.cos(angle), np.sin(angle)]
                    )
            else:
                # For unconnected nodes, place around periphery
                angle = 2 * np.pi * i / len(layer_nodes)
                distance = size * distance_scaler.get(layer_num, 0.2) * 2
                pos[node] = center + distance * np.array([np.cos(angle), np.sin(angle)])

    return pos


def _create_layers(
    graph: "nx.MultiGraph[int | str]",
    ligand_nodes: list[int],
    water_nodes: list[str],
    protein_nodes: list[str],
    max_order: int,
) -> dict[int, list[int] | list[str]]:
    """
    Create layers containing nodes at each level.

    Layer 0: Ligand atoms (fixed positions)
    Layer 1+: Water and protein nodes based on their distance from ligand
    """
    layers: dict[int, list[int] | list[str]] = {0: ligand_nodes}

    # Initialize remaining nodes
    remaining_nodes: set[str] = set(water_nodes + protein_nodes)
    visited_nodes: set[int | str] = set(ligand_nodes)

    for layer_num in range(1, max_order + 2):  # Add extra layers for safety
        current_layer: list[str] = []
        nodes_to_remove: set[str] = set()

        for node in sorted(remaining_nodes):
            # Check if this node is connected to any node in previous layers
            connected_to_previous = next(
                (
                    True
                    for neighbor in graph.neighbors(node)
                    if neighbor in visited_nodes
                ),
                False,
            )

            if connected_to_previous:
                current_layer.append(node)
                nodes_to_remove.add(node)

        if current_layer:
            layers[layer_num] = current_layer
            remaining_nodes -= nodes_to_remove
            visited_nodes.update(current_layer)

        if not remaining_nodes:
            break

    # Add any remaining unconnected nodes to the last layer
    if remaining_nodes:
        last_layer = max(layers) + 1 if layers else 1
        layers[last_layer] = sorted(remaining_nodes)

    return layers


def _resolve_overlaps(
    pos: dict[int | str, np.ndarray],
    water_nodes: list[str],
    protein_nodes: list[str],
    ligand_coords: np.ndarray,
    mol_scale: int,
) -> dict[int | str, np.ndarray]:
    """
    Resolve overlaps between nodes.

    This function detects and resolves overlaps between:
    1. Non-ligand nodes with each other
    2. Non-ligand nodes with ligand atoms
    """
    min_distance_lig = 2.6 * mol_scale
    min_distance_non_lig = 2.9 * mol_scale
    max_iterations = 100

    halfway_iteration = max_iterations // 2
    non_ligand_nodes = water_nodes + protein_nodes

    for iteration in range(max_iterations):
        overlap_found = False
        adjustments = {node: np.zeros(2) for node in non_ligand_nodes}

        # Phase 1: Resolve non-ligand to non-ligand overlaps
        for node1 in non_ligand_nodes:
            if node1 not in pos:
                continue

            for node2 in non_ligand_nodes:
                if node2 not in pos:
                    continue
                if node1 == node2:
                    continue

                delta = pos[node2] - pos[node1]
                if np.allclose(delta, 0):
                    delta = np.array([1, 1])

                dist = np.linalg.norm(delta)

                if 0 < dist < min_distance_non_lig:
                    overlap_found = True
                    force = (min_distance_non_lig - dist) * 0.5
                    if iteration >= halfway_iteration:
                        force *= 1.5
                    direction = delta / dist

                    adjustments[node1] -= direction * force
                    adjustments[node2] += direction * force

        # Apply adjustments
        for node, adjustment in adjustments.items():
            if node in pos:
                pos[node] += adjustment

        # Phase 2: Resolve non-ligand to ligand overlaps
        ligand_adjustments = {node: np.zeros(2) for node in non_ligand_nodes}

        for node in non_ligand_nodes:
            if node not in pos:
                continue

            for ligand_coord in ligand_coords:
                delta = pos[node] - ligand_coord
                dist = np.linalg.norm(delta)

                if 0 < dist < min_distance_lig:
                    overlap_found = True
                    force = (min_distance_lig - dist) * 1.2
                    if iteration >= halfway_iteration:
                        force *= 4 / 3
                    direction = delta / dist
                    ligand_adjustments[node] += direction * force
                    # only apply one adjustment per node to avoid averaging forces when
                    # the node is initially placed in the center of a ring
                    break

        # Apply ligand overlap adjustments
        for node, adjustment in ligand_adjustments.items():
            if node in pos:
                pos[node] += adjustment

        if not overlap_found:
            break

    return pos
