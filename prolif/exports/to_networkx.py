from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from rdkit import Chem

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import metadata_iterator

if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP

# Constants
_BRIDGED_INTERACTIONS: dict[str, str] = {"WaterBridge": "water_residues"}

# Global variable to store maximum order from water bridge interactions
_MAX_ORDER: int = 1


def fp_to_networkx(
    fp: "Fingerprint",
    molsize: int,
    ligand_mol: Chem.Mol,
    kind: Literal["aggregate", "frame"] = "aggregate",
    frame: int = 0,
    display_all: bool = False,
    threshold: float = 0.3,
) -> nx.Graph:
    """
    Convert a ProLIF fingerprint to a NetworkX graph.

    Parameters
    ----------
    fp : Fingerprint
        The fingerprint object containing interaction data
    ligand_mol : Chem.Mol
        The ligand molecule
    kind : Literal["aggregate", "frame"]
        Type of graph to create
    frame : int
        Frame number for single-frame graphs
    display_all : bool
        Whether to display all metadata for frame graphs
    threshold : float
        Minimum occurrence threshold for aggregate graphs

    Returns
    -------
    nx.MultiGraph
        NetworkX graph representation of the interactions
    """
    if not hasattr(fp, "ifp"):
        raise RunRequiredError(
            "Please run the fingerprint analysis before attempting to display results."
        )

    if kind == "frame":
        return _make_frame_graph_from_fp(fp, molsize, ligand_mol, frame, display_all)
    if kind == "aggregate":
        return _make_agg_graph_from_fp(fp, molsize, ligand_mol, threshold)
    raise ValueError(f'{kind!r} must be "aggregate" or "frame"')


def _get_records(ifp: "IFP", all_metadata: bool) -> list[dict[str, Any]]:
    """Extract interaction records from fingerprint data."""
    global _MAX_ORDER
    records: list[dict[str, Any]] = []
    max_order = 0
    for (lig_resid, prot_resid), int_data in ifp.items():
        # get max order from water bridge interactions
        if int_data["WaterBridge"]:
            curr_order = int_data["WaterBridge"][0]["order"]
            max_order = max(max_order, curr_order)

        for int_name, metadata_tuple in int_data.items():
            is_bridged_interaction: str | None = _BRIDGED_INTERACTIONS.get(int_name)

            for metadata in metadata_iterator(metadata_tuple, all_metadata):
                if is_bridged_interaction:
                    records.extend(
                        _process_bridged_interaction(
                            metadata, int_name, lig_resid, prot_resid
                        )
                    )
                else:
                    records.append(
                        _process_standard_interaction(
                            metadata, int_name, lig_resid, prot_resid
                        )
                    )

    # Update global max_order
    _MAX_ORDER = max_order
    return records


def _process_bridged_interaction(
    metadata: dict, int_name: str, lig_resid: Any, prot_resid: Any
) -> list[dict[str, Any]]:
    """Process bridged interactions (e.g., water bridges)."""
    records: list[dict[str, Any]] = []
    distances: list[str] = [d for d in metadata if d.startswith("distance_")]

    for distlabel in distances:
        _, src, dest = distlabel.split("_")

        if src == "ligand":
            components: str = "ligand_water"
            source_id: str = str(lig_resid)
            atoms: tuple = metadata["parent_indices"]["ligand"]
            records.append(
                {
                    "ligand": source_id,
                    "water": dest,
                    "interaction": int_name,
                    "components": components,
                    "atoms": atoms,
                    "distance": metadata[distlabel],
                }
            )
        elif dest == "protein":
            components = "water_protein"
            dest_id: str = str(prot_resid)
            atoms = ()
            records.append(
                {
                    "water": src,
                    "protein": dest_id,
                    "interaction": int_name,
                    "components": components,
                    "atoms": atoms,
                    "distance": metadata[distlabel],
                }
            )
        else:
            components = "water_water"
            atoms = ()
            records.append(
                {
                    "water1": src,
                    "water2": dest,
                    "interaction": int_name,
                    "components": components,
                    "atoms": atoms,
                    "distance": metadata[distlabel],
                }
            )

    return records


def _process_standard_interaction(
    metadata: dict, int_name: str, lig_resid: Any, prot_resid: Any
) -> dict[str, Any]:
    """Process standard ligand-protein interactions."""
    return {
        "ligand": str(lig_resid),
        "protein": str(prot_resid),
        "interaction": int_name,
        "components": "ligand_protein",
        "atoms": metadata["parent_indices"]["ligand"],
        "distance": metadata.get("distance", 0),
    }


def _make_frame_graph_from_fp(
    fp: "Fingerprint",
    molsize: int,
    ligand_mol: Chem.Mol,
    frame: int = 0,
    display_all: bool = False,
) -> nx.MultiGraph:
    """Create a networkx graph from a single frame fingerprint."""
    # Validate frame exists
    if frame not in fp.ifp:
        raise ValueError(f"Frame {frame} not found in fingerprint data")

    graph: nx.MultiGraph = nx.MultiGraph()
    ifp: "IFP" = fp.ifp[frame]
    records: list[dict[str, Any]] = _get_records(ifp, all_metadata=display_all)

    if not records:
        return graph

    # Add ligand structure
    ligand_id: str = records[0]["ligand"]
    _add_ligand_nodes_and_bonds(graph, ligand_mol, ligand_id, molsize)

    # Add interactions
    for record in records:
        components: str = record["components"]
        lig_indices: tuple = record["atoms"]

        if components == "ligand_protein":
            # Standard ligand-protein interaction
            protein_node = record["protein"]

            # Add protein node if it doesn't exist
            if not graph.has_node(protein_node) and protein_node:
                graph.add_node(
                    protein_node, node_type="protein", residue=record["protein"]
                )

            # Add edges for each ligand atom involved in the interaction
            for idx in lig_indices:
                graph.add_edge(
                    idx,
                    protein_node,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=record.get("weight", 1),
                    distance=record["distance"],
                    components=record["components"],
                    weight_spring_layout=10 / record["distance"],
                )

        elif components == "ligand_water":
            # Ligand-water interaction
            water_node: str = record["water"]

            # Add water node if it doesn't exist
            if not graph.has_node(water_node) and water_node:
                graph.add_node(water_node, node_type="water", water_id=water_node)

            # Add edges for each ligand atom involved in the interaction
            for idx in lig_indices:
                graph.add_edge(
                    idx,
                    water_node,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=1,
                    distance=record["distance"],
                    components=record["components"],
                    weight_spring_layout=10 / record["distance"],
                )

        elif components == "water_protein":
            # Water-protein interaction
            water_node = record["water"]
            protein_node = record["protein"]

            # Add water node if it doesn't exist
            if not graph.has_node(water_node) and water_node:
                graph.add_node(water_node, node_type="water", water_id=water_node)

            # Add protein node if it doesn't exist
            if not graph.has_node(protein_node) and protein_node:
                graph.add_node(protein_node, node_type="protein", residue=protein_node)

            # Add edge between water and protein
            graph.add_edge(
                water_node,
                protein_node,
                edge_type="interaction",
                interaction=record["interaction"],
                weight=1,
                distance=record["distance"],
                components=record["components"],
                weight_spring_layout=10 / record["distance"],
            )

        elif components == "water_water":
            # Water-water interaction
            water1_node = record["water1"]
            water2_node = record["water2"]

            # Add water nodes if they don't exist
            if not graph.has_node(water1_node) and water1_node:
                graph.add_node(water1_node, node_type="water", water_id=water1_node)

            if not graph.has_node(water2_node) and water2_node:
                graph.add_node(water2_node, node_type="water", water_id=water2_node)

            # Add edge between water molecules
            if water1_node and water2_node:
                graph.add_edge(
                    water1_node,
                    water2_node,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=1,
                    distance=record["distance"],
                    components=record["components"],
                    weight_spring_layout=10 / record["distance"],
                )

    # Calculate coordinates for all nodes
    graph = calculate_coordinates(graph, ligand_mol, molsize)

    return graph


def _add_ligand_nodes_and_bonds(
    graph: nx.Graph, ligand_mol: Chem.Mol, ligand_id: str, molsize: int
) -> None:
    """Add ligand atoms and bonds to the graph."""
    # Get ligand coordinates from the conformer
    conformer = ligand_mol.GetConformer()

    # Add ligand atom nodes
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx) * molsize
        graph.add_node(
            idx,
            node_type="ligand",
            symbol=atom.GetSymbol(),
            charge=atom.GetFormalCharge(),
            label=atom.GetSymbol(),
            ligand=ligand_id,
            x=float(pos.x),
            y=float(pos.y),
            z=float(pos.z),
        )

    # Add ligand bonds
    for bond in ligand_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        graph.add_edge(
            begin_idx,
            end_idx,
            edge_type="bond",
            bond_type=bond.GetBondType(),
            bond_order=bond.GetBondTypeAsDouble(),
        )


def _make_agg_graph_from_fp(
    fp: "Fingerprint",
    molsize: int,
    ligand_mol: Chem.Mol,
    threshold: float = 0.3,
) -> nx.MultiGraph:
    """Create a networkx graph from an aggregate fingerprint."""
    graph: nx.MultiGraph = nx.MultiGraph()

    # Collect all records across frames
    all_records: list[dict[str, Any]] = []
    for ifp in fp.ifp.values():
        all_records.extend(_get_records(ifp, all_metadata=False))

    if not all_records:
        return graph

    # Add ligand structure
    ligand_id: str = all_records[0]["ligand"]
    _add_ligand_nodes_and_bonds(graph, ligand_mol, ligand_id, molsize)

    # Aggregate and filter data
    grouped_data: dict = _aggregate_interaction_data(all_records)
    filtered_data: dict = _filter_by_threshold(grouped_data, threshold)

    # Convert filtered data to records format for edge creation
    filtered_records: list[dict[str, Any]] = []
    for interaction_key, data_item in filtered_data.items():
        components: str = data_item["components"]

        if components == "ligand_protein":
            ligand, protein, interaction = interaction_key
            filtered_records.append(
                {
                    "ligand": ligand,
                    "protein": protein,
                    "interaction": interaction,
                    "atoms": data_item["atoms"],
                    "weight": data_item["weight"],
                    "distance": data_item["distance"],
                    "components": data_item["components"],
                }
            )
        elif components == "ligand_water":
            # Reconstruct ligand-water record
            filtered_records.append(
                {
                    "ligand": interaction_key[0],
                    "water": interaction_key[1],
                    "interaction": interaction_key[2],
                    "atoms": data_item["atoms"],
                    "weight": data_item["weight"],
                    "distance": data_item["distance"],
                    "components": data_item["components"],
                }
            )
        elif components == "water_protein":
            # Reconstruct water-protein record
            filtered_records.append(
                {
                    "water": interaction_key[0],
                    "protein": interaction_key[1],
                    "interaction": interaction_key[2],
                    "atoms": data_item["atoms"],
                    "weight": data_item["weight"],
                    "distance": data_item["distance"],
                    "components": data_item["components"],
                }
            )
        elif components == "water_water":
            # Reconstruct water-water record
            filtered_records.append(
                {
                    "water1": interaction_key[0],
                    "water2": interaction_key[1],
                    "interaction": interaction_key[2],
                    "atoms": data_item["atoms"],
                    "weight": data_item["weight"],
                    "distance": data_item["distance"],
                    "components": data_item["components"],
                }
            )
    # Add interactions
    for record in filtered_records:
        components = record["components"]

        if components == "ligand_protein":
            # Standard ligand-protein interaction
            protein_node = record["protein"]
            lig_indices = record["atoms"]

            # Add protein node if it doesn't exist
            if not graph.has_node(protein_node) and protein_node:
                graph.add_node(
                    protein_node, node_type="protein", residue=record["protein"]
                )

            # Add edges for each ligand atom involved in the interaction
            for idx in lig_indices:
                graph.add_edge(
                    idx,
                    protein_node,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=record.get("weight"),
                    distance=record["distance"],
                    components=record["components"],
                    weight_spring_layout=10 / record["distance"],
                )

        elif components == "ligand_water":
            # Ligand-water interaction
            water_node = record.get("water", "")
            lig_indices = record["atoms"]

            # Add water node if it doesn't exist
            if not graph.has_node(water_node) and water_node:
                graph.add_node(water_node, node_type="water", water_id=water_node)

            # Add edges for each ligand atom involved in the interaction
            for idx in lig_indices:
                graph.add_edge(
                    idx,
                    water_node,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=record.get("weight"),
                    distance=record["distance"],
                    components=record["components"],
                    weight_spring_layout=10 / record["distance"],
                )

        elif components == "water_protein":
            # Water-protein interaction
            water_node = record.get("water", "")
            protein_node = record["protein"]

            # Add water node if it doesn't exist
            if not graph.has_node(water_node) and water_node:
                graph.add_node(water_node, node_type="water", water_id=water_node)

            # Add protein node if it doesn't exist
            if not graph.has_node(protein_node) and protein_node:
                graph.add_node(protein_node, node_type="protein", residue=protein_node)

            # Add edge between water and protein
            graph.add_edge(
                water_node,
                protein_node,
                edge_type="interaction",
                interaction=record["interaction"],
                weight=record.get("weight"),
                distance=record["distance"],
                components=record["components"],
                weight_spring_layout=10 / record["distance"],
            )

        elif components == "water_water":
            # Water-water interaction
            water1_node: str | None = record.get("water1")
            water2_node: str | None = record.get("water2")

            # Add water nodes if they don't exist
            if not graph.has_node(water1_node) and water1_node:
                graph.add_node(water1_node, node_type="water", water_id=water1_node)

            if not graph.has_node(water2_node) and water2_node:
                graph.add_node(water2_node, node_type="water", water_id=water2_node)

            # Add edge between water molecules
            if water1_node and water2_node:
                graph.add_edge(
                    water1_node,
                    water2_node,
                    edge_type="interaction",
                    interaction=record["interaction"],
                    weight=record.get("weight"),
                    distance=record["distance"],
                    components=record["components"],
                    weight_spring_layout=10 / record["distance"],
                )

    # Calculate coordinates for all nodes
    graph = calculate_coordinates(graph, ligand_mol, molsize)

    return graph


def _aggregate_interaction_data(records: list[dict[str, Any]]) -> dict:
    """Aggregate interaction data across frames."""
    grouped_data: dict = {}

    for record in records:
        components: str = record["components"]

        if components == "ligand_protein":
            # Standard ligand-protein interaction
            key: tuple = (
                record["ligand"],
                record["protein"],
                record["interaction"],
                tuple(record["atoms"]),
            )
        elif components == "ligand_water":
            # Ligand-water interaction
            key = (
                record["ligand"],
                record.get("water", ""),
                record["interaction"],
                tuple(record["atoms"]),
                "ligand_water",
            )
        elif components == "water_protein":
            # Water-protein interaction
            key = (
                record.get("water", ""),
                record["protein"],
                record["interaction"],
                tuple(record["atoms"]),
                "water_protein",
            )
        elif components == "water_water":
            # Water-water interaction
            key = (
                record.get("water1", ""),
                record.get("water2", ""),
                record["interaction"],
                tuple(record["atoms"]),
                "water_water",
            )
        else:
            continue  # skip unknown component types

        if key not in grouped_data:
            grouped_data[key] = {
                "weight": 0,
                "distances": [],
                "components": record["components"],
            }

        grouped_data[key]["weight"] += 1
        grouped_data[key]["distances"].append(record["distance"])

    return grouped_data


def _filter_by_threshold(grouped_data: dict, threshold: float) -> dict:
    """Filter interactions by occurrence threshold."""
    # Calculate interaction totals for different component types
    interaction_totals: dict = {}
    processed_data: dict = {}

    for key, data_item in grouped_data.items():
        components: str = data_item["components"]
        avg_distance: float = sum(data_item["distances"]) / len(data_item["distances"])
        weight: int = data_item["weight"]

        if components == "ligand_protein":
            ligand, protein, interaction, atoms = key
            interaction_key: tuple = (ligand, protein, interaction, "ligand_protein")

            interaction_totals[interaction_key] = (
                interaction_totals.get(interaction_key, 0) + weight
            )

            processed_data[key] = {
                "weight": weight,
                "distance": avg_distance,
                "components": data_item["components"],
                "atoms": atoms,
            }

        elif components in ["ligand_water", "water_protein", "water_water"]:
            node1, node2, interaction, atoms, comp_type = key

            interaction_key = (node1, node2, interaction, comp_type)

            interaction_totals[interaction_key] = (
                interaction_totals.get(interaction_key, 0) + weight
            )

            processed_data[key] = {
                "weight": weight,
                "distance": avg_distance,
                "components": data_item["components"],
                "atoms": atoms,
            }

    # Filter by threshold and keep most occurring atom per interaction
    filtered_data: dict = {}

    # First, add direct ligand-protein interactions that meet the threshold
    for key, data_item in processed_data.items():
        if data_item["components"] == "ligand_protein":
            ligand, protein, interaction, atoms = key
            interaction_key = (ligand, protein, interaction)
            if interaction_totals[interaction_key] >= threshold:
                if (
                    interaction_key not in filtered_data
                    or data_item["weight"] > filtered_data[interaction_key]["weight"]
                ):
                    filtered_data[interaction_key] = data_item

    # Now, process water interactions
    for key, data_item in processed_data.items():
        node1, node2, interaction, atoms, comp_type = key
        interaction_key = (node1, node2, interaction, comp_type)

        if interaction_totals[interaction_key] >= threshold:
            if check_till_protein(node2, interaction_totals, threshold, comp_type):
                if (
                    interaction_key not in filtered_data
                    or data_item["weight"] > filtered_data[interaction_key]["weight"]
                ):
                    filtered_data[interaction_key] = data_item
        else:
            # If interaction doesn't meet threshold, remove all the subsequent interactions till protein from processed_data to avoid having disconnected content
            clean_processed_data(key, processed_data)
    return filtered_data


def clean_processed_data(
    key: tuple, processed_data: dict, visited_keys: set | None = None
) -> None:
    """
    Remove all keys from processed_data that start with the given key.

    This is used to clean up processed_data when an interaction does not meet the threshold.
    Recursively removes all subsequent interactions in the water bridge chain.

    Parameters:
    -----------
    key : tuple
        The key to process and remove
    processed_data : dict
        Dictionary of processed interaction data
    visited_keys : set, optional
        Set of keys that have already been visited to prevent infinite recursion
    """
    # Initialize visited_keys set if not provided
    if visited_keys is None:
        visited_keys = set()

    # Base case: if key doesn't exist in processed_data or has already been visited
    if processed_data.get(key) is None or key in visited_keys:
        return

    # Add current key to visited keys to prevent infinite recursion
    visited_keys.add(key)

    # Mark current interaction as removed by setting weight to -1
    processed_data[key]["weight"] = -1

    # Extract details from the key
    node1, node2, interaction, atoms, comp_type = key

    # Find all subsequent interactions depending on the comp_type
    if comp_type == "ligand_water":
        water_node = node2
        for next_key in list(processed_data.keys()):
            if (
                processed_data.get(next_key) is None
                or processed_data[next_key]["weight"] <= 0
            ):
                continue  # Skip already removed interactions

            next_n1, next_n2, _, _, next_type = next_key

            # Check if this water node is involved in subsequent interactions
            if (next_type == "water_protein" and next_n1 == water_node) or (
                next_type == "water_water" and (next_n1 == water_node)
            ):
                clean_processed_data(next_key, processed_data, visited_keys)

    elif comp_type == "water_water":
        water1, water2 = node1, node2
        for next_key in list(processed_data.keys()):
            if (
                processed_data.get(next_key) is None
                or processed_data[next_key]["weight"] <= 0
            ):
                continue  # Skip already removed interactions

            next_n1, next_n2, _, _, next_type = next_key

            # Check if either water node is involved in subsequent interactions
            if next_n1 in [water1, water2]:
                clean_processed_data(next_key, processed_data, visited_keys)


def check_till_protein(
    node: str, interaction_totals: dict, threshold: float, comp_type: str
) -> bool:
    """
    Check if the node is valid for inclusion in the filtered data.

    For water nodes, performs breadth-first search to determine if there's a path
    to any protein node through interactions meeting the threshold.
    """
    # Protein nodes are always valid
    if comp_type == "water_protein":
        return True

    # breadth-first search to find path to protein
    visited = set()
    queue = [(node, comp_type)]

    while queue:
        current_node, current_type = queue.pop(0)

        if current_node in visited:
            continue

        visited.add(current_node)

        # Check if this node is in any interaction that meets threshold
        for key, total in interaction_totals.items():
            if total < threshold:
                continue

            # If this node appears as first element in any interaction
            if key[0] == current_node:
                # If connected to protein, we're done
                if key[3] == "water_protein":
                    return True
                # Otherwise, add connected node to queue
                if key[3] in ["water_water", "ligand_water"]:
                    queue.append((key[1], key[3]))

    # If we've checked all connections and didn't find a path to protein
    return False


def calculate_coordinates(
    graph: nx.MultiGraph, ligand_mol: Chem.Mol, molsize: int
) -> nx.MultiGraph:
    """
    Calculate coordinates for graph nodes.

    1. Keep ligand nodes fixed at their existing positions
    2. Calculate initial positions for water and protein nodes based on ligand interactions
    3. Apply spring layout to optimize positions while keeping ligand nodes fixed
    4. Resolve overlaps between nodes

    Parameters
    ----------
    graph : nx.MultiGraph
        NetworkX graph with nodes and edges
    ligand_mol : Chem.Mol
        The ligand molecule for coordinate reference
    molsize : int
        Scaling factor for coordinates

    Returns
    -------
    nx.MultiGraph
        Graph with updated node coordinates
    """
    # Get conformer for ligand coordinates
    conformer = ligand_mol.GetConformer()

    # Separate nodes by type
    ligand_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("node_type") == "ligand"
    ]
    water_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("node_type") == "water"
    ]
    protein_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("node_type") == "protein"
    ]

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
                    conformer.GetAtomPosition(n).x * molsize,
                    conformer.GetAtomPosition(n).y * molsize,
                ]
                for n in ligand_nodes
            ]
        )
        # Update the graph with these coordinates
        for i, node in enumerate(ligand_nodes):
            graph.nodes[node]["x"] = float(ligand_coords[i][0])
            graph.nodes[node]["y"] = float(ligand_coords[i][1])

    if len(ligand_coords) == 0:
        # Last resort: create dummy coordinates
        center = np.array([0.0, 0.0])
        width = height = 100.0
    else:
        center = np.mean(ligand_coords, axis=0)
        ligand_bounds = np.max(ligand_coords, axis=0) - np.min(ligand_coords, axis=0)
        width = ligand_bounds[0]
        height = ligand_bounds[1]

    # Calculate initial positions for water and protein nodes
    pos = _calculate_initial_positions(
        graph, ligand_nodes, water_nodes, protein_nodes, center, width, height
    )

    # Apply spring layout with fixed ligand positions
    if len(water_nodes) + len(protein_nodes) > 0 and graph.number_of_edges() > 0:
        try:
            pos = nx.spring_layout(
                graph,
                pos=pos,
                fixed=ligand_nodes,
                scale=2,
                iterations=100,
                center=center,
                weight="weight_spring_layout",
            )
        except Exception:
            # If spring layout fails, keep initial positions
            pass

    # Resolve overlaps
    pos = _resolve_overlaps(
        pos, ligand_nodes, water_nodes, protein_nodes, ligand_coords, width, height
    )

    # Update node positions in graph
    for node, (x, y) in pos.items():
        graph.nodes[node]["x"] = float(x)
        graph.nodes[node]["y"] = float(y)

    return graph


def _calculate_initial_positions(
    graph: nx.Graph,
    ligand_nodes: list,
    water_nodes: list,
    protein_nodes: list,
    center: "np.ndarray",
    width: float,
    height: float,
) -> dict:
    """
    Calculate initial positions for water and protein nodes based on ligand interactions.

    approach:
    1. Find maximum order (number of water molecules in interaction chain)
    2. Create layers containing nodes at each level
    3. Calculate initial positions using direction vectors from ligand center
    """
    pos = {}

    # Set ligand atom positions (fixed)
    for node in ligand_nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            pos[node] = np.array([graph.nodes[node]["x"], graph.nodes[node]["y"]])

    # Use global max_order from _get_records
    # This avoids the need to calculate max_order again from the graph structure
    global _MAX_ORDER
    max_order = _MAX_ORDER

    # Create layers for positioning
    layers = _create_layers(graph, ligand_nodes, water_nodes, protein_nodes, max_order)

    # Position nodes layer by layer
    for layer_num, layer_nodes in layers.items():
        if layer_num == 0:  # Ligand layer - already positioned
            continue

        for node in layer_nodes:
            # Get connected nodes from previous layer
            connected_atoms = []
            for neighbor in graph.neighbors(node):
                if neighbor in pos:
                    connected_atoms.append(neighbor)

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
                    distance = width * 0.3 * (layer_num)
                    pos[node] = avg_pos + direction_unit * distance
                else:
                    # Random position if direction is zero
                    angle = hash(str(node)) % 360 * np.pi / 180
                    distance = width * 0.3 * (layer_num + 1)
                    pos[node] = center + distance * np.array(
                        [np.cos(angle), np.sin(angle)]
                    )
            else:
                # For unconnected nodes, place around periphery
                angle = hash(str(node)) % 360 * np.pi / 180
                distance = width * 0.5 * (layer_num + 1)
                pos[node] = center + distance * np.array([np.cos(angle), np.sin(angle)])

    return pos


def _create_layers(
    graph: nx.Graph,
    ligand_nodes: list,
    water_nodes: list,
    protein_nodes: list,
    max_order: int,
) -> dict:
    """
    Create layers containing nodes at each level.

    Layer 0: Ligand atoms (fixed positions)
    Layer 1+: Water and protein nodes based on their distance from ligand
    """
    layers = {0: ligand_nodes}

    # Initialize remaining nodes
    remaining_nodes = set(water_nodes + protein_nodes)
    visited_nodes = set(ligand_nodes)

    for layer_num in range(1, max_order + 2):  # Add extra layers for safety
        current_layer = []
        nodes_to_remove = set()

        for node in list(
            remaining_nodes
        ):  # Convert to list to avoid modification during iteration
            # Check if this node is connected to any node in previous layers
            connected_to_previous = False
            for neighbor in graph.neighbors(node):
                if neighbor in visited_nodes:
                    connected_to_previous = True
                    break

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
        last_layer = max(layers.keys()) + 1 if layers else 1
        layers[last_layer] = list(remaining_nodes)

    return layers


def _resolve_overlaps(
    pos: dict,
    ligand_nodes: list,
    water_nodes: list,
    protein_nodes: list,
    ligand_coords: "np.ndarray",
    width: float,
    height: float,
) -> dict:
    """
    Resolve overlaps between nodes.

    This function detects and resolves overlaps between:
    1. Non-ligand nodes with each other
    2. Non-ligand nodes with ligand atoms
    """
    min_distance = min(60, max(width, height) * 0.3)
    max_iterations = 100

    non_ligand_nodes = water_nodes + protein_nodes

    for iteration in range(max_iterations):
        overlap_found = False
        adjustments = {node: np.zeros(2) for node in non_ligand_nodes}

        # Phase 1: Resolve non-ligand to non-ligand overlaps
        for i, node1 in enumerate(non_ligand_nodes):
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

                if 0 < dist < min_distance:
                    overlap_found = True
                    force = (min_distance - dist) * 0.5
                    if iteration >= 50:
                        force *= 1.6
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

                if 0 < dist < min_distance:
                    overlap_found = True
                    force = (min_distance - dist) * 1.5
                    if iteration >= 50:
                        force *= 4 / 3
                    direction = delta / dist

                    ligand_adjustments[node] += direction * force

        # Apply ligand overlap adjustments
        for node, adjustment in ligand_adjustments.items():
            if node in pos:
                pos[node] += adjustment

        if not overlap_found:
            break

    return pos
