from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
from rdkit import Chem

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import metadata_iterator

if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP

# Constants
_BRIDGED_INTERACTIONS: dict[str, str] = {"WaterBridge": "water_residues"}


def fp_to_networkx(
    fp: "Fingerprint",
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
        return _make_frame_graph_from_fp(fp, ligand_mol, frame, display_all)
    if kind == "aggregate":
        return _make_agg_graph_from_fp(fp, ligand_mol, threshold)
    raise ValueError(f'{kind!r} must be "aggregate" or "frame"')


def _get_records(ifp: "IFP", all_metadata: bool) -> list[dict[str, Any]]:
    """Extract interaction records from fingerprint data."""
    records: list[dict[str, Any]] = []

    for (lig_resid, prot_resid), int_data in ifp.items():
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
    _add_ligand_nodes_and_bonds(graph, ligand_mol, ligand_id)

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
                )

    return graph


def _add_ligand_nodes_and_bonds(
    graph: nx.Graph, ligand_mol: Chem.Mol, ligand_id: str
) -> None:
    """Add ligand atoms and bonds to the graph."""
    # Add ligand atom nodes
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        graph.add_node(
            idx,
            node_type="ligand",
            symbol=atom.GetSymbol(),
            charge=atom.GetFormalCharge(),
            label=atom.GetSymbol(),
            ligand=ligand_id,
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
    _add_ligand_nodes_and_bonds(graph, ligand_mol, ligand_id)

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
                )

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
