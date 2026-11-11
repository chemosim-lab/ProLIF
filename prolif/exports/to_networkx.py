"""
Export fingerprints as NetworkX graphs --- :mod:`prolif.exports.to_networkx`
=============================================================================

.. autofunction:: fp_to_networkx

"""

from collections import defaultdict
from operator import itemgetter
from statistics import fmean
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from rdkit import Chem

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import get_interaction_records, spring_layout_weight
from prolif.residue import ResidueId

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prolif.fingerprint import Fingerprint

Record = dict[str, Any]
RecordKey = tuple[str, str, str, tuple[int, ...], str]

# (source node type, target node type) for each kind of interaction record
_NODE_TYPES: dict[str, tuple[str, str]] = {
    "ligand_protein": ("ligand", "protein"),
    "ligand_water": ("ligand", "water"),
    "water_protein": ("water", "protein"),
    "water_water": ("water", "water"),
}


def fp_to_networkx(
    fp: "Fingerprint",
    ligand_mol: Chem.Mol,
    kind: Literal["aggregate", "frame"] = "aggregate",
    frame: int = 0,
    display_all: bool = False,
    threshold: float = 0.3,
    molsize: int = 35,
) -> nx.MultiGraph:
    """Convert a fingerprint to a NetworkX graph.

    The graph contains one node per ligand atom (``node_type="ligand"``) connected by
    ``edge_type="bond"`` edges, one node per interacting protein residue
    (``node_type="protein"``) and one node per bridging water molecule
    (``node_type="water"``). Interactions are stored as ``edge_type="interaction"``
    edges between the ligand atoms (or water molecules) and the residues they
    interact with. Every node carries 2D ``x`` and ``y`` coordinates suitable for
    drawing the network, with ligand atoms kept at their input positions.

    Parameters
    ----------
    fp : prolif.fingerprint.Fingerprint
        The fingerprint object containing interaction data.
    ligand_mol : rdkit.Chem.rdchem.Mol
        Ligand molecule with a conformer, used for the ligand atoms coordinates.
    kind : str
        One of ``"aggregate"`` or ``"frame"``.
    frame : int
        Frame number (as stored in ``fp.ifp``) for ``kind="frame"``.
    display_all : bool
        For ``kind="frame"``, create an edge for every occurrence of an interaction
        between a ligand and protein residue instead of only the shortest one.
    threshold : float
        For ``kind="aggregate"``, minimum frequency (fraction of frames, between 0
        and 1) for an interaction between a ligand and protein residue to be kept.
    molsize : int
        Scaling factor applied to the ligand coordinates.

    Returns
    -------
    networkx.MultiGraph
        Graph of the interactions. The ``weight`` attribute of interaction edges is
        the frequency of the interaction for aggregate graphs (``1`` for frame
        graphs), and ``distance`` is the (average) interaction distance.

    Raises
    ------
    prolif.exceptions.RunRequiredError
        If the fingerprint has not been run yet.
    ValueError
        If ``kind`` is unknown or ``frame`` is not in the fingerprint.
    """
    if not hasattr(fp, "ifp"):
        raise RunRequiredError(
            "Please run the fingerprint analysis before attempting to display results."
        )
    if kind == "frame":
        if frame not in fp.ifp:
            raise ValueError(f"Frame {frame} not found in fingerprint data")
        records = get_interaction_records(fp.ifp[frame], all_metadata=display_all)
        for record in records:
            record["weight"] = 1.0
    elif kind == "aggregate":
        records = _aggregate_records(fp, threshold)
    else:
        raise ValueError(f'{kind!r} must be "aggregate" or "frame"')

    graph: nx.MultiGraph = nx.MultiGraph()
    _add_ligand(graph, ligand_mol, molsize, use_segid=fp.use_segid or False)
    _add_interactions(graph, records)
    _compute_coordinates(graph)
    return graph


def _record_key(record: Record) -> RecordKey:
    return (
        record["ligand"],
        record["protein"],
        record["interaction"],
        tuple(record["atoms"]),
        record["components"],
    )


def _aggregate_records(fp: "Fingerprint", threshold: float) -> list[Record]:
    """Aggregate the interactions over all frames.

    The weight of each record is its frequency over the trajectory and its distance
    is the average distance. Interactions between a ligand and protein residue whose
    total frequency (summed over the different ligand atoms) is below ``threshold``
    are discarded, and only the most frequent set of ligand atoms is kept for each
    one. Water molecules that no longer bridge the ligand to a protein residue after
    this filtering are removed as well.
    """
    counts: dict[RecordKey, int] = defaultdict(int)
    distances: dict[RecordKey, list[float]] = defaultdict(list)
    for ifp in fp.ifp.values():
        seen: set[RecordKey] = set()
        for record in get_interaction_records(ifp, all_metadata=False):
            key = _record_key(record)
            # the same water segment can be part of several water bridges
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            distances[key].append(record["distance"])

    n_frames = len(fp.ifp)
    records: list[Record] = [
        {
            "ligand": key[0],
            "protein": key[1],
            "interaction": key[2],
            "atoms": key[3],
            "components": key[4],
            "weight": count / n_frames,
            "distance": fmean(distances[key]),
        }
        for key, count in counts.items()
    ]

    # merge the different ligand atoms of the same residue/interaction group before
    # applying the threshold
    totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for record in records:
        totals[_group_key(record)] += record["weight"]
    records = [r for r in records if totals[_group_key(r)] >= threshold]

    # keep the most occurring ligand atoms
    best: dict[tuple[str, str, str], Record] = {}
    for record in sorted(records, key=lambda r: (-r["weight"], r["atoms"])):
        best.setdefault(_group_key(record), record)

    return _prune_dangling_waters(list(best.values()))


_group_key = itemgetter("ligand", "protein", "interaction")


def _prune_dangling_waters(records: list[Record]) -> list[Record]:
    """Remove the water molecules (and their edges) that are not on a path between
    the ligand and a protein residue, which can happen after thresholding."""
    graph: nx.Graph = nx.Graph()
    node_types: dict[str, str] = {}
    for record in records:
        source_type, target_type = _NODE_TYPES[record["components"]]
        node_types[record["ligand"]] = source_type
        node_types[record["protein"]] = target_type
        graph.add_edge(record["ligand"], record["protein"])

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
    return [r for r in records if r["ligand"] in valid and r["protein"] in valid]


def _add_ligand(
    graph: nx.MultiGraph, ligand_mol: Chem.Mol, molsize: int, use_segid: bool
) -> None:
    """Add the ligand atoms as nodes and bonds as edges."""
    xyz = ligand_mol.GetConformer().GetPositions() * molsize
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        graph.add_node(
            idx,
            node_type="ligand",
            symbol=atom.GetSymbol(),
            charge=atom.GetFormalCharge(),
            label=atom.GetSymbol(),
            ligand=str(ResidueId.from_atom(atom, use_segid=use_segid)),
            x=float(xyz[idx, 0]),
            y=float(xyz[idx, 1]),
            z=float(xyz[idx, 2]),
        )
    for bond in ligand_mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            edge_type="bond",
            bond_type=bond.GetBondType(),
            bond_order=bond.GetBondTypeAsDouble(),
        )


def _add_node(graph: nx.MultiGraph, node: str, node_type: str) -> None:
    if graph.has_node(node):
        return
    if node_type == "protein":
        graph.add_node(node, node_type="protein", residue=node, label=node)
    else:
        graph.add_node(node, node_type="water", water_id=node, label=node)


def _add_interactions(graph: nx.MultiGraph, records: list[Record]) -> None:
    """Add interaction edges, creating the protein and water nodes as needed."""
    for record in records:
        source_type, target_type = _NODE_TYPES[record["components"]]
        target = record["protein"]
        _add_node(graph, target, target_type)
        attrs = {
            "edge_type": "interaction",
            "interaction_type": record["interaction"],
            "weight": record["weight"],
            "distance": record["distance"],
            "components": record["components"],
            "weight_spring_layout": spring_layout_weight(record["distance"]),
        }
        if source_type == "ligand":
            # one edge per ligand atom involved in the interaction
            for idx in record["atoms"]:
                graph.add_edge(idx, target, **attrs)
        else:
            source = record["ligand"]
            _add_node(graph, source, source_type)
            graph.add_edge(source, target, **attrs)


def _compute_coordinates(graph: nx.MultiGraph) -> None:
    """Compute 2D coordinates for the protein and water nodes.

    1. Ligand atoms are kept fixed at their existing positions.
    2. Water and protein nodes are initially placed outside of the ligand, next to
       the nodes they are connected to.
    3. The positions are refined with a spring layout (ligand atoms stay fixed).
    4. Overlaps between nodes are resolved.
    """
    ligand_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "ligand"]
    other_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] != "ligand"]
    if not other_nodes:
        return

    ligand_xy: NDArray[np.float64] = np.array(
        [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in ligand_nodes]
    ).reshape(-1, 2)
    if len(ligand_xy):
        center = ligand_xy.mean(axis=0)
        width, height = ligand_xy.max(axis=0) - ligand_xy.min(axis=0)
    else:
        center = np.zeros(2)
        width = height = 100.0

    pos = _initial_positions(graph, ligand_nodes, other_nodes, center, width)
    if graph.number_of_edges() > 0:
        pos = nx.spring_layout(
            graph,
            pos=pos,
            fixed=ligand_nodes,
            scale=2,
            iterations=100,
            center=center,
            weight="weight_spring_layout",
        )
    min_distance = min(60, max(width, height) * 0.3)
    pos = _resolve_overlaps(pos, other_nodes, ligand_xy, min_distance)

    for node in other_nodes:
        graph.nodes[node]["x"] = float(pos[node][0])
        graph.nodes[node]["y"] = float(pos[node][1])


def _periphery_position(
    index: int, count: int, center: "NDArray[np.float64]", radius: float
) -> "NDArray[np.float64]":
    """Position on a circle around ``center``, evenly spaced for ``count`` nodes."""
    angle = 2 * np.pi * index / count
    return center + radius * np.array([np.cos(angle), np.sin(angle)])


def _initial_positions(
    graph: nx.MultiGraph,
    ligand_nodes: list[Any],
    other_nodes: list[Any],
    center: "NDArray[np.float64]",
    width: float,
) -> dict[Any, "NDArray[np.float64]"]:
    """Initial positions for the water and protein nodes.

    Nodes are placed layer by layer according to their distance from the ligand:
    first the nodes directly connected to ligand atoms, then the nodes connected to
    those, and so on. Each node is pushed away from the ligand center starting from
    the average position of its already placed neighbors.
    """
    pos: dict[Any, NDArray[np.float64]] = {
        node: np.array([graph.nodes[node]["x"], graph.nodes[node]["y"]])
        for node in ligand_nodes
    }
    remaining = set(other_nodes)
    layer = 1
    while remaining:
        current = sorted(
            (n for n in remaining if any(nb in pos for nb in graph.neighbors(n))),
            key=str,
        )
        if not current:
            # not connected to the ligand: place around the periphery
            for i, node in enumerate(sorted(remaining, key=str)):
                radius = width * 0.5 * (layer + 1)
                pos[node] = _periphery_position(i, len(remaining), center, radius)
            break
        for i, node in enumerate(current):
            anchors = np.array([pos[nb] for nb in graph.neighbors(node) if nb in pos])
            avg_pos = anchors.mean(axis=0)
            direction = avg_pos - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                pos[node] = avg_pos + direction / norm * width * 0.3 * layer
            else:
                radius = width * 0.3 * (layer + 1)
                pos[node] = _periphery_position(i, len(current), center, radius)
        remaining.difference_update(current)
        layer += 1
    return pos


def _resolve_overlaps(
    pos: dict[Any, "NDArray[np.float64]"],
    nodes: list[Any],
    ligand_xy: "NDArray[np.float64]",
    min_distance: float,
    max_iterations: int = 100,
) -> dict[Any, "NDArray[np.float64]"]:
    """Push apart the (non-ligand) nodes that are closer than ``min_distance`` from
    each other or from a ligand atom."""
    for iteration in range(max_iterations):
        overlap_found = False
        # stronger pushes in the second half if overlaps persist
        node_factor = 1.6 if iteration >= 50 else 1.0
        ligand_factor = 2.0 if iteration >= 50 else 1.5

        # non-ligand nodes with each other
        shifts = {node: np.zeros(2) for node in nodes}
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                delta = pos[node2] - pos[node1]
                dist = np.linalg.norm(delta)
                if dist == 0:
                    delta = np.array([1.0, 1.0])
                    dist = np.linalg.norm(delta)
                if dist < min_distance:
                    overlap_found = True
                    push = (min_distance - dist) * node_factor * delta / dist
                    shifts[node1] -= push
                    shifts[node2] += push
        for node in nodes:
            pos[node] += shifts[node]

        # non-ligand nodes with ligand atoms
        for node in nodes:
            deltas = pos[node] - ligand_xy
            dists = np.linalg.norm(deltas, axis=1)
            mask = (dists > 0) & (dists < min_distance)
            if mask.any():
                overlap_found = True
                pushes = (min_distance - dists[mask]) * ligand_factor / dists[mask]
                pos[node] += (pushes[:, None] * deltas[mask]).sum(axis=0)

        if not overlap_found:
            break
    return pos
