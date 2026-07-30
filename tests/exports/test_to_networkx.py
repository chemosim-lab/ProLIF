from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from prolif.exports.to_networkx import build_graph, to_networkx


@pytest.fixture(scope="session")
def ligand_mol() -> Chem.Mol:
    """Create a simple ligand molecule (benzene) for testing"""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    EmbedMolecule(mol, randomSeed=42)
    return mol


@pytest.fixture(scope="session")
def fp_mock() -> MagicMock:
    """Create a mock Fingerprint object with interaction data"""
    fp = MagicMock()
    # Data structure for IFP: {residue_pair: {interaction_name: [metadata_dict, ...]}}
    ifp_data = {
        ("LIG1", "PRO100.A"): {
            "Hydrophobic": [
                {"parent_indices": {"ligand": (0,)}, "distance": 3.5, "weight": 1.0}
            ],
            "HBAcceptor": [
                {"parent_indices": {"ligand": (1,)}, "distance": 2.9, "weight": 0.8}
            ],
        },
        ("LIG1", "PHE102.A"): {
            "PiStacking": [
                {
                    "parent_indices": {"ligand": (0, 1, 2, 3, 4, 5)},
                    "distance": 4.0,
                    "weight": 0.9,
                }
            ],
        },
    }
    fp.ifp = {0: ifp_data}
    return fp


@pytest.fixture(scope="session")
def graph(fp_mock: MagicMock, ligand_mol: Chem.Mol) -> "nx.MultiGraph[int | str]":
    return build_graph(fp_mock, ligand_mol, frame=0, kind="frame")


@pytest.fixture
def complex_fp_mock() -> MagicMock:
    """Create a mock Fingerprint object for overlap testing"""
    fp = MagicMock()
    ifp_data = {
        ("LIG1", "PRO100.A"): {
            "Hydrophobic": [
                {"parent_indices": {"ligand": (0,)}, "distance": 3.5, "weight": 1.0}
            ],
            "HBAcceptor": [
                {"parent_indices": {"ligand": (1,)}, "distance": 2.9, "weight": 0.8}
            ],
        },
        ("LIG1", "PHE102.A"): {
            "PiStacking": [
                {
                    "parent_indices": {"ligand": (0, 1, 2, 3, 4, 5)},
                    "distance": 4.0,
                    "weight": 0.9,
                }
            ],
        },
        ("LIG1", "ARG103.A"): {
            "Hydrophobic": [
                {"parent_indices": {"ligand": (2,)}, "distance": 3.8, "weight": 0.7}
            ],
        },
        ("LIG1", "LYS104.A"): {
            "Hydrophobic": [
                {"parent_indices": {"ligand": (3,)}, "distance": 4.1, "weight": 0.6}
            ],
        },
        ("LIG1", "ASP105.A"): {
            "HBDonor": [
                {"parent_indices": {"ligand": (4, 10)}, "distance": 3.2, "weight": 0.5}
            ],
        },
        ("LIG1", "GLU106.A"): {
            "HBAcceptor": [
                {"parent_indices": {"ligand": (5,)}, "distance": 3.0, "weight": 0.4}
            ],
        },
    }
    fp.ifp = {0: ifp_data}
    return fp


def test_build_graph_structure(
    graph: "nx.MultiGraph[int | str]", ligand_mol: Chem.Mol
) -> None:
    """Test build_graph output structure"""

    # Check that the graph has the correct number of nodes
    # All atoms + 2 protein residues
    expected_nodes = ligand_mol.GetNumAtoms() + 2
    assert len(graph.nodes) == expected_nodes

    # Check for ligand atoms
    for i in range(ligand_mol.GetNumAtoms()):
        assert i in graph.nodes
        assert graph.nodes[i]["node_type"] == "ligand"

    # Check for protein residues
    expected_residues = ["PRO100.A", "PHE102.A"]
    for res in expected_residues:
        assert res in graph.nodes
        assert graph.nodes[res]["node_type"] == "protein"

    # Check for bonds
    bond_edges = 0
    for _, _, data in graph.edges(data=True):
        if data.get("edge_type") == "bond":
            bond_edges += 1

    # There should be the same number of bonds as in the molecule
    assert bond_edges == ligand_mol.GetNumBonds()


def test_interaction_edge_attributes(
    graph: "nx.MultiGraph[int | str]", fp_mock: MagicMock, ligand_mol: Chem.Mol
) -> None:
    """Test that interaction edges have the correct attributes"""
    # Track interactions found in the graph
    interaction_types = {"Hydrophobic", "HBAcceptor", "PiStacking"}
    found_interactions = dict.fromkeys(interaction_types, False)

    # Check all edges for interactions
    interaction_count = 0
    for u, v, data in graph.edges(data=True):
        if data["edge_type"] == "interaction":
            interaction_count += 1

            # One end should be a string (residue ID) and one an int (atom index)
            # Depending on direction, u or v could be the atom index
            atom_idx = u if isinstance(u, int) else v
            res_id = v if isinstance(u, int) else u
            assert isinstance(atom_idx, int)
            assert isinstance(res_id, str)

            # Mark interaction as found if matches
            interaction = data["interaction"]
            if interaction in found_interactions:
                found_interactions[interaction] = True
                assert atom_idx in data["atoms"]

    # All interactions should have been found
    for interaction, found in found_interactions.items():
        assert found, f"Interaction {interaction} not found in the graph"


def test_node_attributes(
    graph: "nx.MultiGraph[int | str]", ligand_mol: Chem.Mol
) -> None:
    """Test that nodes have the correct attributes"""
    # Check ligand atom attributes
    for i in range(ligand_mol.GetNumAtoms()):
        node = graph.nodes[i]
        assert node["node_type"] == "ligand"
        assert node["symbol"] == ligand_mol.GetAtomWithIdx(i).GetSymbol()
        assert all(coord in node for coord in ["x", "y", "z"])


def test_empty_interaction_graph(ligand_mol: Chem.Mol) -> None:
    """Test with an empty interaction graph"""
    fp = MagicMock()
    fp.ifp = {0: {}}
    G = build_graph(fp, ligand_mol, frame=0, kind="frame")

    # build_graph returns an empty graph if there are no records
    assert len(G.nodes) == 0
    assert len(G.edges) == 0


def test_no_overlaps_in_to_networkx_coordinates(
    complex_fp_mock: MagicMock, ligand_mol: Chem.Mol
) -> None:
    """Test that coordinates generated by to_networkx don't have overlaps between
    residues and between residues and ligand atoms"""
    mol_scale = 35
    G = to_networkx(
        complex_fp_mock, ligand_mol, frame=0, kind="frame", mol_scale=mol_scale
    )

    # Extract protein residue nodes
    protein_nodes = {}
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "protein":
            protein_nodes[node_id] = np.array([data.get("x", 0), data.get("y", 0)])

    # Extract ligand atom nodes
    ligand_nodes = {}
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "ligand":
            ligand_nodes[node_id] = np.array([data.get("x", 0), data.get("y", 0)])

    # Define minimum acceptable distance
    min_distance = 2.9 * mol_scale

    # Test 1: Check for residue-residue overlaps
    for res1_id, res1_pos in protein_nodes.items():
        for res2_id, res2_pos in protein_nodes.items():
            if res1_id != res2_id:
                distance = float(np.linalg.norm(res1_pos - res2_pos))
                assert distance > min_distance, (
                    f"Residue-residue overlap detected between {res1_id} and "
                    f"{res2_id}: {distance=:.1f}"
                )

    # Test 2: Check for residue-ligand atom overlaps
    # mol_scale * distance between carbon and centroid of benzene ring * tolerance
    min_distance = mol_scale * 2.6

    for res_id, res_pos in protein_nodes.items():
        for atom_id, atom_pos in ligand_nodes.items():
            distance = float(np.linalg.norm(res_pos - atom_pos))
            assert distance > min_distance, (
                f"Residue-ligand overlap detected between {res_id} and atom "
                f"{atom_id}: {distance:.1f}"
            )
