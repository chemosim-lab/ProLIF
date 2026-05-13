from typing import Any, cast

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from prolif.plotting.network.lignetwork import LigNetwork


@pytest.fixture
def simple_ligand_mol() -> Chem.Mol:
    """Create a simple ligand molecule (benzene) for testing"""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    EmbedMolecule(mol, randomSeed=42)
    return mol


@pytest.fixture
def simple_interaction_df() -> pd.DataFrame:
    """Create a simple DataFrame with ligand-protein interactions"""
    # Create a simple dataframe with interactions
    data = [
        # ligand, protein, interaction, atoms, weight, distance, components
        ["LIG1", "PRO100.A", "Hydrophobic", (0,), 1.0, 3.5, "ligand_protein"],
        ["LIG1", "PRO100.A", "HBAcceptor", (1,), 0.8, 2.9, "ligand_protein"],
        [
            "LIG1",
            "PHE102.A",
            "PiStacking",
            (0, 1, 2, 3, 4, 5),
            0.9,
            4.0,
            "ligand_protein",
        ],
    ]

    # Create the multi-level index
    idx = pd.MultiIndex.from_tuples(
        [(row[0], row[1], row[2], row[3]) for row in data],
        names=["ligand", "protein", "interaction", "atoms"],
    )

    # Create DataFrame with weight and distance columns
    df = pd.DataFrame(
        [(row[4], row[5], row[6]) for row in data],
        index=idx,
        columns=["weight", "distance", "components"],
    )

    return df


@pytest.fixture
def lignetwork_obj(
    simple_ligand_mol: Chem.Mol, simple_interaction_df: pd.DataFrame
) -> LigNetwork:
    """Create a LigNetwork object for testing"""
    return LigNetwork(simple_interaction_df, simple_ligand_mol)


def test_to_networkx_graph(
    lignetwork_obj: LigNetwork, simple_ligand_mol: Chem.Mol
) -> None:
    """Test to_networkx"""
    G = lignetwork_obj.to_networkx()

    # Check that the graph has the correct number of nodes
    # All atoms + 2 protein residues
    expected_nodes = simple_ligand_mol.GetNumAtoms() + 2
    assert len(G.nodes) == expected_nodes

    # Check for ligand atoms
    for i in range(simple_ligand_mol.GetNumAtoms()):
        assert i in G.nodes
        assert G.nodes[i]["node_type"] == "ligand"

    # Check for protein residues
    expected_residues = ["PRO100.A", "PHE102.A"]  # Updated to match the data
    for res in expected_residues:
        assert res in G.nodes
        assert G.nodes[res]["node_type"] == "protein"
        assert G.nodes[res]["label"] == res

    # Check for bonds
    bond_edges = 0
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "bond":
            bond_edges += 1

    # There should be the same number of bonds as in the molecule
    assert bond_edges == simple_ligand_mol.GetNumBonds()


def test_interaction_edge_attributes(
    lignetwork_obj: LigNetwork, simple_interaction_df: pd.DataFrame
) -> None:
    """Test that interaction edges have the correct attributes"""
    G = lignetwork_obj.to_networkx()

    # Track interactions found in the graph
    interaction_types = set(simple_interaction_df.index.get_level_values("interaction"))
    found_interactions = dict.fromkeys(interaction_types, False)

    # Check all edges for interactions
    interaction_count = 0
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "interaction":
            interaction_count += 1

            # One end should be a string (residue ID) and one an int (atom index)
            assert (isinstance(u, int) and isinstance(v, str)) or (
                isinstance(u, str) and isinstance(v, int)
            )

            # Get the atom index and residue ID
            atom_idx = u if isinstance(u, int) else v
            residue = v if isinstance(v, str) else u

            # Mark interaction as found if matches
            interaction_type = data.get("interaction_type")
            if interaction_type in found_interactions:
                found_interactions[interaction_type] = True

                # Find matching row in dataframe - use .xs() instead of .loc[] for type safety
                try:
                    matching_rows = simple_interaction_df.xs(
                        (slice(None), residue, interaction_type),
                        level=cast(Any, ["ligand", "protein", "interaction"]),
                        drop_level=False,
                    )
                except KeyError:
                    matching_rows = pd.DataFrame()

                # At least one row should match
                assert not matching_rows.empty

                # For at least one row, the atom should be part of the interaction
                atom_in_interaction = False
                for idx, _ in matching_rows.iterrows():
                    # Get the atoms from the multi-index - it's the 'atoms' level (4th position)
                    atoms = cast(
                        tuple[int, ...], idx[-1] if isinstance(idx, tuple) else idx
                    )
                    if atom_idx in atoms:
                        atom_in_interaction = True
                        break

                assert atom_in_interaction

    # All interactions should have been found
    for interaction, found in found_interactions.items():
        assert found, f"Interaction {interaction} not found in the graph"


def test_node_attributes(lignetwork_obj: LigNetwork) -> None:
    """Test that nodes have the correct attributes"""
    G = lignetwork_obj.to_networkx()

    # Check ligand atom attributes
    for i in range(lignetwork_obj.mol.GetNumAtoms()):
        node = G.nodes[i]
        assert node["node_type"] == "ligand"
        assert node["symbol"] == lignetwork_obj.mol.GetAtomWithIdx(i).GetSymbol()
        assert "coords" in node
        assert len(node["coords"]) == 3  # 3D coordinates


def test_empty_interaction_df() -> None:
    """Test with an empty interactions DataFrame"""
    # Create a small molecule
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    EmbedMolecule(mol, randomSeed=42)

    # Create an empty DataFrame with the right structure
    idx = pd.MultiIndex.from_tuples(
        [], names=["ligand", "protein", "interaction", "atoms"]
    )
    df = pd.DataFrame([], index=idx, columns=["weight", "distance", "components"])

    # Create the LigNetwork and convert to NetworkX
    network = LigNetwork(df, mol)
    G = network.to_networkx()

    # Should only have ligand atoms, no protein nodes, no interaction edges
    assert len(G.nodes) == mol.GetNumAtoms()
    assert (
        len([n for n, d in G.nodes(data=True) if d.get("node_type") == "protein"]) == 0
    )
    assert (
        len([e for e in G.edges(data=True) if e[2].get("edge_type") == "interaction"])
        == 0
    )
