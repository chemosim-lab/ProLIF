import warnings
from typing import Any
from unittest.mock import MagicMock

import MDAnalysis as mda
import networkx as nx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

import prolif as plf
from prolif.exceptions import RunRequiredError
from prolif.exports.to_networkx import (
    _AggKey,
    _AggValue,
    _prune_dangling_waters,
    build_graph,
    to_networkx,
)


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


# ---- tests on real fingerprints ----


@pytest.fixture(scope="module")
def fp_mol() -> tuple[plf.Fingerprint, plf.Molecule]:
    u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
    lig = u.select_atoms("resname LIG")
    prot = u.select_atoms("protein and byres around 6.5 group ligand", ligand=lig)
    fp = plf.Fingerprint()
    fp.run(u.trajectory[0:3], lig, prot, progress=False, n_jobs=1)
    return fp, plf.Molecule.from_mda(lig)


@pytest.fixture(scope="module")
def water_fp_mol(
    water_mols: tuple[plf.Molecule, plf.Molecule, plf.Molecule],
) -> tuple[plf.Fingerprint, plf.Molecule]:
    ligand, protein, water = water_mols
    fp = plf.Fingerprint(
        ["HBDonor", "WaterBridge"],
        parameters={"WaterBridge": {"water": water, "order": 2}},
    )
    fp.run_from_iterable([ligand], protein, progress=False)
    return fp, ligand


def interaction_edges(
    graph: "nx.MultiGraph[int | str]",
) -> list[tuple[Any, Any, dict[str, Any]]]:
    return [
        (u, v, d)
        for u, v, d in graph.edges(data=True)
        if d["edge_type"] == "interaction"
    ]


def nodes_of_type(graph: "nx.MultiGraph[int | str]", node_type: str) -> list[Any]:
    return [n for n, d in graph.nodes(data=True) if d["node_type"] == node_type]


def frequency(fp: plf.Fingerprint, target: str, interaction: str) -> float:
    """Fraction of frames where the interaction with the protein residue occurs"""
    count = sum(
        1
        for ifp in fp.ifp.values()
        for (_, prot), data in ifp.items()
        if str(prot) == target and interaction in data
    )
    return count / len(fp.ifp)


def test_raises_not_executed(ligand_mol: Chem.Mol) -> None:
    fp = plf.Fingerprint()
    with pytest.raises(RunRequiredError, match="Please run the fingerprint analysis"):
        to_networkx(fp, ligand_mol)


def test_raises_kind(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
        to_networkx(fp, lig_mol, kind="foo")  # type: ignore[arg-type]


def test_raises_missing_frame(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    with pytest.raises(ValueError, match="Frame 42 not found"):
        to_networkx(fp, lig_mol, kind="frame", frame=42)


def test_frame_graph(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    graph = to_networkx(fp, lig_mol, kind="frame", frame=0, mol_scale=10)
    assert isinstance(graph, nx.MultiGraph)

    # ligand atoms and bonds
    assert len(nodes_of_type(graph, "ligand")) == lig_mol.GetNumAtoms()
    bonds = [d for *_, d in graph.edges(data=True) if d["edge_type"] == "bond"]
    assert len(bonds) == lig_mol.GetNumBonds()
    xyz = lig_mol.GetConformer().GetPositions() * 10
    lig_resid = str(next(iter(fp.ifp[0]))[0])
    for atom in lig_mol.GetAtoms():
        node = graph.nodes[atom.GetIdx()]
        assert node["symbol"] == atom.GetSymbol()
        assert node["ligand"] == lig_resid
        assert np.allclose([node["x"], node["y"], node["z"]], xyz[atom.GetIdx()])

    # protein residues
    protein_nodes = nodes_of_type(graph, "protein")
    assert set(protein_nodes) == {str(prot) for _, prot in fp.ifp[0]}
    for res in protein_nodes:
        assert graph.nodes[res]["residue"] == res
        assert np.isfinite([graph.nodes[res]["x"], graph.nodes[res]["y"]]).all()

    # interactions
    edges = interaction_edges(graph)
    expected = {
        (str(prot), name) for (_, prot), data in fp.ifp[0].items() for name in data
    }
    assert {(v, d["interaction"]) for _, v, d in edges} == expected
    for u, v, d in edges:
        assert isinstance(u, int) and isinstance(v, str)
        assert d["components"] == "ligand_protein"
        assert u in d["atoms"]
        assert d["weight"] == 1
        assert d["distance"] > 0
        assert d["weight_spring_layout"] == pytest.approx(10 / d["distance"])


def test_frame_display_all(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    default = interaction_edges(to_networkx(fp, lig_mol, kind="frame"))
    all_edges = interaction_edges(
        to_networkx(fp, lig_mol, kind="frame", display_all=True)
    )
    assert len(all_edges) >= len(default)
    # no duplicated edges
    keys = [(u, v, d["interaction"], d["atoms"]) for u, v, d in all_edges]
    assert len(keys) == len(set(keys))


def test_aggregate_frequencies(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    graph = to_networkx(fp, lig_mol, kind="aggregate", threshold=0)
    edges = interaction_edges(graph)
    assert edges
    atom_sets: dict[tuple[str, str], set[tuple[int, ...]]] = {}
    for u, v, d in edges:
        # the weight is the frequency of the set of ligand atoms drawn, which is at
        # most the frequency of the interaction between the two residues
        assert 0 < d["weight"] <= frequency(fp, v, d["interaction"]) <= 1
        assert u in d["atoms"]
        atom_sets.setdefault((v, d["interaction"]), set()).add(d["atoms"])
    # only the most frequent set of ligand atoms is kept for each interaction
    assert all(len(atoms) == 1 for atoms in atom_sets.values())
    # interactions always involving the same atoms have the interaction frequency
    for (v, interaction), atoms in atom_sets.items():
        (atoms_tuple,) = atoms
        seen = {
            metadata["parent_indices"]["ligand"]
            for ifp in fp.ifp.values()
            for (_, prot), data in ifp.items()
            if str(prot) == v
            for metadata in data.get(interaction, ())
        }
        if seen == {atoms_tuple}:
            weight = next(
                d["weight"]
                for _, v_, d in edges
                if v_ == v and d["interaction"] == interaction
            )
            assert weight == pytest.approx(frequency(fp, v, interaction))


def test_aggregate_threshold(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    no_filter = to_networkx(fp, lig_mol, threshold=0)
    strict = to_networkx(fp, lig_mol, threshold=1)
    groups_no_filter = {
        (v, d["interaction"]) for _, v, d in interaction_edges(no_filter)
    }
    groups_strict = {(v, d["interaction"]) for _, v, d in interaction_edges(strict)}
    assert groups_strict and groups_strict < groups_no_filter
    for target, interaction in groups_strict:
        assert frequency(fp, target, interaction) == 1
    # weights are frequencies, so nothing can pass a threshold above 1
    empty = to_networkx(fp, lig_mol, threshold=1.5)
    assert not interaction_edges(empty)
    assert len(empty.nodes) == lig_mol.GetNumAtoms()


def test_aggregate_threshold_sums_over_atoms(ligand_mol: Chem.Mol) -> None:
    """An interaction present in every frame but through different ligand atoms
    passes the threshold, and the most frequent set of atoms is used."""
    fp = MagicMock()
    frames = [(0,), (1,), (1,)]
    fp.ifp = {
        i: {
            ("LIG1", "PRO100.A"): {
                "Hydrophobic": [{"parent_indices": {"ligand": atoms}, "distance": 3.5}]
            }
        }
        for i, atoms in enumerate(frames)
    }
    graph = to_networkx(fp, ligand_mol, kind="aggregate", threshold=1)
    edges = interaction_edges(graph)
    assert len(edges) == 1
    u, v, d = edges[0]
    assert (u, v, d["atoms"]) == (1, "PRO100.A", (1,))
    assert d["weight"] == pytest.approx(2 / 3)


def test_zero_distance(ligand_mol: Chem.Mol) -> None:
    """Interactions without a distance don't break the layout weights"""
    fp = MagicMock()
    fp.ifp = {
        0: {("LIG1", "PRO100.A"): {"Foo": [{"parent_indices": {"ligand": (0,)}}]}}
    }
    graph = to_networkx(fp, ligand_mol, kind="frame")
    ((_, _, d),) = interaction_edges(graph)
    assert d["distance"] == 0
    assert d["weight_spring_layout"] == 1


@pytest.mark.parametrize("kind", ["frame", "aggregate"])
def test_water_bridge(
    water_fp_mol: tuple[plf.Fingerprint, plf.Molecule], kind: str
) -> None:
    fp, lig_mol = water_fp_mol
    graph = to_networkx(fp, lig_mol, kind=kind)  # type: ignore[arg-type]
    expected_waters = {
        str(water)
        for ifp in fp.ifp.values()
        for data in ifp.values()
        for metadata in data.get("WaterBridge", ())
        for water in metadata["water_residues"]
    }
    assert len(expected_waters) == 2
    water_nodes = nodes_of_type(graph, "water")
    assert set(water_nodes) == expected_waters
    for water in water_nodes:
        assert graph.nodes[water]["water_id"] == water
        # every water molecule bridges the ligand (or another water) to the protein
        assert graph.degree[water] >= 2
    components = {d["components"] for *_, d in interaction_edges(graph)}
    assert components == {
        "ligand_protein",
        "ligand_water",
        "water_water",
        "water_protein",
    }
    for *_, d in interaction_edges(graph):
        if d["components"] != "ligand_protein":
            assert d["interaction"] == "WaterBridge"
            # single frame: a segment shared by several bridges is only counted once
            assert d["weight"] == 1
    # protein residues reached through the second water are connected to the ligand
    for _, v, d in interaction_edges(graph):
        if d["components"] == "water_protein":
            assert nx.has_path(graph, 0, v)


def test_prune_dangling_waters() -> None:
    value = _AggValue(weight=1.0, distance=3.0)

    def rec(src: str, dest: str, components: str) -> _AggKey:
        return (src, dest, "WaterBridge", (), components)

    chain: dict[_AggKey, _AggValue] = {
        rec("LIG1", "W1", "ligand_water"): value,
        rec("W1", "W2", "water_water"): value,
        rec("W2", "RES1", "water_protein"): value,
        ("LIG1", "RES2", "Hydrophobic", (0,), "ligand_protein"): value,
    }
    dangling: dict[_AggKey, _AggValue] = {
        rec("LIG1", "W3", "ligand_water"): value,  # water only touching the ligand
        rec("W3", "W4", "water_water"): value,
        rec("W5", "RES3", "water_protein"): value,  # water only touching the protein
    }
    assert _prune_dangling_waters({**chain, **dangling}) == chain
    assert _prune_dangling_waters(dangling) == {}


def test_deterministic_coordinates(
    fp_mol: tuple[plf.Fingerprint, plf.Molecule],
) -> None:
    fp, lig_mol = fp_mol
    graph = to_networkx(fp, lig_mol, kind="aggregate", mol_scale=35)
    again = to_networkx(fp, lig_mol, kind="aggregate", mol_scale=35)
    for node in nodes_of_type(graph, "protein"):
        assert np.isfinite([graph.nodes[node]["x"], graph.nodes[node]["y"]]).all()
        assert graph.nodes[node]["x"] == again.nodes[node]["x"]
        assert graph.nodes[node]["y"] == again.nodes[node]["y"]


def test_empty_fingerprint(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    """No warning when no interaction was detected"""
    _, lig_mol = fp_mol
    u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
    lig = u.select_atoms("resname LIG")
    prot = u.select_atoms("protein and resid 45:50")  # far from the ligand
    fp = plf.Fingerprint(["Hydrophobic"])
    fp.run(u.trajectory[0:1], lig, prot, progress=False, n_jobs=1)
    assert not fp.ifp[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = to_networkx(fp, lig_mol, kind="frame")
    assert len(graph.nodes) == 0
