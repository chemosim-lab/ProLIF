from typing import Any, cast

import MDAnalysis as mda
import networkx as nx
import numpy as np
import pytest

import prolif as plf
from prolif.exceptions import RunRequiredError
from prolif.exports import fp_to_networkx
from prolif.exports.to_networkx import _prune_dangling_waters
from prolif.plotting.network import LigNetwork
from prolif.plotting.utils import spring_layout_weight


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


def interaction_edges(graph: nx.MultiGraph) -> list[tuple[Any, Any, dict[str, Any]]]:
    return [
        (u, v, d)
        for u, v, d in graph.edges(data=True)
        if d["edge_type"] == "interaction"
    ]


def nodes_of_type(graph: nx.MultiGraph, node_type: str) -> list[Any]:
    return [n for n, d in graph.nodes(data=True) if d["node_type"] == node_type]


def test_raises_not_executed(ligand_mol: plf.Molecule) -> None:
    fp = plf.Fingerprint()
    with pytest.raises(RunRequiredError, match="Please run the fingerprint analysis"):
        fp_to_networkx(fp, ligand_mol)


def test_raises_kind(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    with pytest.raises(ValueError, match='must be "aggregate" or "frame"'):
        fp_to_networkx(fp, lig_mol, kind="foo")  # type: ignore[arg-type]


def test_raises_missing_frame(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    with pytest.raises(ValueError, match="Frame 42 not found"):
        fp_to_networkx(fp, lig_mol, kind="frame", frame=42)


def test_frame_graph(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    graph = fp_to_networkx(fp, lig_mol, kind="frame", frame=0, molsize=10)
    assert isinstance(graph, nx.MultiGraph)

    # ligand atoms and bonds
    ligand_nodes = nodes_of_type(graph, "ligand")
    assert len(ligand_nodes) == lig_mol.GetNumAtoms()
    bonds = [d for *_, d in graph.edges(data=True) if d["edge_type"] == "bond"]
    assert len(bonds) == lig_mol.GetNumBonds()
    xyz = lig_mol.GetConformer().GetPositions() * 10
    for atom in lig_mol.GetAtoms():
        node = graph.nodes[atom.GetIdx()]
        assert node["symbol"] == atom.GetSymbol()
        assert node["label"] == atom.GetSymbol()
        assert node["ligand"] == "LIG1.G"
        assert np.allclose([node["x"], node["y"], node["z"]], xyz[atom.GetIdx()])

    # protein residues
    expected_residues = {str(prot) for _, prot in fp.ifp[0]}
    protein_nodes = nodes_of_type(graph, "protein")
    assert set(protein_nodes) == expected_residues
    for res in protein_nodes:
        node = graph.nodes[res]
        assert node["residue"] == res
        assert node["label"] == res
        assert np.isfinite([node["x"], node["y"]]).all()

    # interactions
    edges = interaction_edges(graph)
    assert edges
    expected_interactions = {
        (str(prot), name) for (_, prot), data in fp.ifp[0].items() for name in data
    }
    assert {(v, d["interaction_type"]) for _, v, d in edges} == expected_interactions
    for u, v, d in edges:
        assert isinstance(u, int) and isinstance(v, str)
        assert d["components"] == "ligand_protein"
        assert d["weight"] == 1
        assert d["distance"] > 0
        assert d["weight_spring_layout"] == pytest.approx(10 / d["distance"])


def test_frame_display_all(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    default = interaction_edges(fp_to_networkx(fp, lig_mol, kind="frame"))
    all_edges = interaction_edges(
        fp_to_networkx(fp, lig_mol, kind="frame", display_all=True)
    )
    assert len(all_edges) >= len(default)


def test_aggregate_matches_lignetwork(
    fp_mol: tuple[plf.Fingerprint, plf.Molecule],
) -> None:
    """The aggregate graph should follow the same rules as the LigNetwork plot."""
    fp, lig_mol = fp_mol
    graph = fp_to_networkx(fp, lig_mol, kind="aggregate", threshold=0.3)
    df = LigNetwork._make_agg_df_from_fp(fp, threshold=0.3)
    expected = {}
    for idx, row in df.iterrows():
        _, prot, interaction, _ = cast("tuple[str, str, str, tuple[int, ...]]", idx)
        expected[prot, interaction] = row["weight"]
    edges = interaction_edges(graph)
    # same residue/interaction groups with the same frequency (the ligand atoms
    # kept for a group may differ when several sets of atoms are equally frequent)
    assert {(v, d["interaction_type"]) for _, v, d in edges} == set(expected)
    for u, v, d in edges:
        assert d["weight"] == pytest.approx(expected[v, d["interaction_type"]])
        assert 0 < d["weight"] <= 1
        assert any(
            u in metadata["parent_indices"]["ligand"]
            for ifp in fp.ifp.values()
            for (_, prot), data in ifp.items()
            if str(prot) == v
            for metadata in data.get(d["interaction_type"], ())
        )


def test_aggregate_threshold(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    no_filter = fp_to_networkx(fp, lig_mol, threshold=0)
    strict = fp_to_networkx(fp, lig_mol, threshold=1)
    groups_no_filter = {
        (v, d["interaction_type"]) for _, v, d in interaction_edges(no_filter)
    }
    groups_strict = {
        (v, d["interaction_type"]) for _, v, d in interaction_edges(strict)
    }
    assert groups_strict and groups_strict < groups_no_filter
    # the threshold applies to the frequency summed over the ligand atoms of a group
    for prot, interaction in groups_strict:
        total = sum(
            1
            for ifp in fp.ifp.values()
            for (_, p), data in ifp.items()
            if str(p) == prot and interaction in data
        )
        assert total == len(fp.ifp)
    # weights are frequencies, so nothing can pass a threshold above 1
    empty = fp_to_networkx(fp, lig_mol, threshold=1.5)
    assert not interaction_edges(empty)
    assert len(empty.nodes) == lig_mol.GetNumAtoms()


def test_no_interactions(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    """Graph with only the ligand when the fingerprint is empty"""
    _, lig_mol = fp_mol
    u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
    lig = u.select_atoms("resname LIG")
    prot = u.select_atoms("protein and resid 45:50")  # far from the ligand
    fp = plf.Fingerprint(["Hydrophobic"])
    fp.run(u.trajectory[0:1], lig, prot, progress=False, n_jobs=1)
    assert not fp.ifp[0]
    graph = fp_to_networkx(fp, lig_mol, kind="frame")
    assert set(graph.nodes) == set(range(lig_mol.GetNumAtoms()))
    assert not interaction_edges(graph)


@pytest.mark.parametrize("kind", ["frame", "aggregate"])
def test_water_bridge(
    water_fp_mol: tuple[plf.Fingerprint, plf.Molecule], kind: str
) -> None:
    fp, lig_mol = water_fp_mol
    graph = fp_to_networkx(fp, lig_mol, kind=kind)  # type: ignore[arg-type]
    water_nodes = nodes_of_type(graph, "water")
    assert set(water_nodes) == {"TIP383.X", "TIP317.X"}
    for water in water_nodes:
        assert graph.nodes[water]["water_id"] == water
        assert graph.nodes[water]["label"] == water
    components = {d["components"] for *_, d in interaction_edges(graph)}
    assert components == {
        "ligand_protein",
        "ligand_water",
        "water_water",
        "water_protein",
    }
    # the whole chain ligand -> TIP383 -> TIP317 -> THR399 must be present
    assert any(isinstance(n, int) for n in graph.neighbors("TIP383.X"))
    assert graph.has_edge("TIP383.X", "TIP317.X")
    assert graph.has_edge("TIP317.X", "THR399.X")
    assert graph.has_edge("TIP383.X", "TRP400.X")
    for *_, d in interaction_edges(graph):
        if d["components"] != "ligand_protein":
            assert d["interaction_type"] == "WaterBridge"
            assert d["weight"] == 1


def test_prune_dangling_waters() -> None:
    def rec(src: str, dest: str, components: str) -> dict[str, Any]:
        return {"ligand": src, "protein": dest, "components": components}

    chain = [
        rec("LIG1", "W1", "ligand_water"),
        rec("W1", "W2", "water_water"),
        rec("W2", "RES1", "water_protein"),
        rec("LIG1", "RES2", "ligand_protein"),
    ]
    dangling = [
        rec("LIG1", "W3", "ligand_water"),  # water only touching the ligand
        rec("W3", "W4", "water_water"),
        rec("W5", "RES3", "water_protein"),  # water only touching the protein
    ]
    assert _prune_dangling_waters(chain + dangling) == chain
    assert _prune_dangling_waters(dangling) == []


def test_coordinates(fp_mol: tuple[plf.Fingerprint, plf.Molecule]) -> None:
    fp, lig_mol = fp_mol
    graph = fp_to_networkx(fp, lig_mol, kind="aggregate")
    ligand_nodes = nodes_of_type(graph, "ligand")
    ligand_xy = np.array(
        [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in ligand_nodes]
    )
    others = nodes_of_type(graph, "protein")
    positions = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in others])
    assert np.isfinite(positions).all()
    # residues don't sit on top of each other or on top of ligand atoms
    for i, xy in enumerate(positions):
        assert (np.linalg.norm(positions[i + 1 :] - xy, axis=1) > 1).all()
        assert (np.linalg.norm(ligand_xy - xy, axis=1) > 1).all()
    # deterministic
    again = fp_to_networkx(fp, lig_mol, kind="aggregate")
    for node in others:
        assert graph.nodes[node]["x"] == again.nodes[node]["x"]
        assert graph.nodes[node]["y"] == again.nodes[node]["y"]


def test_spring_layout_weight() -> None:
    assert spring_layout_weight(2.5) == 4
    assert spring_layout_weight(0) == 1
