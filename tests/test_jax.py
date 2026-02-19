"""
Tests for the JAX-accelerated interaction fingerprinting module.

All tests are skipped if JAX is not installed. The key test is
``test_analyze_trajectory_matches_prolif``, which checks that
``analyze_trajectory()`` produces results matching ``Fingerprint.run()``
on the standard ProLIF test trajectory for all 9 interaction types.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

import prolif
from prolif.datafiles import TOP, TRAJ

jax = pytest.importorskip("jax", reason="JAX not installed")

from prolif.interactions._jax import JAX_AVAILABLE, analyze_trajectory  # noqa: E402

if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe


pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")

INTERACTIONS = [
    "Hydrophobic",
    "Cationic",
    "Anionic",
    "VdWContact",
    "HBAcceptor",
    "HBDonor",
    "PiStacking",
    "CationPi",
    "PiCation",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jax_result(u: "Universe"):
    """Run analyze_trajectory on the standard test trajectory."""
    return analyze_trajectory(
        u,
        ligand_selection="resname LIG",
        protein_selection="protein",
    )


@pytest.fixture(scope="module")
def prolif_fp(u: "Universe"):
    """Run Fingerprint.run on the standard test trajectory.

    """
    fp = prolif.Fingerprint(
        [
            "Hydrophobic",
            "Cationic",
            "Anionic",
            "VdWContact",
            "HBAcceptor",
            "HBDonor",
            "PiStacking",
            "CationPi",
            "PiCation",
        ]
    )
    fp.run(u.trajectory, u.select_atoms("resname LIG"), u.select_atoms("protein"),
           progress=False, n_jobs=1)
    return fp


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_jax_available():
    assert JAX_AVAILABLE


def test_import():
    from prolif.interactions._jax import analyze_trajectory  # noqa: F401
    from prolif.interactions._jax import has_interactions_frames  # noqa: F401


# ---------------------------------------------------------------------------
# Output shape and types
# ---------------------------------------------------------------------------


def test_result_shape(jax_result, u: "Universe"):
    F = len(u.trajectory)
    R = jax_result.n_residues
    assert jax_result.n_frames == F
    assert R > 0
    for name in INTERACTIONS:
        arr = jax_result.interactions[name]
        assert arr.shape == (F, R), f"{name}: expected ({F}, {R}), got {arr.shape}"
        assert arr.dtype == bool, f"{name}: expected bool dtype"


def test_result_has_all_interactions(jax_result):
    for name in INTERACTIONS:
        assert name in jax_result.interactions, f"Missing interaction: {name}"


def test_result_has_residue_ids(jax_result):
    assert len(jax_result.residue_ids) == jax_result.n_residues


def test_to_dataframe(jax_result, u: "Universe"):
    df = jax_result.to_dataframe()
    assert len(df) == len(u.trajectory)
    assert df.columns.names == ["residue", "interaction"]


# ---------------------------------------------------------------------------
# Accuracy: JAX must match ProLIF on all interactions
# ---------------------------------------------------------------------------


def _prolif_interaction_array(prolif_fp, interaction_name, residue_ids):
    """Extract (F, R) bool array from ProLIF Fingerprint matching jax residue_ids.

    ProLIF ifp keys are (lig_id, res_id) tuples; we match on the protein res_id.
    """
    F = len(prolif_fp.ifp)
    R = len(residue_ids)
    result = np.zeros((F, R), dtype=bool)
    rid_to_idx = {rid: i for i, rid in enumerate(residue_ids)}

    for f_i in range(F):
        for (_, res_id), interactions in prolif_fp.ifp[f_i].items():
            r_i = rid_to_idx.get(res_id)
            if r_i is not None and interaction_name in interactions:
                result[f_i, r_i] = True

    return result


@pytest.mark.parametrize("interaction_name", INTERACTIONS)
def test_analyze_trajectory_matches_prolif(jax_result, prolif_fp, interaction_name):
    """JAX and ProLIF must agree on every (frame, residue) pair."""
    jax_arr = jax_result.interactions[interaction_name]  # (F, R)

    prolif_arr = _prolif_interaction_array(
        prolif_fp, interaction_name, jax_result.residue_ids
    )

    total = jax_arr.size
    matches = int((jax_arr == prolif_arr).sum())
    accuracy = matches / total * 100

    assert accuracy == 100.0, (
        f"{interaction_name}: {accuracy:.1f}% accuracy "
        f"({total - matches} mismatches out of {total})"
    )


# ---------------------------------------------------------------------------
# residue_mode="all" captures at least as many residues as "first"
# ---------------------------------------------------------------------------


def test_residue_mode_all_ge_first(u: "Universe"):
    result_first = analyze_trajectory(
        u,
        ligand_selection="resname LIG",
        protein_selection="protein",
        residue_mode="first",
    )
    result_all = analyze_trajectory(
        u,
        ligand_selection="resname LIG",
        protein_selection="protein",
        residue_mode="all",
    )
    assert result_all.n_residues >= result_first.n_residues


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_pairwise_distances():
    import jax.numpy as jnp

    from prolif.interactions._jax.primitives import pairwise_distances

    coords1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords2 = jnp.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    d = pairwise_distances(coords1, coords2)
    assert d.shape == (2, 2)
    np.testing.assert_allclose(d[0, 0], 3.0, atol=1e-5)
    np.testing.assert_allclose(d[0, 1], 4.0, atol=1e-5)
    np.testing.assert_allclose(d[1, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(d[1, 1], 3.0, atol=1e-5)


def test_ring_normal_perpendicular():
    import jax.numpy as jnp

    from prolif.interactions._jax.primitives import ring_normal

    # Simple square ring in the XY plane — normal should be along Z
    coords = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    indices = jnp.array([0, 1, 2, 3])
    n = ring_normal(coords, indices)
    assert n.shape == (3,)
    # Normal must be perpendicular to the XY-plane vectors
    np.testing.assert_allclose(abs(float(n[2])), 1.0, atol=1e-5)


def test_angle_between_vectors():
    import jax.numpy as jnp

    from prolif.interactions._jax.primitives import angle_between_vectors

    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    angle = float(angle_between_vectors(v1, v2))
    np.testing.assert_allclose(angle, np.pi / 2, atol=1e-5)

    v3 = jnp.array([-1.0, 0.0, 0.0])
    angle2 = float(angle_between_vectors(v1, v3))
    np.testing.assert_allclose(angle2, np.pi, atol=1e-5)
