"""
JAX-accelerated geometric primitives for molecular interaction calculations.

These functions provide GPU-compatible, vectorized implementations of
geometric operations used in interaction fingerprinting. All functions
are designed to work with batched inputs and can be JIT-compiled.
"""

from typing import cast

import jax.numpy as jnp
from jax.ops import segment_sum


def pairwise_distances(coords1: jnp.ndarray, coords2: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise Euclidean distances between two sets of points.

    Args:
        coords1: (N, 3) array of 3D points.
        coords2: (M, 3) array of 3D points.

    Returns:
        (N, M) distance matrix where entry [i,j] is the distance
        between coords1[i] and coords2[j].
    """
    diff = coords1[:, None, :] - coords2[None, :, :]
    return cast(jnp.ndarray, jnp.linalg.norm(diff, axis=-1))


def batch_centroids(
    coords: jnp.ndarray, group_indices: list[jnp.ndarray]
) -> jnp.ndarray:
    """Compute centroids for multiple groups of atoms.

    Uses segment_sum for efficient computation over variable-size groups.

    Args:
        coords: (N, 3) array of all atom coordinates.
        group_indices: List of K arrays, each containing atom indices for one group.

    Returns:
        (K, 3) array of centroid positions.
    """
    K = len(group_indices)
    flat_indices = jnp.concatenate(group_indices)
    segment_ids = jnp.repeat(jnp.arange(K), jnp.array([len(g) for g in group_indices]))

    points = coords[flat_indices]
    sums = segment_sum(points, segment_ids, num_segments=K)
    counts = segment_sum(jnp.ones(len(flat_indices)), segment_ids, num_segments=K)

    return sums / counts[:, None]


def batch_centroids_masked(
    coords: jnp.ndarray,
    index_padded: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute centroids for groups using padded indices and masks.

    Args:
        coords: (N, 3) array of all atom coordinates.
        index_padded: (K, S) int array of atom indices per group.
        mask: (K, S) bool array where True marks a valid atom in the group.

    Returns:
        (K, 3) array of centroid positions.
    """
    gathered = coords[index_padded]
    weights = mask[..., None]
    weighted = gathered * weights
    sums = jnp.sum(weighted, axis=1)
    counts = jnp.sum(mask, axis=1, keepdims=True)
    return sums / counts


def batch_ring_normals_masked(
    coords: jnp.ndarray,
    index_padded: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute unit normals for rings using the same method as ProLIF's
    ``get_ring_normal_vector``: cross product of unit vectors from the centroid
    to the first two ring atoms.

    Args:
        coords: (N, 3) array of all atom coordinates.
        index_padded: (K, S) int array of atom indices per ring.
        mask: (K, S) bool array where True marks a valid atom in the ring.

    Returns:
        (K, 3) array of unit normal vectors.
    """
    pts = coords[index_padded]
    centroid = (pts * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    a = pts[:, 0, :] - centroid
    b = pts[:, 1, :] - centroid
    a_hat = a / jnp.clip(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b_hat = b / jnp.clip(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    normals = jnp.cross(a_hat, b_hat)
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    return cast(
        jnp.ndarray,
        jnp.where(norms > 0, normals / norms, jnp.array([0.0, 0.0, 1.0])[None, :]),
    )


def ring_normal(coords: jnp.ndarray, ring_indices: jnp.ndarray) -> jnp.ndarray:
    """Compute unit normal vector to a ring plane using the same method as
    ProLIF's ``get_ring_normal_vector``: cross product of unit vectors from the
    centroid to the first two ring atoms.

    Args:
        coords: (N, 3) array of all atom coordinates.
        ring_indices: (R,) array of atom indices forming the ring (R >= 3).

    Returns:
        (3,) unit normal vector perpendicular to the ring plane.
    """
    pts = coords[ring_indices]
    centroid = pts.mean(axis=0)
    a = pts[0] - centroid
    b = pts[1] - centroid
    a_hat = a / jnp.clip(jnp.linalg.norm(a), 1e-8)
    b_hat = b / jnp.clip(jnp.linalg.norm(b), 1e-8)
    normal = jnp.cross(a_hat, b_hat)
    norm = jnp.linalg.norm(normal)
    return cast(
        jnp.ndarray,
        jnp.where(norm > 0, normal / norm, jnp.array([0.0, 0.0, 1.0])),
    )


def batch_ring_normals(
    coords: jnp.ndarray, ring_indices_list: list[jnp.ndarray]
) -> jnp.ndarray:
    """Compute unit normal vectors for multiple rings using Newell's method."""
    normals = [ring_normal(coords, idxs) for idxs in ring_indices_list]
    return jnp.stack(normals, axis=0)


def angle_between_vectors(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Compute angle between vectors.

    Handles arbitrary batch dimensions via axis=-1 operations.

    Args:
        v1: (..., 3) array of vectors.
        v2: (..., 3) array of vectors with same batch shape as v1.

    Returns:
        (...,) array of angles in radians, range [0, pi].
    """
    dot = jnp.sum(v1 * v2, axis=-1)
    norm1 = jnp.linalg.norm(v1, axis=-1)
    norm2 = jnp.linalg.norm(v2, axis=-1)
    cos_angle = dot / (norm1 * norm2)
    return jnp.arccos(jnp.clip(cos_angle, -1, 1))


def angle_at_vertex(
    p1: jnp.ndarray, vertex: jnp.ndarray, p2: jnp.ndarray
) -> jnp.ndarray:
    """Compute angle formed at a vertex point.

    Calculates the angle p1-vertex-p2, useful for angles like D-H-A
    in hydrogen bonds.

    Args:
        p1: (..., 3) first point.
        vertex: (..., 3) vertex point where angle is measured.
        p2: (..., 3) second point.

    Returns:
        (...,) angles in radians.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    return angle_between_vectors(v1, v2)


def point_to_plane_distance(
    points: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray
) -> jnp.ndarray:
    """Compute signed distances from points to a plane.

    Useful for cation-pi and pi-stacking geometric checks.

    Args:
        points: (N, 3) array of points.
        plane_point: (3,) a point on the plane (e.g., ring centroid).
        plane_normal: (3,) unit normal vector to the plane.

    Returns:
        (N,) signed distances. Positive values indicate the point is on
        the same side as the normal vector.
    """
    return jnp.dot(points - plane_point, plane_normal)
