"""Frame-batched helpers for JAX interaction calculations."""

from __future__ import annotations

import math
import subprocess
from typing import Any

import jax
import jax.numpy as jnp

import prolif
from prolif.interactions import (
    Anionic,
    Cationic,
    CationPi,
    EdgeToFace,
    FaceToFace,
    HBAcceptor,
    HBDonor,
    Hydrophobic,
    MetalAcceptor,
    MetalDonor,
    PiStacking,
    VdWContact,
    XBAcceptor,
    XBDonor,
)

from .primitives import angle_at_vertex, angle_between_vectors, pairwise_distances

try:
    import pynvml  # type: ignore[import-untyped]
except ImportError:
    pynvml = None

try:
    from MDAnalysis.topology import guessers as mda_guessers
except ImportError:
    mda_guessers = None


def pairwise_distances_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
) -> jnp.ndarray:
    """Compute distances for each frame and residue in a trajectory.

    Args:
        lig_coords_f: (F, N, 3) ligand coordinates per frame.
        res_coords_f: (F, R, M, 3) residue coordinates per frame
            (padded to the same M across residues; masks handled upstream).

    Returns:
        (F, R, N, M) array of pairwise distances.

    Notes:
        - Shapes must be consistent across frames to avoid recompilation.
        - This function does not apply masks; callers should mask padded
          atoms as needed.
    """

    def _frame_distances(lig: jnp.ndarray, res_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda rc: pairwise_distances(lig, rc))(res_batch)

    return jax.vmap(_frame_distances)(lig_coords_f, res_coords_f)


def hbacceptor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    acc_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched hydrogen bond (acceptor) geometry.

    Args:
        lig_coords_f: (F, N, 3) ligand coords per frame.
        res_coords_f: (F, M, 3) residue coords per frame.
        acc_idx: (Na,) ligand acceptor indices.
        d_idx: (K,) residue donor indices.
        h_idx: (K,) residue hydrogen indices (paired with donors).
        distance_cutoff: Max A-D distance.
        dha_angle_min/max: D-H-A angle limits (deg).

    Returns:
        mask: (F, Na, K) boolean
        distances: (F, Na, K) A-D distances
        angles: (F, Na, K) D-H-A angles (deg)
    """
    acc = lig_coords_f[:, acc_idx, :]
    donors = res_coords_f[:, d_idx, :]
    hydrogens = res_coords_f[:, h_idx, :]

    dvec = acc[:, :, None, :] - donors[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dc = jnp.nextafter(jnp.asarray(distance_cutoff, dists.dtype), jnp.inf)
    dist_ok = dists <= dc

    ang = angle_at_vertex(
        donors[:, None, :, :],
        hydrogens[:, None, :, :],
        acc[:, :, None, :],
    )
    ang_deg = jnp.degrees(ang)
    min_deg = jnp.nextafter(jnp.asarray(dha_angle_min, ang_deg.dtype), -jnp.inf)
    max_deg = jnp.nextafter(jnp.asarray(dha_angle_max, ang_deg.dtype), jnp.inf)
    ang_ok = (ang_deg >= min_deg) & (ang_deg <= max_deg)

    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def hbdonor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    d_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    acc_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched hydrogen bond (donor) geometry (inverted).

    The distance is measured between donor and acceptor heavy atoms to mirror
    SingleAngle semantics after role inversion. The angle remains D-H-A in
    degrees. Returns (mask, distances, angles) with shapes (F, Nd, Ka).
    """
    donors = lig_coords_f[:, d_idx, :]
    hydrogens = lig_coords_f[:, h_idx, :]
    acc = res_coords_f[:, acc_idx, :]

    dvec = donors[:, :, None, :] - acc[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dc = jnp.nextafter(jnp.asarray(distance_cutoff, dists.dtype), jnp.inf)
    dist_ok = dists <= dc

    ang = angle_at_vertex(
        donors[:, :, None, :], hydrogens[:, :, None, :], acc[:, None, :, :]
    )
    ang_deg = jnp.degrees(ang)
    min_deg = jnp.nextafter(jnp.asarray(dha_angle_min, ang_deg.dtype), -jnp.inf)
    max_deg = jnp.nextafter(jnp.asarray(dha_angle_max, ang_deg.dtype), jnp.inf)
    ang_ok = (ang_deg >= min_deg) & (ang_deg <= max_deg)
    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def xbacceptor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    a_idx: jnp.ndarray,
    r_idx: jnp.ndarray,
    x_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched halogen bond (acceptor) geometry.

    Returns (mask, distances[A-X], axd_angles, xar_angles) with shapes (F, Na, K).
    """
    acc = lig_coords_f[:, a_idx, :]
    neigh = lig_coords_f[:, r_idx, :]
    hal = res_coords_f[:, x_idx, :]
    don = res_coords_f[:, d_idx, :]

    dvec = acc[:, :, None, :] - hal[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    axd = angle_at_vertex(
        acc[:, :, None, :],
        hal[:, None, :, :],
        don[:, None, :, :],
    )
    axd_deg = jnp.degrees(axd)
    axd_ok = (axd_deg >= axd_angle_min) & (axd_deg <= axd_angle_max)

    xar = angle_at_vertex(
        hal[:, None, :, :],
        acc[:, :, None, :],
        neigh[:, :, None, :],
    )
    xar_deg = jnp.degrees(xar)
    xar_ok = (xar_deg >= xar_angle_min) & (xar_deg <= xar_angle_max)

    mask = dist_ok & axd_ok & xar_ok
    return mask, dists, axd_deg, xar_deg


def xbdonor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    x_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    a_idx: jnp.ndarray,
    r_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched halogen bond (donor) geometry (inverted).

    Returns (mask, distances[X-A], axd_angles, xar_angles) with shapes (F, Nx, Ka).
    """
    hal = lig_coords_f[:, x_idx, :]
    don = lig_coords_f[:, d_idx, :]
    acc = res_coords_f[:, a_idx, :]
    neigh = res_coords_f[:, r_idx, :]

    dvec = hal[:, :, None, :] - acc[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    axd = angle_at_vertex(
        acc[:, None, :, :],
        hal[:, :, None, :],
        don[:, :, None, :],
    )
    axd_deg = jnp.degrees(axd)
    axd_ok = (axd_deg >= axd_angle_min) & (axd_deg <= axd_angle_max)

    xar = angle_at_vertex(
        hal[:, :, None, :],
        acc[:, None, :, :],
        neigh[:, None, :, :],
    )
    xar_deg = jnp.degrees(xar)
    xar_ok = (xar_deg >= xar_angle_min) & (xar_deg <= xar_angle_max)

    mask = dist_ok & axd_ok & xar_ok
    return mask, dists, axd_deg, xar_deg


def _ring_centroids_normals_frames(
    coords_f: jnp.ndarray,
    rings: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ring centroids and normals across frames for a list of rings.

    Args:
        coords_f: (F, N, 3) coordinates per frame
        rings: list of index arrays (variable ring sizes allowed)

    Returns:
        centroids: (F, K, 3)
        normals: (F, K, 3) unit normals
    """
    F = int(coords_f.shape[0])
    centroids_list = []
    normals_list = []
    for ring_idx in rings:
        rc = coords_f[:, ring_idx, :]
        centroid = rc.mean(axis=1)
        a = rc[:, 0, :] - centroid
        b = rc[:, 1, :] - centroid
        a_hat = a / jnp.clip(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
        b_hat = b / jnp.clip(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
        n = jnp.cross(a_hat, b_hat)
        n /= jnp.clip(jnp.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        centroids_list.append(centroid)
        normals_list.append(n)
    if centroids_list:
        centroids = jnp.stack(centroids_list, axis=1)
        normals = jnp.stack(normals_list, axis=1)
    else:
        centroids = jnp.zeros((F, 0, 3))
        normals = jnp.zeros((F, 0, 3))
    return centroids, normals


def cationpi_frames(
    ring_coords_f: jnp.ndarray,
    ring_list: list[jnp.ndarray],
    cation_coords_f: jnp.ndarray,
    cation_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 4.5,
    angle_min: float = 0.0,
    angle_max: float = 30.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched Cation-Pi geometry for one orientation (ring vs cation).

    Angles are folded into [0, 90] to account for ring-normal sign ambiguity.

    Returns:
        mask: (F, Kr, Kc)
        distances: (F, Kr, Kc) centroid-cation distances
        angles: (F, Kr, Kc) folded normal-centroid vector angles (deg)
    """
    F = int(ring_coords_f.shape[0])
    centroids, normals = _ring_centroids_normals_frames(ring_coords_f, ring_list)
    cations = (
        cation_coords_f[:, cation_idx, :] if cation_idx.size else jnp.zeros((F, 0, 3))
    )
    if centroids.shape[1] == 0 or cations.shape[1] == 0:
        z = jnp.zeros((F, centroids.shape[1], cations.shape[1]))
        return z.astype(bool), z, z

    vec = cations[:, None, :, :] - centroids[:, :, None, :]
    dists = jnp.linalg.norm(vec, axis=-1)
    dist_ok = dists <= distance_cutoff

    ang = angle_between_vectors(normals[:, :, None, :], vec)
    ang_deg = jnp.degrees(ang)
    ang_deg = jnp.minimum(ang_deg, 180.0 - ang_deg)
    ang_ok = (ang_deg >= angle_min) & (ang_deg <= angle_max)
    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def _compute_intersect_check(
    lc: jnp.ndarray,
    ln: jnp.ndarray,
    rc: jnp.ndarray,
    rn: jnp.ndarray,
    intersect_radius: float,
) -> jnp.ndarray:
    """Check whether ring-plane intersection geometry passes the radius criterion.

    Args:
        lc: (F, Kl, 3) ligand ring centroids.
        ln: (F, Kl, 3) ligand ring normals.
        rc: (F, Kr, 3) residue ring centroids.
        rn: (F, Kr, 3) residue ring normals.
        intersect_radius: Distance threshold in Angstroms.

    Returns:
        (F, Kl, Kr) boolean array, True where intersect_dist <= intersect_radius.
    """
    d = jnp.cross(ln[:, :, None, :], rn[:, None, :, :])
    d_norm = jnp.linalg.norm(d, axis=-1, keepdims=True)
    d_hat = d / jnp.where(d_norm > 1e-8, d_norm, jnp.ones_like(d_norm))

    shape = d.shape
    ln_exp = jnp.broadcast_to(ln[:, :, None, :], shape)
    rn_exp = jnp.broadcast_to(rn[:, None, :, :], shape)
    lc_exp = jnp.broadcast_to(lc[:, :, None, :], shape)
    rc_exp = jnp.broadcast_to(rc[:, None, :, :], shape)

    A = jnp.stack([ln_exp, rn_exp, d_hat], axis=-2)
    ln_dot_lc = jnp.sum(ln_exp * lc_exp, axis=-1)
    rn_dot_rc = jnp.sum(rn_exp * rc_exp, axis=-1)
    b = jnp.stack([ln_dot_lc, rn_dot_rc, jnp.zeros_like(ln_dot_lc)], axis=-1)

    det = jnp.linalg.det(A)
    valid = jnp.abs(det) > 1e-8

    safe_A = jnp.where(valid[..., None, None], A, jnp.eye(3))
    safe_b = jnp.where(valid[..., None], b, jnp.zeros_like(b))
    Q0 = jnp.linalg.solve(safe_A, safe_b[..., None]).squeeze(-1)

    vec = lc_exp - Q0
    scalar = jnp.sum(d_hat * vec, axis=-1, keepdims=True)
    P_lig = Q0 + d_hat * scalar

    dist_lig = jnp.linalg.norm(lc_exp - P_lig, axis=-1)
    dist_res = jnp.linalg.norm(rc_exp - P_lig, axis=-1)
    min_dist = jnp.minimum(dist_lig, dist_res)

    return valid & (min_dist <= intersect_radius)


def pistacking_frames(
    lig_coords_f: jnp.ndarray,
    lig_rings: list[jnp.ndarray],
    res_coords_f: jnp.ndarray,
    res_rings: list[jnp.ndarray],
    *,
    distance_cutoff: float = 6.5,
    plane_angle_min: float = 0.0,
    plane_angle_max: float = 30.0,
    ncc_angle_min: float = 0.0,
    ncc_angle_max: float = 60.0,
    intersect: bool = False,
    intersect_radius: float = 1.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched Pi-stacking geometry for ring-ring pairs.

    When ``intersect=True``, the ring-plane intersection distance criterion
    is applied with ``intersect_radius``.

    Returns:
        mask: (F, Kl, Kr)
        distances: (F, Kl, Kr) centroid-centroid distances
        plane_angles: (F, Kl, Kr) normals plane angle (deg)
        ncc_angles: (F, Kl, Kr) min of the two normal→centroid angles (deg)
    """
    F = int(lig_coords_f.shape[0])
    lc, ln = _ring_centroids_normals_frames(lig_coords_f, lig_rings)
    rc, rn = _ring_centroids_normals_frames(res_coords_f, res_rings)
    if lc.shape[1] == 0 or rc.shape[1] == 0:
        z = jnp.zeros((F, lc.shape[1], rc.shape[1]))
        return z.astype(bool), z, z, z

    cc_vec = rc[:, None, :, :] - lc[:, :, None, :]
    dists = jnp.linalg.norm(cc_vec, axis=-1)
    dist_ok = dists <= distance_cutoff

    pa = angle_between_vectors(ln[:, :, None, :], rn[:, None, :, :])
    pa_deg = jnp.degrees(pa)
    pa_deg = jnp.minimum(pa_deg, 180.0 - pa_deg)
    pa_ok = (pa_deg >= plane_angle_min) & (pa_deg <= plane_angle_max)

    n1 = angle_between_vectors(ln[:, :, None, :], cc_vec)
    n2 = angle_between_vectors(rn[:, None, :, :], -cc_vec)
    n1_deg = jnp.minimum(jnp.degrees(n1), 180.0 - jnp.degrees(n1))
    n2_deg = jnp.minimum(jnp.degrees(n2), 180.0 - jnp.degrees(n2))
    n1_ok = (n1_deg >= ncc_angle_min) & (n1_deg <= ncc_angle_max)
    n2_ok = (n2_deg >= ncc_angle_min) & (n2_deg <= ncc_angle_max)
    ncc_ok = n1_ok | n2_ok
    ncc_deg = jnp.minimum(n1_deg, n2_deg)

    mask = dist_ok & pa_ok & ncc_ok
    if intersect:
        mask &= _compute_intersect_check(lc, ln, rc, rn, intersect_radius)
    return mask, dists, pa_deg, ncc_deg


def build_actor_masks(
    lig_mol: Any,
    residues: list[Any],
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Compute boolean masks for distance-only actor atoms.

    Returns ligand and residue masks for Hydrophobic, Cationic, Anionic,
    MetalDonor, and MetalAcceptor SMARTS patterns. Residue masks are padded
    to a common length across residues.

    For inverted interactions (Anionic, MetalAcceptor), ligand and residue
    patterns are swapped before matching.
    """

    INVERTED = frozenset({"Anionic", "MetalAcceptor"})

    inters = {
        "Hydrophobic": Hydrophobic(),
        "Cationic": Cationic(),
        "Anionic": Anionic(),
        "MetalDonor": MetalDonor(),
        "MetalAcceptor": MetalAcceptor(),
    }

    N = lig_mol.GetNumAtoms()
    max_m = max((r.GetNumAtoms() for r in residues), default=0)

    lig_masks = {}
    res_masks = {}
    for name, inter in inters.items():
        inverted = name in INVERTED
        lig_pat = inter.prot_pattern if inverted else inter.lig_pattern
        res_pat = inter.lig_pattern if inverted else inter.prot_pattern

        lm = jnp.zeros((N,), dtype=bool)
        lmatches = lig_mol.GetSubstructMatches(lig_pat)
        if lmatches:
            idxs = [m[0] for m in lmatches]
            lm = lm.at[jnp.array(idxs)].set(True)
        lig_masks[name] = lm

        r_rows = []
        for r in residues:
            m = jnp.zeros((max_m,), dtype=bool)
            pmatches = r.GetSubstructMatches(res_pat)
            if pmatches:
                idxs = [mm[0] for mm in pmatches]
                mm = jnp.zeros((r.GetNumAtoms(),), dtype=bool)
                mm = mm.at[jnp.array(idxs)].set(True)
                m = m.at[: r.GetNumAtoms()].set(mm)
            r_rows.append(m)
        res_masks[name] = (
            jnp.stack(r_rows, axis=0) if r_rows else jnp.zeros((0, 0), dtype=bool)
        )

    return lig_masks, res_masks


def build_angle_indices_rdkit(
    lig_mol: Any,
    residues: list[Any],
) -> dict[str, dict[str, Any]]:
    """Build angle-index tables from RDKit molecules."""

    hb_acc = HBAcceptor()
    hb_don = HBDonor()
    xb_acc = XBAcceptor()
    xb_don = XBDonor()

    hb_acc_idx = []
    lmatches = lig_mol.GetSubstructMatches(hb_acc.lig_pattern)
    if lmatches:
        hb_acc_idx = [m[0] for m in lmatches]
    hb_acc_idx = (
        jnp.array(hb_acc_idx, dtype=int) if hb_acc_idx else jnp.zeros((0,), dtype=int)
    )

    hb_d_rows, hb_h_rows = [], []
    for r in residues:
        pmatches = r.GetSubstructMatches(hb_acc.prot_pattern)
        pairs = []
        for m in pmatches or []:
            if len(m) >= 2:
                pairs.append((m[0], m[1]))
        d = (
            jnp.array([p[0] for p in pairs], dtype=int)
            if pairs
            else jnp.zeros((0,), dtype=int)
        )
        h = (
            jnp.array([p[1] for p in pairs], dtype=int)
            if pairs
            else jnp.zeros((0,), dtype=int)
        )
        hb_d_rows.append(d)
        hb_h_rows.append(h)

    lmatches = lig_mol.GetSubstructMatches(xb_acc.lig_pattern)
    a = [m[0] for m in lmatches] if lmatches else []
    r = [m[1] for m in lmatches] if lmatches else []
    xbacc_a_idx = jnp.array(a, dtype=int) if a else jnp.zeros((0,), dtype=int)
    xbacc_r_idx = jnp.array(r, dtype=int) if r else jnp.zeros((0,), dtype=int)

    xb_x_rows, xb_d_rows = [], []
    for res in residues:
        pmatches = res.GetSubstructMatches(xb_acc.prot_pattern)
        pairs = []
        for m in pmatches or []:
            if len(m) >= 2:
                pairs.append((m[1], m[0]))
        x = (
            jnp.array([p[0] for p in pairs], dtype=int)
            if pairs
            else jnp.zeros((0,), dtype=int)
        )
        d = (
            jnp.array([p[1] for p in pairs], dtype=int)
            if pairs
            else jnp.zeros((0,), dtype=int)
        )
        xb_x_rows.append(x)
        xb_d_rows.append(d)

    lmatches = lig_mol.GetSubstructMatches(hb_don.prot_pattern)
    hb_lig_pairs = []
    for m in lmatches or []:
        if len(m) >= 2:
            hb_lig_pairs.append((m[0], m[1]))
    hb_lig_d_idx = (
        jnp.array([p[0] for p in hb_lig_pairs], dtype=int)
        if hb_lig_pairs
        else jnp.zeros((0,), dtype=int)
    )
    hb_lig_h_idx = (
        jnp.array([p[1] for p in hb_lig_pairs], dtype=int)
        if hb_lig_pairs
        else jnp.zeros((0,), dtype=int)
    )

    hb_res_acc_rows = []
    for res in residues:
        pmatches = res.GetSubstructMatches(hb_don.lig_pattern)
        acc = (
            jnp.array([m[0] for m in (pmatches or [])], dtype=int)
            if pmatches
            else jnp.zeros((0,), dtype=int)
        )
        hb_res_acc_rows.append(acc)

    lmatches = lig_mol.GetSubstructMatches(xb_don.lig_pattern)
    xbdon_pairs = []
    for m in lmatches or []:
        if len(m) >= 2:
            xbdon_pairs.append((m[1], m[0]))
    xbdon_lig_x_idx = (
        jnp.array([p[0] for p in xbdon_pairs], dtype=int)
        if xbdon_pairs
        else jnp.zeros((0,), dtype=int)
    )
    xbdon_lig_d_idx = (
        jnp.array([p[1] for p in xbdon_pairs], dtype=int)
        if xbdon_pairs
        else jnp.zeros((0,), dtype=int)
    )

    xbdon_res_a_rows, xbdon_res_r_rows = [], []
    for res in residues:
        pmatches = res.GetSubstructMatches(xb_don.lig_pattern)
        a_idx = (
            jnp.array([m[0] for m in (pmatches or [])], dtype=int)
            if pmatches
            else jnp.zeros((0,), dtype=int)
        )
        r_idx = (
            jnp.array([m[1] for m in (pmatches or [])], dtype=int)
            if pmatches
            else jnp.zeros((0,), dtype=int)
        )
        xbdon_res_a_rows.append(a_idx)
        xbdon_res_r_rows.append(r_idx)

    return {
        "hb": {
            "acc_idx": hb_acc_idx,
            "res_d_idx": hb_d_rows,
            "res_h_idx": hb_h_rows,
        },
        "hb_donor": {
            "lig_d_idx": hb_lig_d_idx,
            "lig_h_idx": hb_lig_h_idx,
            "res_a_idx": hb_res_acc_rows,
        },
        "xbacc": {
            "lig_a_idx": xbacc_a_idx,
            "lig_r_idx": xbacc_r_idx,
            "res_x_idx": xb_x_rows,
            "res_d_idx": xb_d_rows,
        },
        "xbdon": {
            "lig_x_idx": xbdon_lig_x_idx,
            "lig_d_idx": xbdon_lig_d_idx,
            "res_a_idx": xbdon_res_a_rows,
            "res_r_idx": xbdon_res_r_rows,
        },
    }


def build_angle_indices(
    lig_ag: Any,
    residue_ags: list[Any],
) -> dict[str, dict[str, Any]]:
    """Build angle-index tables from MDAnalysis AtomGroups.

    Returns the same key structure as :func:`build_angle_indices_rdkit`.
    """

    lig_mol = prolif.Molecule.from_mda(lig_ag)
    res_mols = [prolif.Molecule.from_mda(ag) for ag in residue_ags]

    hb_acc = HBAcceptor()
    hb_don = HBDonor()
    hb_acc_idx = []
    lmatches = lig_mol.GetSubstructMatches(hb_acc.lig_pattern)
    if lmatches:
        hb_acc_idx = [m[0] for m in lmatches]
    hb_acc_idx = (
        jnp.array(hb_acc_idx, dtype=int) if hb_acc_idx else jnp.zeros((0,), dtype=int)
    )

    hb_d_rows, hb_h_rows = [], []
    for r in res_mols:
        pmatches = r.GetSubstructMatches(hb_acc.prot_pattern)
        pairs = []
        for m in pmatches or []:
            if len(m) >= 2:
                pairs.append((m[0], m[1]))
        d = (
            jnp.array([p[0] for p in pairs], dtype=int)
            if pairs
            else jnp.zeros((0,), dtype=int)
        )
        h = (
            jnp.array([p[1] for p in pairs], dtype=int)
            if pairs
            else jnp.zeros((0,), dtype=int)
        )
        hb_d_rows.append(d)
        hb_h_rows.append(h)

    lmatches = lig_mol.GetSubstructMatches(hb_don.prot_pattern)
    hb_lig_pairs = []
    for m in lmatches or []:
        if len(m) >= 2:
            hb_lig_pairs.append((m[0], m[1]))
    hb_lig_d_idx = (
        jnp.array([p[0] for p in hb_lig_pairs], dtype=int)
        if hb_lig_pairs
        else jnp.zeros((0,), dtype=int)
    )
    hb_lig_h_idx = (
        jnp.array([p[1] for p in hb_lig_pairs], dtype=int)
        if hb_lig_pairs
        else jnp.zeros((0,), dtype=int)
    )

    hb_res_acc_rows = []
    for r in res_mols:
        pmatches = r.GetSubstructMatches(hb_don.lig_pattern)
        acc = (
            jnp.array([m[0] for m in (pmatches or [])], dtype=int)
            if pmatches
            else jnp.zeros((0,), dtype=int)
        )
        hb_res_acc_rows.append(acc)

    return {
        "hb": {
            "acc_idx": hb_acc_idx,
            "res_d_idx": hb_d_rows,
            "res_h_idx": hb_h_rows,
        },
        "hb_donor": {
            "lig_d_idx": hb_lig_d_idx,
            "lig_h_idx": hb_lig_h_idx,
            "res_a_idx": hb_res_acc_rows,
        },
        "xbacc": {
            "lig_a_idx": jnp.zeros((0,), dtype=int),
            "lig_r_idx": jnp.zeros((0,), dtype=int),
            "res_x_idx": [jnp.zeros((0,), dtype=int) for _ in res_mols],
            "res_d_idx": [jnp.zeros((0,), dtype=int) for _ in res_mols],
        },
        "xbdon": {
            "lig_x_idx": jnp.zeros((0,), dtype=int),
            "lig_d_idx": jnp.zeros((0,), dtype=int),
            "res_a_idx": [jnp.zeros((0,), dtype=int) for _ in res_mols],
            "res_r_idx": [jnp.zeros((0,), dtype=int) for _ in res_mols],
        },
    }


def build_ring_cation_indices(lig_mol: Any, residues: list[Any]) -> dict[str, Any]:
    """Build ring and cation index tables for ring-based interactions."""

    pi = PiStacking()
    ftf = FaceToFace()
    etf = EdgeToFace()
    ring_patterns = getattr(ftf, "pi_ring", []) or getattr(etf, "pi_ring", [])
    lig_rings = []
    for pat in ring_patterns:
        for m in lig_mol.GetSubstructMatches(pat):
            lig_rings.append(jnp.array(list(m), dtype=int))
    res_rings = []
    for r in residues:
        rr = []
        for pat in ring_patterns:
            for m in r.GetSubstructMatches(pat):
                rr.append(jnp.array(list(m), dtype=int))
        res_rings.append(rr)

    cp = CationPi()
    lmatches = lig_mol.GetSubstructMatches(cp.cation)
    lig_cations = (
        jnp.array([m[0] for m in lmatches], dtype=int)
        if lmatches
        else jnp.zeros((0,), dtype=int)
    )
    res_cations = []
    for r in residues:
        pm = r.GetSubstructMatches(cp.cation)
        rc = (
            jnp.array([m[0] for m in pm], dtype=int)
            if pm
            else jnp.zeros((0,), dtype=int)
        )
        res_cations.append(rc)

    return {
        "lig_rings": lig_rings,
        "res_rings": res_rings,
        "lig_cations": lig_cations,
        "res_cations": res_cations,
        "pi": pi,
        "cp": cp,
    }


def build_vdw_radii(
    lig_mol: Any,
    residues: list[Any],
    *,
    lig_ag: Any = None,
    residue_ags: list[Any] | None = None,
    use_real: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Build per-atom van der Waals radii arrays aligned with coordinate order.

    When ``use_real`` is True, radii are derived from MDAnalysis AtomGroups to
    match the coordinate ordering in real-frame mode. Otherwise, radii are
    derived from RDKit molecules to match duplicate-frame mode.

    Returns ligand radii (N,), residue radii (R, M) padded to max M, and the
    VdW tolerance used in contact checks.
    """

    def _symbol_from_atom(atom: Any) -> str:
        try:
            sym = atom.element
        except Exception:
            sym = None
        if not sym:
            if mda_guessers is not None:
                try:
                    sym = mda_guessers.guess_atom_element(getattr(atom, "name", ""))
                except Exception:
                    sym = "C"
            else:
                sym = "C"
        return str(sym).capitalize()

    vdw = VdWContact()

    if use_real:
        if lig_ag is None or residue_ags is None:
            raise ValueError(
                "lig_ag and residue_ags must be provided when use_real=True"
            )
        lig_elems = [_symbol_from_atom(a) for a in lig_ag.atoms]
        lig_radii = jnp.array(
            [vdw.vdwradii.get(e, 1.7) for e in lig_elems], dtype=float
        )

        max_m = max((ag.n_atoms for ag in residue_ags), default=0)
        res_rows = []
        for ag in residue_ags:
            elems = [_symbol_from_atom(a) for a in ag.atoms]
            row = jnp.array([vdw.vdwradii.get(e, 1.7) for e in elems], dtype=float)
            if ag.n_atoms < max_m:
                pad = jnp.zeros((max_m - ag.n_atoms,), dtype=float)
                row = jnp.concatenate([row, pad], axis=0)
            res_rows.append(row)
        res_radii = (
            jnp.stack(res_rows, axis=0) if res_rows else jnp.zeros((0, 0), dtype=float)
        )
    else:
        N = lig_mol.GetNumAtoms()
        lig_elems = [lig_mol.GetAtomWithIdx(i).GetSymbol() for i in range(N)]
        lig_radii = jnp.array(
            [vdw.vdwradii.get(e, 1.7) for e in lig_elems], dtype=float
        )

        max_m = max((r.GetNumAtoms() for r in residues), default=0)
        res_rows = []
        for r in residues:
            m = r.GetNumAtoms()
            elems = [r.GetAtomWithIdx(i).GetSymbol() for i in range(m)]
            row = jnp.array([vdw.vdwradii.get(e, 1.7) for e in elems], dtype=float)
            if m < max_m:
                pad = jnp.zeros((max_m - m,), dtype=float)
                row = jnp.concatenate([row, pad], axis=0)
            res_rows.append(row)
        res_radii = (
            jnp.stack(res_rows, axis=0) if res_rows else jnp.zeros((0, 0), dtype=float)
        )

    return lig_radii, res_radii, float(vdw.tolerance)


def has_interactions_frames(  # noqa: PLR0912
    lig_f: jnp.ndarray,
    res_f: jnp.ndarray,
    res_valid_mask: jnp.ndarray,
    lig_masks: dict,
    res_actor_masks: dict,
    angle_idx: dict,
    ring_idx: dict,
    vdw_radii: tuple,
    vicinity_cutoff: float = 6.0,
) -> dict[str, jnp.ndarray]:
    """Evaluate nine interactions across frames and residues, returning booleans.

    Returns a mapping name → (F, R) boolean arrays for:
    Hydrophobic, Cationic, Anionic, VdWContact, HBAcceptor, HBDonor,
    PiStacking, CationPi, PiCation.

    Args:
        vicinity_cutoff: Per-frame minimum atom-atom distance cutoff. Any
            (frame, residue) pair where the closest atom-atom distance exceeds
            this value is masked to False. Defaults to 6.0.
    """
    F = int(lig_f.shape[0])
    R = int(res_f.shape[1]) if res_f.ndim == 4 else 0
    results = {}

    d = pairwise_distances_frames(lig_f, res_f)

    m = (
        (d <= 4.5)
        & lig_masks["Hydrophobic"][None, None, :, None]
        & (res_actor_masks["Hydrophobic"] & res_valid_mask)[None, :, None, :]
    )
    results["Hydrophobic"] = (
        jnp.any(m, axis=(2, 3)) if R else jnp.zeros((F, 0), dtype=bool)
    )

    for k in ("Cationic", "Anionic"):
        m = (
            (d <= 4.5)
            & lig_masks[k][None, None, :, None]
            & (res_actor_masks[k] & res_valid_mask)[None, :, None, :]
        )
        results[k] = jnp.any(m, axis=(2, 3)) if R else jnp.zeros((F, 0), dtype=bool)

    lig_radii, res_radii, vdw_tol = vdw_radii
    radii_sum = lig_radii[None, None, :, None] + res_radii[None, :, None, :]
    m = (d <= (radii_sum + vdw_tol)) & res_valid_mask[None, :, None, :]
    results["VdWContact"] = (
        jnp.any(m, axis=(2, 3)) if R else jnp.zeros((F, 0), dtype=bool)
    )

    acc_idx = angle_idx["hb"]["acc_idx"]
    hb_acc_out = []
    for r_i in range(R):
        d_idx = angle_idx["hb"]["res_d_idx"][r_i]
        h_idx = angle_idx["hb"]["res_h_idx"][r_i]
        if acc_idx.size and d_idx.size:
            m_, _, _ = hbacceptor_frames(
                lig_f, res_f[:, r_i, :, :], acc_idx, d_idx, h_idx
            )
            hb_acc_out.append(jnp.any(m_, axis=(1, 2)))
        else:
            hb_acc_out.append(jnp.zeros((F,), dtype=bool))
    results["HBAcceptor"] = (
        jnp.stack(hb_acc_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)
    )

    hb_don_out = []
    lig_d = angle_idx["hb_donor"]["lig_d_idx"]
    lig_h = angle_idx["hb_donor"]["lig_h_idx"]
    for r_i in range(R):
        acc = angle_idx["hb_donor"]["res_a_idx"][r_i]
        if lig_d.size and acc.size:
            m_, _, _ = hbdonor_frames(lig_f, res_f[:, r_i, :, :], lig_d, lig_h, acc)
            hb_don_out.append(jnp.any(m_, axis=(1, 2)))
        else:
            hb_don_out.append(jnp.zeros((F,), dtype=bool))
    results["HBDonor"] = (
        jnp.stack(hb_don_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)
    )

    lig_rings = ring_idx["lig_rings"]
    res_rings = ring_idx["res_rings"]
    lig_cations = ring_idx["lig_cations"]
    res_cations = ring_idx["res_cations"]
    pi = ring_idx["pi"]
    cp = ring_idx["cp"]

    cationpi_out = []
    picat_out = []
    for r_i in range(R):
        has_cationpi = jnp.zeros((F,), dtype=bool)
        has_picat = jnp.zeros((F,), dtype=bool)
        if lig_cations.size and len(res_rings[r_i]):
            m_, _, _ = cationpi_frames(
                res_f[:, r_i, :, :],
                res_rings[r_i],
                lig_f,
                lig_cations,
                distance_cutoff=float(cp.distance),
                angle_min=math.degrees(float(cp.angle[0])),
                angle_max=math.degrees(float(cp.angle[1])),
            )
            has_cationpi = jnp.any(m_, axis=(1, 2))
        if len(lig_rings) and res_cations[r_i].size:
            m_, _, _ = cationpi_frames(
                lig_f,
                lig_rings,
                res_f[:, r_i, :, :],
                res_cations[r_i],
                distance_cutoff=float(cp.distance),
                angle_min=math.degrees(float(cp.angle[0])),
                angle_max=math.degrees(float(cp.angle[1])),
            )
            has_picat = jnp.any(m_, axis=(1, 2))
        cationpi_out.append(has_cationpi)
        picat_out.append(has_picat)
    results["CationPi"] = (
        jnp.stack(cationpi_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)
    )
    results["PiCation"] = (
        jnp.stack(picat_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)
    )

    ps_out = []
    for r_i in range(R):
        if len(lig_rings) and len(res_rings[r_i]):
            ftf = pi.ftf
            mF, _, _, _ = pistacking_frames(
                lig_f,
                lig_rings,
                res_f[:, r_i, :, :],
                res_rings[r_i],
                distance_cutoff=float(ftf.distance),
                plane_angle_min=math.degrees(float(ftf.plane_angle[0])),
                plane_angle_max=math.degrees(float(ftf.plane_angle[1])),
                ncc_angle_min=math.degrees(float(ftf.normal_to_centroid_angle[0])),
                ncc_angle_max=math.degrees(float(ftf.normal_to_centroid_angle[1])),
            )
            etf = pi.etf
            mE, _, _, _ = pistacking_frames(
                lig_f,
                lig_rings,
                res_f[:, r_i, :, :],
                res_rings[r_i],
                distance_cutoff=float(etf.distance),
                plane_angle_min=math.degrees(float(etf.plane_angle[0])),
                plane_angle_max=math.degrees(float(etf.plane_angle[1])),
                ncc_angle_min=math.degrees(float(etf.normal_to_centroid_angle[0])),
                ncc_angle_max=math.degrees(float(etf.normal_to_centroid_angle[1])),
                intersect=True,
                intersect_radius=float(etf.intersect_radius),
            )
            ps_out.append(jnp.any(mF | mE, axis=(1, 2)))
        else:
            ps_out.append(jnp.zeros((F,), dtype=bool))
    results["PiStacking"] = (
        jnp.stack(ps_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)
    )

    if R and vicinity_cutoff is not None:
        masked_d = jnp.where(res_valid_mask[None, :, None, :], d, jnp.inf)
        min_dist = masked_d.min(axis=(2, 3))
        vicinity_mask = min_dist <= vicinity_cutoff
        results = {k: v & vicinity_mask for k, v in results.items()}

    return results


def prepare_for_device(
    lig_f: jnp.ndarray,
    res_f: jnp.ndarray,
    res_valid_mask: jnp.ndarray,
    lig_masks: dict,
    res_actor_masks: dict,
    angle_idx: dict,
    ring_idx: dict,
    vdw_radii: tuple,
    device: str = "cpu",
) -> tuple:
    """Move frame-batched, small metadata arrays to a target device.

    Args:
        lig_f: (F, N, 3) ligand coordinates per frame.
        res_f: (F, R, M, 3) residue coordinates per frame.
        res_valid_mask: (R, M) boolean mask for valid atoms.
        lig_masks: Dict of ligand SMARTS masks per interaction.
        res_actor_masks: Dict of residue SMARTS masks per interaction.
        angle_idx: Dict of angle indices from build_angle_indices.
        ring_idx: Dict of ring indices from build_ring_cation_indices.
        vdw_radii: Tuple of (lig_radii, res_radii, tolerance).
        device: 'cpu' or 'gpu'. Coordinates remain on host; chunks are moved
            transiently during computation. Reused small structures (masks,
            indices, radii) are placed on the device.

    Returns:
        Tuple of all inputs moved to the specified device.
        Returns unchanged inputs if device='cpu' or no GPU available.
    """
    if device != "gpu":
        return (
            lig_f,
            res_f,
            res_valid_mask,
            lig_masks,
            res_actor_masks,
            angle_idx,
            ring_idx,
            vdw_radii,
        )

    gpus = jax.devices("gpu")
    if not gpus:
        return (
            lig_f,
            res_f,
            res_valid_mask,
            lig_masks,
            res_actor_masks,
            angle_idx,
            ring_idx,
            vdw_radii,
        )

    dev = gpus[0]

    res_valid_mask = jax.device_put(res_valid_mask, device=dev)
    lig_masks = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, device=dev), lig_masks
    )
    res_actor_masks = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, device=dev), res_actor_masks
    )

    angle_idx = {
        "hb": {
            "acc_idx": jax.device_put(angle_idx["hb"]["acc_idx"], device=dev),
            "res_d_idx": [
                jax.device_put(a, device=dev) for a in angle_idx["hb"]["res_d_idx"]
            ],
            "res_h_idx": [
                jax.device_put(a, device=dev) for a in angle_idx["hb"]["res_h_idx"]
            ],
        },
        "hb_donor": {
            "lig_d_idx": jax.device_put(angle_idx["hb_donor"]["lig_d_idx"], device=dev),
            "lig_h_idx": jax.device_put(angle_idx["hb_donor"]["lig_h_idx"], device=dev),
            "res_a_idx": [
                jax.device_put(a, device=dev)
                for a in angle_idx["hb_donor"]["res_a_idx"]
            ],
        },
        "xbacc": {
            "lig_a_idx": jax.device_put(angle_idx["xbacc"]["lig_a_idx"], device=dev),
            "lig_r_idx": jax.device_put(angle_idx["xbacc"]["lig_r_idx"], device=dev),
            "res_x_idx": [
                jax.device_put(a, device=dev) for a in angle_idx["xbacc"]["res_x_idx"]
            ],
            "res_d_idx": [
                jax.device_put(a, device=dev) for a in angle_idx["xbacc"]["res_d_idx"]
            ],
        },
        "xbdon": {
            "lig_x_idx": jax.device_put(angle_idx["xbdon"]["lig_x_idx"], device=dev),
            "lig_d_idx": jax.device_put(angle_idx["xbdon"]["lig_d_idx"], device=dev),
            "res_a_idx": [
                jax.device_put(a, device=dev) for a in angle_idx["xbdon"]["res_a_idx"]
            ],
            "res_r_idx": [
                jax.device_put(a, device=dev) for a in angle_idx["xbdon"]["res_r_idx"]
            ],
        },
    }

    ring_idx = {
        "lig_rings": [jax.device_put(a, device=dev) for a in ring_idx["lig_rings"]],
        "res_rings": [
            [jax.device_put(a, device=dev) for a in rr] for rr in ring_idx["res_rings"]
        ],
        "lig_cations": jax.device_put(ring_idx["lig_cations"], device=dev),
        "res_cations": [jax.device_put(a, device=dev) for a in ring_idx["res_cations"]],
        "pi": ring_idx["pi"],
        "cp": ring_idx["cp"],
    }

    lr, rr, vdw_tol = vdw_radii
    vdw_radii = (
        jax.device_put(lr, device=dev),
        jax.device_put(rr, device=dev),
        vdw_tol,
    )

    return (
        lig_f,
        res_f,
        res_valid_mask,
        lig_masks,
        res_actor_masks,
        angle_idx,
        ring_idx,
        vdw_radii,
    )


def get_gpu_device() -> Any | None:
    """Return the first GPU device if available, else None.

    Safe on CPU-only installs where requesting 'gpu' would raise.
    """
    try:
        gpus = jax.devices("gpu")
    except Exception:
        return None
    return gpus[0] if gpus else None


def get_gpu_memory_info() -> tuple[float, float] | None:
    """Return (free_mb, total_mb) for GPU 0 if available, else None.

    Tries NVML first, then falls back to parsing `nvidia-smi` output.
    """
    dev = get_gpu_device()
    if dev is None:
        return None

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = info.free / (1024 * 1024)
            total_mb = info.total / (1024 * 1024)
            return float(free_mb), float(total_mb)
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        line = out.strip().splitlines()[0]
        cols = [c.strip() for c in line.split(",")]
        free_mb = float(cols[0])
        total_mb = float(cols[1])
        return free_mb, total_mb
    except Exception:
        return None


def estimate_memory_per_frame(
    n_ligand_atoms: int,
    n_residues: int,
    max_residue_atoms: int,
) -> float:
    """Estimate GPU memory usage per frame in bytes.

    This is a rough estimate accounting for:
        - Distance matrices: R x N x M x 4 bytes
        - Intermediate arrays and masks
        - JAX overhead multiplier (~3x for JIT buffers)

    Args:
        n_ligand_atoms: Number of ligand atoms (N).
        n_residues: Number of residues (R).
        max_residue_atoms: Max atoms per residue (M).

    Returns:
        Estimated bytes per frame.
    """
    N, R, M = n_ligand_atoms, n_residues, max_residue_atoms

    coords_lig = N * 3 * 4
    coords_res = R * M * 3 * 4
    distance_matrix = R * N * M * 4
    masks_and_results = R * N * M * 4

    base = coords_lig + coords_res + distance_matrix + masks_and_results

    overhead_multiplier = 3.0

    return base * overhead_multiplier


def calculate_chunk_size(
    n_ligand_atoms: int,
    n_residues: int,
    max_residue_atoms: int,
    available_memory_mb: float | None = None,
    memory_fraction: float = 0.7,
) -> int:
    """Calculate safe chunk size (frames per batch) for GPU.

    Args:
        n_ligand_atoms: Number of ligand atoms.
        n_residues: Number of residues.
        max_residue_atoms: Max atoms per residue.
        available_memory_mb: GPU memory in MB. If None, auto-detect.
        memory_fraction: Fraction of memory to use (default 0.7 for safety).

    Returns:
        Recommended number of frames per chunk.
    """
    if available_memory_mb is None:
        mem = get_gpu_memory_info()
        if mem is None:
            return 1000
        available_memory_mb = mem[0]

    bytes_per_frame = estimate_memory_per_frame(
        n_ligand_atoms, n_residues, max_residue_atoms
    )

    usable_bytes = available_memory_mb * 1024 * 1024 * memory_fraction
    chunk_size = int(usable_bytes / bytes_per_frame)

    return max(1, min(chunk_size, 10000))


def chunked_has_interactions_frames(
    lig_f: jnp.ndarray,
    res_f: jnp.ndarray,
    res_valid_mask: jnp.ndarray,
    lig_masks: dict,
    res_actor_masks: dict,
    angle_idx: dict,
    ring_idx: dict,
    vdw_radii: tuple,
    chunk_size: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Evaluate interactions in memory-safe chunks.

    Automatically chunks large trajectories to avoid GPU OOM.
    Results are concatenated along the frame dimension.

    Args:
        lig_f: (F, N, 3) ligand coordinates.
        res_f: (F, R, M, 3) residue coordinates.
        res_valid_mask: (R, M) valid atom mask.
        lig_masks: SMARTS masks per interaction.
        res_actor_masks: Residue SMARTS masks.
        angle_idx: Angle indices dict.
        ring_idx: Ring indices dict.
        vdw_radii: VdW radii tuple.
        chunk_size: Frames per chunk. If None, auto-calculate.

    Returns:
        Dict of interaction name → (F, R) boolean arrays.
    """
    F = int(lig_f.shape[0])
    N = int(lig_f.shape[1])
    R = int(res_f.shape[1])
    M = int(res_f.shape[2])

    if chunk_size is None:
        chunk_size = calculate_chunk_size(N, R, M)

    dev = get_gpu_device()

    all_results: list[dict[str, jnp.ndarray]] = []
    for i in range(0, F, chunk_size):
        j = min(i + chunk_size, F)

        if dev is not None:
            chunk_lig = jax.device_put(lig_f[i:j], device=dev)
            chunk_res = jax.device_put(res_f[i:j], device=dev)
        else:
            chunk_lig = lig_f[i:j]
            chunk_res = res_f[i:j]

        chunk_result = has_interactions_frames(
            chunk_lig,
            chunk_res,
            res_valid_mask,
            lig_masks,
            res_actor_masks,
            angle_idx,
            ring_idx,
            vdw_radii,
        )

        chunk_result_host = jax.device_get(chunk_result)
        all_results.append({k: jnp.asarray(v) for k, v in chunk_result_host.items()})

    if len(all_results) == 1:
        return all_results[0]

    combined = {}
    for key in all_results[0]:
        combined[key] = jnp.concatenate([r[key] for r in all_results], axis=0)

    return combined
