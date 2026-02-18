"""
JAX-accelerated interaction detection integrated with ProLIF SMARTS matching.

This module provides JAX-accelerated versions of ProLIF interactions that:
1. Use ProLIF/RDKit for SMARTS pattern matching (chemistry)
2. Use JAX for batched geometric calculations (distance, angles)

"""

from itertools import product
from math import radians, degrees
from typing import Iterator

import jax.numpy as jnp
from rdkit.Geometry import Point3D

from prolif.interactions.base import Distance, SingleAngle, DoubleAngle, BasePiStacking
from prolif.utils import angle_between_limits, get_centroid, get_ring_normal_vector
from .primitives import pairwise_distances, angle_at_vertex, angle_between_vectors


def compute_distances_batch(
    lig_coords,
    prot_coords,
) -> jnp.ndarray:
    """Compute pairwise distances between two atom sets using JAX.

    Args:
        lig_coords: (N, 3) coordinates array (array-like); converted to `jnp.ndarray`.
        prot_coords: (M, 3) coordinates array (array-like); converted to `jnp.ndarray`.

    Returns:
        (N, M) `jnp.ndarray` of Euclidean distances.
    """
    lig_jax = jnp.asarray(lig_coords)
    prot_jax = jnp.asarray(prot_coords)
    return pairwise_distances(lig_jax, prot_jax)


class JAXDistanceMixin:
    """Mixin to accelerate Distance-based interactions with JAX.

    This mixin overrides the detect method to batch distance calculations
    while still using ProLIF's SMARTS matching.
    """

    def detect(self, lig_res, prot_res) -> Iterator:
        """Detect interactions using JAX-accelerated distance calculation."""
        lig_matches = lig_res.GetSubstructMatches(self.lig_pattern)
        prot_matches = prot_res.GetSubstructMatches(self.prot_pattern)

        if not (lig_matches and prot_matches):
            return

        lig_indices = [m[0] for m in lig_matches]
        prot_indices = [m[0] for m in prot_matches]

        lig_coords = lig_res.xyz[lig_indices]
        prot_coords = prot_res.xyz[prot_indices]

        distances = compute_distances_batch(lig_coords, prot_coords)

        within_threshold = distances <= self.distance

        for i, lig_match in enumerate(lig_matches):
            for j, prot_match in enumerate(prot_matches):
                if within_threshold[i, j]:
                    yield self.metadata(
                        lig_res,
                        prot_res,
                        lig_match,
                        prot_match,
                        distance=float(distances[i, j]),
                    )


def detect_distance_batch(
    interaction,
    lig_res,
    residues: list,
) -> list[list[dict]]:
    """Batch detect Distance interactions across multiple residues.

    This is the main entry point for JAX-accelerated interaction detection.

    Args:
        interaction: A Distance-based interaction instance (Hydrophobic, Cationic, etc.)
        lig_res: Ligand residue
        residues: List of protein residues to check

    Returns:
        List of lists, one per residue, containing interaction metadata dicts.
    """
    lig_matches = lig_res.GetSubstructMatches(interaction.lig_pattern)
    if not lig_matches:
        return [[] for _ in residues]

    lig_indices = [m[0] for m in lig_matches]
    lig_coords = lig_res.xyz[lig_indices]

    results = []

    all_prot_data = []
    for res in residues:
        prot_matches = res.GetSubstructMatches(interaction.prot_pattern)
        if prot_matches:
            prot_indices = [m[0] for m in prot_matches]
            prot_coords = res.xyz[prot_indices]
            all_prot_data.append((prot_matches, prot_coords))
        else:
            all_prot_data.append(([], None))

    for idx, (prot_matches, prot_coords) in enumerate(all_prot_data):
        res_results = []

        if prot_matches:
            distances = compute_distances_batch(lig_coords, prot_coords)
            within_threshold = distances <= interaction.distance

            for i, lig_match in enumerate(lig_matches):
                for j, prot_match in enumerate(prot_matches):
                    if within_threshold[i, j]:
                        res_results.append(
                            interaction.metadata(
                                lig_res,
                                residues[idx],
                                lig_match,
                                prot_match,
                                distance=float(distances[i, j]),
                            )
                        )

        results.append(res_results)

    return results


def _is_inverted_interaction(interaction) -> bool:
    """Check if an interaction was created with invert_role."""
    name = type(interaction).__name__
    return name in ('HBDonor', 'XBDonor', 'Anionic', 'PiCation', 'MetalAcceptor')


def _get_interaction_type(interaction) -> str:
    """Determine the base type of an interaction."""
    cls = type(interaction)
    name = cls.__name__

    if name == 'VdWContact':
        return 'vdwcontact'
    if name == 'PiStacking':
        return 'pistacking_composite'
    if name in ('FaceToFace', 'EdgeToFace'):
        return 'pistacking'
    if name in ('CationPi', 'PiCation'):
        return 'cationpi'

    if isinstance(interaction, BasePiStacking):
        return 'pistacking'
    if hasattr(interaction, 'pi_ring') and hasattr(interaction, 'cation'):
        return 'cationpi'
    if isinstance(interaction, DoubleAngle):
        return 'doubleangle'
    if isinstance(interaction, SingleAngle):
        return 'singleangle'
    if isinstance(interaction, Distance):
        return 'distance'

    return 'distance'


def _has_distance_interaction(interaction, lig_res, residues, is_inverted) -> list[bool]:
    """Check Distance-based interactions."""
    if is_inverted:
        lig_pattern = interaction.prot_pattern
        prot_pattern = interaction.lig_pattern
    else:
        lig_pattern = interaction.lig_pattern
        prot_pattern = interaction.prot_pattern

    lig_matches = lig_res.GetSubstructMatches(lig_pattern)
    if not lig_matches:
        return [False] * len(residues)

    lig_indices = [m[0] for m in lig_matches]
    lig_coords = lig_res.xyz[lig_indices]

    results = []
    for res in residues:
        prot_matches = res.GetSubstructMatches(prot_pattern)
        if not prot_matches:
            results.append(False)
            continue

        prot_indices = [m[0] for m in prot_matches]
        prot_coords = res.xyz[prot_indices]
        distances = compute_distances_batch(lig_coords, prot_coords)
        results.append(bool((distances <= interaction.distance).any()))

    return results


def _has_singleangle_interaction(interaction, lig_res, residues, is_inverted) -> list[bool]:
    """Check SingleAngle-based interactions (HBAcceptor, HBDonor).

    For SingleAngle:
    - lig_pattern matches L1 (single atom)
    - prot_pattern matches P1-P2 (two atoms for angle)
    - Distance is L1 to P1 or P2
    - Angle is P1-P2...L1

    For inverted (HBDonor):
    - Roles are swapped: ligand has P1-P2, protein has L1
    """
    results = []
    for res in residues:
        if is_inverted:
            prot_matches = lig_res.GetSubstructMatches(interaction.prot_pattern)
            lig_matches = res.GetSubstructMatches(interaction.lig_pattern)
            prot_xyz, lig_xyz = lig_res.xyz, res.xyz
        else:
            lig_matches = lig_res.GetSubstructMatches(interaction.lig_pattern)
            prot_matches = res.GetSubstructMatches(interaction.prot_pattern)
            lig_xyz, prot_xyz = lig_res.xyz, res.xyz

        if not (lig_matches and prot_matches):
            results.append(False)
            continue

        found = False
        for lig_match, prot_match in product(lig_matches, prot_matches):
            l1 = Point3D(*lig_xyz[lig_match[0]])
            p1 = Point3D(*prot_xyz[prot_match[0]])
            p2 = Point3D(*prot_xyz[prot_match[1]])
            dist = interaction._measure_distance(l1, p1, p2)
            if dist <= interaction.distance:
                p2p1 = p2.DirectionVector(p1)
                p2l1 = p2.DirectionVector(l1)
                angle = p2p1.AngleTo(p2l1)
                if angle_between_limits(angle, *interaction.angle):
                    found = True
                    break
        results.append(found)

    return results


def _has_doubleangle_interaction(interaction, lig_res, residues, is_inverted) -> list[bool]:
    """Check DoubleAngle-based interactions (XBAcceptor, XBDonor).

    For DoubleAngle:
    - lig_pattern matches L1-L2 (two atoms)
    - prot_pattern matches P1-P2 (two atoms)

    For inverted (XBDonor):
    - Roles are swapped
    """
    results = []
    for res in residues:
        if is_inverted:
            prot_matches = lig_res.GetSubstructMatches(interaction.prot_pattern)
            lig_matches = res.GetSubstructMatches(interaction.lig_pattern)
            prot_xyz, lig_xyz = lig_res.xyz, res.xyz
        else:
            lig_matches = lig_res.GetSubstructMatches(interaction.lig_pattern)
            prot_matches = res.GetSubstructMatches(interaction.prot_pattern)
            lig_xyz, prot_xyz = lig_res.xyz, res.xyz

        if not (lig_matches and prot_matches):
            results.append(False)
            continue

        found = False
        for lig_match, prot_match in product(lig_matches, prot_matches):
            l1 = Point3D(*lig_xyz[lig_match[0]])
            l2 = Point3D(*lig_xyz[lig_match[1]])
            p1 = Point3D(*prot_xyz[prot_match[0]])
            p2 = Point3D(*prot_xyz[prot_match[1]])
            dist = interaction._measure_distance(l1, l2, p1, p2)
            if dist <= interaction.distance:
                p2p1 = p2.DirectionVector(p1)
                p2l1 = p2.DirectionVector(l1)
                l1p2p1 = p2p1.AngleTo(p2l1)
                if angle_between_limits(l1p2p1, *interaction.L1P2P1_angle):
                    l1p2 = l1.DirectionVector(p2)
                    l1l2 = l1.DirectionVector(l2)
                    l2l1p2 = l1p2.AngleTo(l1l2)
                    if angle_between_limits(l2l1p2, *interaction.L2L1P2_angle):
                        found = True
                        break
        results.append(found)

    return results


def _has_cationpi_interaction(interaction, lig_res, residues, is_inverted) -> list[bool]:
    """Check CationPi/PiCation interactions."""
    results = []
    for res in residues:
        if is_inverted:
            ring_matches = []
            for pi_ring in interaction.pi_ring:
                ring_matches.extend(lig_res.GetSubstructMatches(pi_ring))
            cation_matches = res.GetSubstructMatches(interaction.cation)
            ring_xyz, cation_xyz = lig_res.xyz, res.xyz
        else:
            cation_matches = lig_res.GetSubstructMatches(interaction.cation)
            ring_matches = []
            for pi_ring in interaction.pi_ring:
                ring_matches.extend(res.GetSubstructMatches(pi_ring))
            ring_xyz, cation_xyz = res.xyz, lig_res.xyz

        if not (ring_matches and cation_matches):
            results.append(False)
            continue

        found = False
        for ring_match, cation_match in product(ring_matches, cation_matches):
            ring_coords = ring_xyz[list(ring_match)]
            centroid = Point3D(*get_centroid(ring_coords))
            cation = Point3D(*cation_xyz[cation_match[0]])
            dist = centroid.Distance(cation)
            if dist <= interaction.distance:
                normal = get_ring_normal_vector(centroid, ring_coords)
                centroid_cation = centroid.DirectionVector(cation)
                angle = normal.AngleTo(centroid_cation)
                if angle_between_limits(angle, *interaction.angle, ring=True):
                    found = True
                    break
        results.append(found)

    return results


def _has_pistacking_interaction(interaction, lig_res, residues, is_inverted) -> list[bool]:
    """Check PiStacking interactions (FaceToFace, EdgeToFace, PiStacking)."""
    results = []
    for res in residues:
        found = False
        for pi_rings in product(interaction.pi_ring, repeat=2):
            lig_matches = lig_res.GetSubstructMatches(pi_rings[1])
            res_matches = res.GetSubstructMatches(pi_rings[0])

            if not (lig_matches and res_matches):
                continue

            for lig_match, res_match in product(lig_matches, res_matches):
                lig_pi_coords = lig_res.xyz[list(lig_match)]
                lig_centroid = Point3D(*get_centroid(lig_pi_coords))
                res_pi_coords = res.xyz[list(res_match)]
                res_centroid = Point3D(*get_centroid(res_pi_coords))
                centroid_dist = lig_centroid.Distance(res_centroid)

                if centroid_dist > interaction.distance:
                    continue

                lig_normal = get_ring_normal_vector(lig_centroid, lig_pi_coords)
                res_normal = get_ring_normal_vector(res_centroid, res_pi_coords)
                plane_angle = lig_normal.AngleTo(res_normal)

                if not angle_between_limits(plane_angle, *interaction.plane_angle, ring=True):
                    continue

                c1c2 = lig_centroid.DirectionVector(res_centroid)
                c2c1 = res_centroid.DirectionVector(lig_centroid)
                n1c1c2 = lig_normal.AngleTo(c1c2)
                n2c2c1 = res_normal.AngleTo(c2c1)

                ncc_ok = (
                    angle_between_limits(n1c1c2, *interaction.normal_to_centroid_angle, ring=True) or
                    angle_between_limits(n2c2c1, *interaction.normal_to_centroid_angle, ring=True)
                )

                if ncc_ok:
                    if interaction.intersect:
                        intersect = interaction._get_intersect_point(
                            lig_normal, lig_centroid, res_normal, res_centroid
                        )
                        if intersect is not None:
                            intersect_dist = min(
                                lig_centroid.Distance(intersect),
                                res_centroid.Distance(intersect)
                            )
                            if intersect_dist <= interaction.intersect_radius:
                                found = True
                                break
                    else:
                        found = True
                        break
            if found:
                break
        results.append(found)

    return results


def _has_pistacking_composite(interaction, lig_res, residues) -> list[bool]:
    """Check PiStacking which is FaceToFace OR EdgeToFace."""
    ftf_results = _has_pistacking_interaction(interaction.ftf, lig_res, residues, False)
    etf_results = _has_pistacking_interaction(interaction.etf, lig_res, residues, False)
    return [f or e for f, e in zip(ftf_results, etf_results)]


def _has_vdwcontact_interaction(interaction, lig_res, residues) -> list[bool]:
    """Check VdWContact interactions (all atoms, based on VdW radii)."""
    lig_coords = lig_res.xyz
    lig_elements = [lig_res.GetAtomWithIdx(i).GetSymbol() for i in range(lig_res.GetNumAtoms())]
    lig_radii = jnp.array([interaction.vdwradii.get(e, 1.7) for e in lig_elements])

    results = []
    for res in residues:
        res_coords = res.xyz
        res_elements = [res.GetAtomWithIdx(i).GetSymbol() for i in range(res.GetNumAtoms())]
        res_radii = jnp.array([interaction.vdwradii.get(e, 1.7) for e in res_elements])

        distances = compute_distances_batch(lig_coords, res_coords)
        radii_sum = lig_radii[:, None] + res_radii[None, :]
        has_contact = bool((distances <= (radii_sum + interaction.tolerance)).any())
        results.append(has_contact)

    return results


def has_interaction_batch(
    interaction,
    lig_res,
    residues: list,
) -> list[bool]:
    """Check if interaction exists for each residue (boolean result).

    Args:
        interaction: A ProLIF interaction instance
        lig_res: Ligand residue
        residues: List of protein residues

    Returns:
        List of booleans, True if interaction exists for that residue.
    """
    is_inverted = _is_inverted_interaction(interaction)
    itype = _get_interaction_type(interaction)

    if itype == 'distance':
        return _has_distance_interaction(interaction, lig_res, residues, is_inverted)
    elif itype == 'singleangle':
        return _has_singleangle_interaction(interaction, lig_res, residues, is_inverted)
    elif itype == 'doubleangle':
        return _has_doubleangle_interaction(interaction, lig_res, residues, is_inverted)
    elif itype == 'cationpi':
        return _has_cationpi_interaction(interaction, lig_res, residues, is_inverted)
    elif itype == 'pistacking':
        return _has_pistacking_interaction(interaction, lig_res, residues, is_inverted)
    elif itype == 'pistacking_composite':
        return _has_pistacking_composite(interaction, lig_res, residues)
    elif itype == 'vdwcontact':
        return _has_vdwcontact_interaction(interaction, lig_res, residues)
    else:
        return _has_distance_interaction(interaction, lig_res, residues, is_inverted)
