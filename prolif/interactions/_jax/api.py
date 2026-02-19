"""
High-level API for JAX-accelerated interaction fingerprinting.

This module provides a simple, user-friendly interface that hides the complexity
of coordinate extraction, SMARTS mask building, device placement, and chunking.

Example usage:
    >>> from prolif.interactions._jax import analyze_trajectory
    >>> results = analyze_trajectory(
    ...     universe,
    ...     ligand_selection="resname LIG",
    ...     protein_selection="protein",
    ...     device="gpu",
    ... )
    >>> print(results.interactions["HBDonor"].sum())  # total HB donor interactions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from MDAnalysis.lib.distances import capped_distance

import prolif
from prolif import ResidueId
from prolif.utils import get_residues_near_ligand

from .framebatch import (
    build_actor_masks,
    build_angle_indices,
    build_angle_indices_rdkit,
    build_ring_cation_indices,
    build_vdw_radii,
    calculate_chunk_size,
    chunked_has_interactions_frames,
    has_interactions_frames,
    prepare_for_device,
)


@dataclass
class InteractionResult:
    """Container for JAX-accelerated interaction fingerprint results.

    Attributes:
        interactions: Mapping from interaction name to (F, R) boolean array.
            True indicates the interaction is present for that frame/residue pair.
        residue_ids: List of ProLIF ResidueId objects for each residue column.
        n_frames: Number of frames processed.
        n_residues: Number of residues evaluated.
        interaction_names: List of interaction names in evaluation order.
    """

    interactions: dict[str, np.ndarray]
    residue_ids: list["ResidueId"]
    n_frames: int
    n_residues: int
    interaction_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.interaction_names:
            self.interaction_names = list(self.interactions.keys())

    def to_dataframe(self) -> Any:
        """Convert results to a pandas DataFrame with MultiIndex columns.

        Returns a DataFrame with shape (n_frames, n_residues * n_interactions)
        using a MultiIndex of (residue_id, interaction_name) for columns.
        Columns are ordered as: (res0, int0), (res0, int1), ..., (res1, int0), ...
        """
        import pandas as pd

        columns = pd.MultiIndex.from_product(
            [self.residue_ids, self.interaction_names],
            names=["residue", "interaction"],
        )

        reshaped = np.zeros(
            (self.n_frames, self.n_residues * len(self.interaction_names)), dtype=bool
        )
        for r_i in range(self.n_residues):
            for i_i, name in enumerate(self.interaction_names):
                col_idx = r_i * len(self.interaction_names) + i_i
                reshaped[:, col_idx] = self.interactions[name][:, r_i]

        return pd.DataFrame(reshaped, columns=columns)

    def get_contacts(
        self, interaction: str | None = None
    ) -> list[tuple[int, "ResidueId", str]]:
        """Get list of (frame_idx, residue_id, interaction_name) tuples
        where contacts occur.

        Args:
            interaction: If specified, filter to only this interaction type.

        Returns:
            List of tuples identifying each detected contact.
        """
        contacts = []
        names = [interaction] if interaction else self.interaction_names
        for name in names:
            if name not in self.interactions:
                continue
            arr = self.interactions[name]
            for f_i, r_i in np.argwhere(arr):
                contacts.append((int(f_i), self.residue_ids[int(r_i)], name))
        return contacts

    def count_by_residue(self) -> dict["ResidueId", dict[str, int]]:
        """Count interactions per residue across all frames.

        Returns:
            Mapping from residue_id to {interaction_name: count}.
        """
        result: dict[ResidueId, dict[str, int]] = {}
        for r_i, rid in enumerate(self.residue_ids):
            result[rid] = {}
            for name in self.interaction_names:
                result[rid][name] = int(self.interactions[name][:, r_i].sum())
        return result

    def count_by_frame(self) -> dict[int, dict[str, int]]:
        """Count interactions per frame across all residues.

        Returns:
            Mapping from frame_index to {interaction_name: count}.
        """
        result: dict[int, dict[str, int]] = {}
        for f_i in range(self.n_frames):
            result[f_i] = {}
            for name in self.interaction_names:
                result[f_i][name] = int(self.interactions[name][f_i, :].sum())
        return result


def _build_trajectory_frames(
    u: Any,
    lig_ag: Any,
    residue_ags: list[Any],
    max_frames: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build per-frame coordinate arrays from MDAnalysis trajectory."""
    F_total = len(u.trajectory)
    F = F_total if not max_frames or max_frames <= 0 else min(F_total, int(max_frames))

    N = lig_ag.n_atoms
    max_m = max((r.n_atoms for r in residue_ags), default=0)

    lig_frames = []
    res_frames = []
    res_masks = []
    for r in residue_ags:
        m = r.n_atoms
        rm = np.zeros((max_m,), dtype=bool)
        rm[:m] = True
        res_masks.append(rm)
    res_valid_mask = jnp.array(
        np.stack(res_masks, axis=0) if res_masks else np.zeros((0, 0), dtype=bool)
    )

    for _ in u.trajectory[:F]:
        lig_frames.append(np.array(lig_ag.positions, dtype=float).reshape(N, 3))
        row_list = []
        for r in residue_ags:
            coords = np.array(r.positions, dtype=float)
            m = coords.shape[0]
            if m < max_m:
                pad = np.zeros((max_m - m, 3), dtype=float)
                coords = np.concatenate([coords, pad], axis=0)
            row_list.append(coords)
        res_frames.append(
            np.stack(row_list, axis=0) if row_list else np.zeros((0, 0, 3), dtype=float)
        )

    lig_f = jnp.array(
        np.stack(lig_frames, axis=0) if lig_frames else np.zeros((0, N, 3), dtype=float)
    )
    res_f = jnp.array(
        np.stack(res_frames, axis=0)
        if res_frames
        else np.zeros((0, len(residue_ags), max_m, 3), dtype=float)
    )
    return lig_f, res_f, res_valid_mask


def _get_residue_ags_from_ids(
    prot_ag: Any,
    residue_ids: list[ResidueId],
) -> list[Any]:
    """Map ProLIF ResidueIds to MDAnalysis AtomGroups via direct residue lookup.

    Builds a lookup from (resname, resid, segid) to AtomGroup once, then
    matches each ResidueId directly without string-based selection.
    """
    by_key = {(r.resname, r.resid, r.segid): r.atoms for r in prot_ag.residues}
    by_resname_resid: dict[tuple[str, int], Any] = {}
    for r in prot_ag.residues:
        by_resname_resid.setdefault((r.resname, r.resid), r.atoms)

    result: list[Any] = []
    for rid in residue_ids:
        key = (rid.name or "", rid.number, rid.chain or "")
        ag = by_key.get(key) or by_resname_resid.get((rid.name or "", rid.number))
        if ag is None:
            raise ValueError(f"Could not find residue {rid} in protein AtomGroup")
        result.append(ag)
    return result


def _scan_residues_all_frames(
    universe: Any,
    lig_ag: Any,
    prot_ag: Any,
    cutoff: float,
    max_frames: int | None = None,
    stride: int = 1,
) -> set[tuple[str, int, str]]:
    """Scan all trajectory frames to find residues within cutoff of the ligand.

    Uses MDAnalysis capped_distance for efficient, PBC-aware neighbor searching.
    Returns a set of (resname, resid, segid) tuples identifying unique residues
    that come within the cutoff distance of any ligand atom at any frame.

    Args:
        universe: MDAnalysis Universe with trajectory loaded.
        lig_ag: Ligand AtomGroup.
        prot_ag: Protein AtomGroup.
        cutoff: Distance cutoff in Angstroms.
        max_frames: Maximum frames to scan. If None, scan all frames.
        stride: Frame stride for scanning. Default 1 scans every frame.

    Returns:
        Set of (resname, resid, segid) tuples for residues found within cutoff.
    """
    F_total = len(universe.trajectory)
    F = F_total if not max_frames or max_frames <= 0 else min(F_total, int(max_frames))

    residue_set: set[tuple[str, int, str]] = set()

    for ts in universe.trajectory[:F:stride]:
        box = (
            ts.dimensions
            if ts.dimensions is not None and ts.dimensions[0] > 0
            else None
        )

        pairs = capped_distance(
            lig_ag.positions,
            prot_ag.positions,
            max_cutoff=cutoff,
            box=box,
            return_distances=False,
        )

        if pairs.size == 0:
            continue

        prot_indices = np.unique(pairs[:, 1])
        nearby_atoms = prot_ag.atoms[prot_indices]

        residue_set.update(
            (res.resname, res.resid, res.segid) for res in nearby_atoms.residues
        )

    return residue_set


def _residue_keys_to_ids(
    residue_keys: set[tuple[str, int, str]],
    prot_mol: Any,
) -> list[ResidueId]:
    """Convert (resname, resid, segid) tuples to ProLIF ResidueId objects.

    Matches residue keys against the protein molecule to retrieve the
    corresponding ProLIF ResidueId objects, preserving the chain information.

    Args:
        residue_keys: Set of (resname, resid, segid) tuples.
        prot_mol: ProLIF Molecule for the protein.

    Returns:
        List of matching ProLIF ResidueId objects.
    """
    matched_ids: list[ResidueId] = []
    for resname, resid, segid in sorted(residue_keys):
        for rid in prot_mol.residues:
            if rid.name == resname and rid.number == resid:
                if segid and rid.chain and rid.chain != segid:
                    continue
                matched_ids.append(rid)
                break
        else:
            matched_ids.append(ResidueId(resname, resid, segid or None))

    return matched_ids


def analyze_trajectory(
    universe: Any,
    ligand_selection: str = "resname LIG",
    protein_selection: str = "protein",
    *,
    cutoff: float = 6.0,
    max_frames: int | None = None,
    device: str = "cpu",
    chunk_size: int | None = None,
    residue_mode: str = "all",
    scan_stride: int = 1,
) -> InteractionResult:
    """Compute interaction fingerprints for all frames in a trajectory.

    This is the main entry point for JAX-accelerated interaction analysis.
    It handles all the complexity of coordinate extraction, SMARTS matching,
    device placement, and memory-safe chunking.

    Args:
        universe: MDAnalysis Universe with trajectory loaded.
        ligand_selection: MDAnalysis selection string for the ligand.
        protein_selection: MDAnalysis selection string for the protein.
        cutoff: Distance cutoff (Angstroms) for selecting residues near the ligand.
        max_frames: Maximum number of frames to process. If None, process all.
        device: 'cpu' or 'gpu'. GPU provides significant speedup for large trajectories.
        chunk_size: Frames per GPU batch. If None, auto-calculated based on
            available GPU memory. Ignored for CPU.
        residue_mode: Residue selection strategy. Default is 'all'.
            'all': Pre-scan all frames to find any residue that enters the
                cutoff at any point. Correct for drifting or unbinding ligands.
                Use residue_mode='first' to opt into faster setup for stable
                bound systems where the ligand doesn't move much.
            'first': Select residues within cutoff in the first frame only.
                Fast setup, but misses residues that only come into contact
                in later frames.
        scan_stride: Frame stride for the 'all' mode pre-scan. Default 1 scans
            every frame. Higher values reduce scan time at the cost of potentially
            missing briefly-contacting residues. Ignored for 'first' mode.

    Returns:
        InteractionResult containing boolean arrays for each interaction type:
        Hydrophobic, Cationic, Anionic, VdWContact, HBAcceptor, HBDonor,
        PiStacking, CationPi, PiCation.

    Example:
        >>> import MDAnalysis as mda
        >>> from prolif.interactions._jax import analyze_trajectory
        >>>
        >>> u = mda.Universe("topology.pdb", "trajectory.xtc")
        >>> results = analyze_trajectory(
        ...     u,
        ...     ligand_selection="resname LIG",
        ...     protein_selection="protein",
        ...     device="gpu",
        ...     residue_mode="all",
        ... )
        >>>
        >>> hb_counts = results.interactions["HBDonor"].sum(axis=0)
        >>> for rid, count in zip(results.residue_ids, hb_counts):
        ...     if count > 0:
        ...         print(f"{rid}: {count} HB donor contacts")
    """
    logger = logging.getLogger(__name__)

    if residue_mode not in {"first", "all"}:
        raise ValueError(f"residue_mode must be 'first' or 'all', got '{residue_mode}'")

    lig_ag = universe.select_atoms(ligand_selection)
    prot_ag = universe.select_atoms(protein_selection)

    if lig_ag.n_atoms == 0:
        raise ValueError(f"Ligand selection '{ligand_selection}' matched no atoms")
    if prot_ag.n_atoms == 0:
        raise ValueError(f"Protein selection '{protein_selection}' matched no atoms")

    universe.trajectory[0]

    lig_mol = prolif.Molecule.from_mda(lig_ag)
    prot_mol = prolif.Molecule.from_mda(prot_ag)

    if residue_mode == "first":
        residue_ids = get_residues_near_ligand(lig_mol, prot_mol, cutoff=cutoff)
    else:
        first_frame_ids = get_residues_near_ligand(lig_mol, prot_mol, cutoff=cutoff)

        residue_keys = _scan_residues_all_frames(
            universe,
            lig_ag,
            prot_ag,
            cutoff,
            max_frames=max_frames,
            stride=scan_stride,
        )
        residue_ids = _residue_keys_to_ids(residue_keys, prot_mol)

        logger.info(
            "Residue scan complete: %d residues within %.1f A across trajectory "
            "(first frame: %d residues)",
            len(residue_ids),
            cutoff,
            len(first_frame_ids),
        )

        universe.trajectory[0]

    residues = [prot_mol[rid] for rid in residue_ids]
    residue_ags = _get_residue_ags_from_ids(prot_ag, residue_ids)

    if not residues:
        return InteractionResult(
            interactions={
                name: np.zeros((0, 0), dtype=bool)
                for name in [
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
            },
            residue_ids=[],
            n_frames=0,
            n_residues=0,
        )

    lig_f, res_f, res_valid_mask = _build_trajectory_frames(
        universe, lig_ag, residue_ags, max_frames=max_frames
    )
    F = int(lig_f.shape[0])

    lig_res_for_smarts = next(iter(lig_mol.residues.values()))
    lig_masks, res_actor_masks = build_actor_masks(lig_mol, residues)
    angle_idx = build_angle_indices(lig_ag, residue_ags)
    ring_idx = build_ring_cation_indices(lig_res_for_smarts, residues)
    vdw_radii = build_vdw_radii(
        lig_mol, residues, lig_ag=lig_ag, residue_ags=residue_ags, use_real=True
    )

    (
        lig_f,
        res_f,
        res_valid_mask,
        lig_masks,
        res_actor_masks,
        angle_idx,
        ring_idx,
        vdw_radii,
    ) = prepare_for_device(
        lig_f,
        res_f,
        res_valid_mask,
        lig_masks,
        res_actor_masks,
        angle_idx,
        ring_idx,
        vdw_radii,
        device=device,
    )

    if device == "gpu":
        N = int(lig_f.shape[1])
        R = int(res_f.shape[1])
        M = int(res_f.shape[2])
        if chunk_size is None:
            chunk_size = calculate_chunk_size(N, R, M)
        results = chunked_has_interactions_frames(
            lig_f,
            res_f,
            res_valid_mask,
            lig_masks,
            res_actor_masks,
            angle_idx,
            ring_idx,
            vdw_radii,
            chunk_size=chunk_size,
        )
    else:
        results = has_interactions_frames(
            lig_f,
            res_f,
            res_valid_mask,
            lig_masks,
            res_actor_masks,
            angle_idx,
            ring_idx,
            vdw_radii,
            vicinity_cutoff=cutoff,
        )

    np_results = {k: np.asarray(v) for k, v in results.items()}

    return InteractionResult(
        interactions=np_results,
        residue_ids=list(residue_ids),
        n_frames=F,
        n_residues=len(residues),
    )


def analyze_frame(
    ligand: Any,
    protein: Any,
    *,
    cutoff: float = 6.0,
) -> dict[ResidueId, dict[str, bool]]:
    """Compute interactions for a single frame (ProLIF Molecule objects).

    This is a convenience function for single-frame analysis that accepts
    ProLIF Molecule objects directly, matching the ProLIF API style.

    Args:
        ligand: ProLIF Molecule for the ligand.
        protein: ProLIF Molecule for the protein.
        cutoff: Distance cutoff for residue selection.

    Returns:
        Dict mapping ResidueId -> {interaction_name: bool}.
    """
    residue_ids = get_residues_near_ligand(ligand, protein, cutoff=cutoff)
    residues = [protein[rid] for rid in residue_ids]

    if not residues:
        return {}

    N = ligand.GetNumAtoms()
    max_m = max(r.GetNumAtoms() for r in residues)

    conf = ligand.GetConformer()
    lig_coords = np.array([conf.GetAtomPosition(i) for i in range(N)], dtype=float)
    lig_f = jnp.array(lig_coords[None, :, :])

    res_coords = []
    res_masks = []
    for r in residues:
        rc = r.GetConformer()
        m = r.GetNumAtoms()
        coords = np.array([rc.GetAtomPosition(i) for i in range(m)], dtype=float)
        if m < max_m:
            pad = np.zeros((max_m - m, 3), dtype=float)
            coords = np.concatenate([coords, pad], axis=0)
        res_coords.append(coords)
        mask = np.zeros((max_m,), dtype=bool)
        mask[:m] = True
        res_masks.append(mask)

    res_f = jnp.array(np.stack(res_coords, axis=0)[None, :, :, :])
    res_valid_mask = jnp.array(np.stack(res_masks, axis=0))

    lig_masks, res_actor_masks = build_actor_masks(ligand, residues)
    angle_idx = build_angle_indices_rdkit(ligand, residues)
    ring_idx = build_ring_cation_indices(ligand, residues)
    vdw_radii = build_vdw_radii(ligand, residues)

    results = has_interactions_frames(
        lig_f,
        res_f,
        res_valid_mask,
        lig_masks,
        res_actor_masks,
        angle_idx,
        ring_idx,
        vdw_radii,
        vicinity_cutoff=cutoff,
    )

    output: dict[ResidueId, dict[str, bool]] = {}
    for r_i, rid in enumerate(residue_ids):
        output[rid] = {}
        for name, arr in results.items():
            output[rid][name] = bool(arr[0, r_i])
    return output
