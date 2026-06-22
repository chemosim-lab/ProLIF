"""High-level API for JAX interaction fingerprinting."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import jax
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
    has_interactions_frames,
    prepare_for_device,
)

INTERACTION_NAMES = [
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
GPU_AUTO_CHUNK_CAP = 256


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


def _resolve_frame_count(trajectory: Any, max_frames: int | None = None) -> int:
    """Return number of frames to process after applying max_frames."""
    total = len(trajectory)
    return total if not max_frames or max_frames <= 0 else min(total, int(max_frames))


def _build_residue_valid_mask(residue_ags: list[Any]) -> tuple[jnp.ndarray, int]:
    """Build (R, M) valid-atom mask and return max atoms per residue."""
    max_m = max((r.n_atoms for r in residue_ags), default=0)
    rows = []
    for r in residue_ags:
        m = r.n_atoms
        row = np.zeros((max_m,), dtype=bool)
        row[:m] = True
        rows.append(row)
    mask = jnp.array(np.stack(rows, axis=0) if rows else np.zeros((0, 0), dtype=bool))
    return mask, max_m


def _iter_trajectory_chunks(
    universe: Any,
    lig_ag: Any,
    residue_ags: list[Any],
    max_m: int,
    chunk_size: int,
    max_frames: int | None = None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, int]]:
    """Yield streamed coordinate chunks as (lig_chunk, res_chunk, n_frames_chunk)."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    F = _resolve_frame_count(universe.trajectory, max_frames=max_frames)
    N = lig_ag.n_atoms

    lig_frames: list[np.ndarray] = []
    res_frames: list[np.ndarray] = []

    for _ in universe.trajectory[:F]:
        lig_frames.append(np.array(lig_ag.positions, dtype=float).reshape(N, 3))

        row_list = []
        for r in residue_ags:
            coords = np.array(r.positions, dtype=float)
            m = coords.shape[0]
            if m < max_m:
                pad = np.zeros((max_m - m, 3), dtype=float)
                coords = np.concatenate([coords, pad], axis=0)
            row_list.append(coords)

        if row_list:
            res_frames.append(np.stack(row_list, axis=0))
        else:
            res_frames.append(np.zeros((0, max_m, 3), dtype=float))

        if len(lig_frames) == chunk_size:
            yield (
                jnp.array(np.stack(lig_frames, axis=0)),
                jnp.array(np.stack(res_frames, axis=0)),
                len(lig_frames),
            )
            lig_frames.clear()
            res_frames.clear()

    if lig_frames:
        yield (
            jnp.array(np.stack(lig_frames, axis=0)),
            jnp.array(np.stack(res_frames, axis=0)),
            len(lig_frames),
        )


def _get_residue_ags_from_ids(
    prot_ag: Any,
    residue_ids: list[ResidueId],
    use_segid: bool = False,
) -> list[Any]:
    """Return protein AtomGroups matching the provided residue identifiers."""
    by_key: dict[tuple[str, int, str | None], Any] = {}
    for r in prot_ag.residues:
        chain: str | None
        if use_segid:
            chain = str(int(r.segindex))
        else:
            try:
                chain = str(r.atoms[0].chainID).strip() or None
            except Exception:
                chain = None
        by_key[r.resname, r.resid, chain] = r.atoms

    result: list[Any] = []
    for rid in residue_ids:
        key = (rid.name or "", rid.number, rid.chain)
        ag = by_key.get(key)
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
    use_segid: bool = False,
) -> set[tuple[str, int, str | None]]:
    """Return residue keys that contact the ligand within the distance cutoff."""
    F_total = len(universe.trajectory)
    F = F_total if not max_frames or max_frames <= 0 else min(F_total, int(max_frames))

    residue_set: set[tuple[str, int, str | None]] = set()

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

        for res in nearby_atoms.residues:
            chain: str | None
            if use_segid:
                chain = str(int(res.segindex))
            else:
                try:
                    chain = str(res.atoms[0].chainID).strip() or None
                except Exception:
                    chain = None
            residue_set.add((res.resname, res.resid, chain))

    return residue_set


def _residue_keys_to_ids(
    residue_keys: set[tuple[str, int, str | None]],
    prot_mol: Any,
) -> list[ResidueId]:
    """Return residue IDs for the given residue keys."""
    by_key = {(rid.name, rid.number, rid.chain): rid for rid in prot_mol.residues}
    matched_ids: list[ResidueId] = []
    for key in sorted(
        residue_keys, key=lambda x: (x[0], x[1], "" if x[2] is None else x[2])
    ):
        rid = by_key.get(key)
        if rid is None:
            raise KeyError(f"Could not map residue key {key} to a ProLIF ResidueId")
        matched_ids.append(rid)

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

    Args:
        universe: MDAnalysis Universe with trajectory loaded.
        ligand_selection: MDAnalysis selection string for the ligand.
        protein_selection: MDAnalysis selection string for the protein.
        cutoff: Distance cutoff (Angstroms) for selecting residues near the ligand.
        max_frames: Maximum number of frames to process. If None, process all.
        device: 'cpu' or 'gpu'. GPU provides significant speedup for large trajectories.
        chunk_size: Frames per streamed batch. If None, GPU uses an auto-size
            capped at 256 frames and CPU uses 256 frames.
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

    use_segid = prolif.Molecule._use_segid(lig_ag) or prolif.Molecule._use_segid(
        prot_ag
    )
    lig_mol = prolif.Molecule.from_mda(lig_ag, use_segid=use_segid)
    prot_mol = prolif.Molecule.from_mda(prot_ag, use_segid=use_segid)

    if residue_mode == "first":
        residue_ids = get_residues_near_ligand(
            lig_mol, prot_mol, cutoff=cutoff, use_segid=use_segid
        )
    else:
        first_frame_ids = get_residues_near_ligand(
            lig_mol, prot_mol, cutoff=cutoff, use_segid=use_segid
        )

        residue_keys = _scan_residues_all_frames(
            universe,
            lig_ag,
            prot_ag,
            cutoff,
            max_frames=max_frames,
            stride=scan_stride,
            use_segid=use_segid,
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
    residue_ags = _get_residue_ags_from_ids(prot_ag, residue_ids, use_segid=use_segid)

    if not residues:
        return InteractionResult(
            interactions={
                name: np.zeros((0, 0), dtype=bool) for name in INTERACTION_NAMES
            },
            residue_ids=[],
            n_frames=0,
            n_residues=0,
        )

    res_valid_mask, max_m = _build_residue_valid_mask(residue_ags)
    N = lig_ag.n_atoms
    R = len(residue_ags)
    if chunk_size is None:
        if device == "gpu":
            chunk_size = min(calculate_chunk_size(N, R, max_m), GPU_AUTO_CHUNK_CAP)
        else:
            chunk_size = 256
    chunk_size = int(max(1, chunk_size))

    lig_res_for_smarts = next(iter(lig_mol.residues.values()))
    lig_masks, res_actor_masks = build_actor_masks(lig_mol, residues)
    angle_idx = build_angle_indices(lig_ag, residue_ags)
    ring_idx = build_ring_cation_indices(lig_res_for_smarts, residues)
    vdw_radii = build_vdw_radii(
        lig_mol, residues, lig_ag=lig_ag, residue_ags=residue_ags, use_real=True
    )

    (
        _,
        _,
        res_valid_mask,
        lig_masks,
        res_actor_masks,
        angle_idx,
        ring_idx,
        vdw_radii,
    ) = prepare_for_device(
        jnp.zeros((0, N, 3), dtype=float),
        jnp.zeros((0, R, max_m, 3), dtype=float),
        res_valid_mask,
        lig_masks,
        res_actor_masks,
        angle_idx,
        ring_idx,
        vdw_radii,
        device=device,
    )

    gpu_dev = None
    if device == "gpu":
        try:
            gpus = jax.devices("gpu")
            gpu_dev = gpus[0] if gpus else None
        except Exception:
            gpu_dev = None

    chunk_results: list[dict[str, np.ndarray]] = []
    total_frames = 0

    for lig_chunk, res_chunk, n_chunk_frames in _iter_trajectory_chunks(
        universe,
        lig_ag,
        residue_ags,
        max_m=max_m,
        chunk_size=chunk_size,
        max_frames=max_frames,
    ):
        try:
            lig_chunk_dev = lig_chunk
            res_chunk_dev = res_chunk
            if gpu_dev is not None:
                lig_chunk_dev = jax.device_put(lig_chunk, device=gpu_dev)
                res_chunk_dev = jax.device_put(res_chunk, device=gpu_dev)

            chunk_result = has_interactions_frames(
                lig_chunk_dev,
                res_chunk_dev,
                res_valid_mask,
                lig_masks,
                res_actor_masks,
                angle_idx,
                ring_idx,
                vdw_radii,
                vicinity_cutoff=cutoff,
            )
        except Exception as exc:
            if device == "gpu":
                raise RuntimeError(
                    f"JAX chunk evaluation failed at chunk_size={chunk_size}. "
                    "Try a smaller chunk_size (for example: 128, 64, or 32)."
                ) from exc
            raise
        chunk_result = jax.device_get(chunk_result)
        chunk_results.append({k: np.asarray(v) for k, v in chunk_result.items()})
        total_frames += n_chunk_frames

    if chunk_results:
        np_results = {
            name: np.concatenate([chunk[name] for chunk in chunk_results], axis=0)
            for name in INTERACTION_NAMES
        }
    else:
        np_results = {
            name: np.zeros((0, len(residues)), dtype=bool) for name in INTERACTION_NAMES
        }

    return InteractionResult(
        interactions=np_results,
        residue_ids=list(residue_ids),
        n_frames=total_frames,
        n_residues=len(residues),
    )


def analyze_frame(
    ligand: Any,
    protein: Any,
    *,
    cutoff: float = 6.0,
) -> dict[ResidueId, dict[str, bool]]:
    """Compute interactions for a single ligand/protein frame.

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
