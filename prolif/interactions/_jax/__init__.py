"""
JAX-accelerated helpers for ProLIF.

High-level API (recommended):
- analyze_trajectory: Process full MD trajectories with automatic GPU/chunking
- analyze_frame: Single-frame analysis matching ProLIF Molecule API

Low-level primitives (for custom pipelines):
- pairwise_distances, has_interactions_frames, etc.
"""

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    # High-level API (recommended entry points)
    from .api import InteractionResult, analyze_frame, analyze_trajectory
    from .framebatch import (
        build_actor_masks,
        build_angle_indices,
        build_ring_cation_indices,
        build_vdw_radii,
        calculate_chunk_size,
        cationpi_frames,
        chunked_has_interactions_frames,
        estimate_memory_per_frame,
        get_gpu_device,
        get_gpu_memory_info,
        has_interactions_frames,
        hbacceptor_frames,
        hbdonor_frames,
        pairwise_distances_frames,
        pistacking_frames,
        prepare_for_device,
        xbacceptor_frames,
        xbdonor_frames,
    )
    from .integration import compute_distances_batch, has_interaction_batch

    # Low-level primitives
    from .primitives import pairwise_distances

    __all__ = [
        "JAX_AVAILABLE",
        "InteractionResult",
        "analyze_frame",
        "analyze_trajectory",
        "build_actor_masks",
        "build_angle_indices",
        "build_ring_cation_indices",
        "build_vdw_radii",
        "calculate_chunk_size",
        "cationpi_frames",
        "chunked_has_interactions_frames",
        "compute_distances_batch",
        "estimate_memory_per_frame",
        "get_gpu_device",
        "get_gpu_memory_info",
        "has_interaction_batch",
        "has_interactions_frames",
        "hbacceptor_frames",
        "hbdonor_frames",
        "pairwise_distances",
        "pairwise_distances_frames",
        "pistacking_frames",
        "prepare_for_device",
        "xbacceptor_frames",
        "xbdonor_frames",
    ]
else:
    __all__ = ["JAX_AVAILABLE"]
