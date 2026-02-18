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
    from .api import analyze_trajectory, analyze_frame, InteractionResult

    # Low-level primitives
    from .primitives import pairwise_distances
    from .integration import compute_distances_batch, has_interaction_batch
    from .framebatch import (
        pairwise_distances_frames,
        hbacceptor_frames,
        hbdonor_frames,
        build_actor_masks,
        build_angle_indices,
        build_ring_cation_indices,
        build_vdw_radii,
        has_interactions_frames,
        xbacceptor_frames,
        xbdonor_frames,
        cationpi_frames,
        pistacking_frames,
        prepare_for_device,
        get_gpu_device,
        get_gpu_memory_info,
        estimate_memory_per_frame,
        calculate_chunk_size,
        chunked_has_interactions_frames,
    )

    __all__ = [
        # Availability flag
        "JAX_AVAILABLE",
        # High-level API
        "analyze_trajectory",
        "analyze_frame",
        "InteractionResult",
        # Low-level primitives
        "pairwise_distances",
        "compute_distances_batch",
        "has_interaction_batch",
        "pairwise_distances_frames",
        "hbacceptor_frames",
        "hbdonor_frames",
        "build_actor_masks",
        "build_angle_indices",
        "build_ring_cation_indices",
        "build_vdw_radii",
        "has_interactions_frames",
        "xbacceptor_frames",
        "xbdonor_frames",
        "cationpi_frames",
        "pistacking_frames",
        "prepare_for_device",
        "get_gpu_device",
        "get_gpu_memory_info",
        "estimate_memory_per_frame",
        "calculate_chunk_size",
        "chunked_has_interactions_frames",
    ]
else:
    __all__ = ["JAX_AVAILABLE"]
