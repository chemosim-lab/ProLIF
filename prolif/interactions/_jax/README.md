# JAX-Accelerated Interaction Fingerprinting

This module provides GPU and CPU acceleration for protein-ligand interaction
fingerprinting using JAX. It achieves significant speedups over the standard
ProLIF implementation by vectorizing geometry calculations across trajectory
frames.

## Performance

Benchmarks on a typical protein-ligand system (40 ligand atoms, 20 residues):

| Backend | Speedup vs ProLIF | Notes |
|---------|-------------------|-------|
| CPU (JAX) | 80-100x | After JIT warmup |
| GPU (JAX) | 100-200x | After JIT warmup |

The speedup comes from:
- Frame-batching: geometry is vectorized across all trajectory frames
- JIT compilation: JAX compiles optimized kernels for the specific system size
- SMARTS matching done once: pattern matching is structure-based, not coordinate-based

## Requirements

```
jax
jaxlib
```

For GPU acceleration:
```
jax[cuda12]  # or appropriate CUDA version
```

## Quick Start

### Trajectory Analysis

```python
import MDAnalysis as mda
from prolif.interactions._jax import analyze_trajectory

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.xtc")

# Run analysis (CPU)
results = analyze_trajectory(
    u,
    ligand_selection="resname LIG",
    protein_selection="protein",
)

# Run analysis (GPU)
results = analyze_trajectory(
    u,
    ligand_selection="resname LIG",
    protein_selection="protein",
    device="gpu",
)

# Access results
print(f"Analyzed {results.n_frames} frames, {results.n_residues} residues")

# Get hydrogen bond donor counts per residue
for rid, count in zip(results.residue_ids, results.interactions["HBDonor"].sum(axis=0)):
    if count > 0:
        print(f"{rid}: {count} HB donor contacts")
```

### Single Frame Analysis

```python
import prolif
from prolif.interactions._jax import analyze_frame

# Build molecules
lig_mol = prolif.Molecule.from_mda(ligand_atomgroup)
prot_mol = prolif.Molecule.from_mda(protein_atomgroup)

# Analyze single frame
results = analyze_frame(lig_mol, prot_mol)

# Results: {ResidueId: {"HBDonor": True, "Hydrophobic": False, ...}}
for rid, interactions in results.items():
    active = [k for k, v in interactions.items() if v]
    if active:
        print(f"{rid}: {', '.join(active)}")
```

## API Reference

### analyze_trajectory

```python
analyze_trajectory(
    universe,
    ligand_selection="resname LIG",
    protein_selection="protein",
    *,
    cutoff=6.0,
    max_frames=None,
    device="cpu",
    chunk_size=None,
    residue_mode="first",
    scan_stride=1,
) -> InteractionResult
```

**Parameters:**

- `universe`: MDAnalysis Universe with trajectory loaded
- `ligand_selection`: MDAnalysis selection string for the ligand
- `protein_selection`: MDAnalysis selection string for the protein
- `cutoff`: Distance cutoff (Angstroms) for selecting residues near the ligand
- `max_frames`: Maximum number of frames to process. If None, process all.
- `device`: `"cpu"` or `"gpu"`. GPU provides significant speedup for large trajectories.
- `chunk_size`: Frames per GPU batch. If None, auto-calculated based on
  available GPU memory. Ignored for CPU.
- `residue_mode`: Residue selection strategy (see Residue Selection Modes below).
  `"first"` (default) or `"all"`.
- `scan_stride`: Frame stride for the `"all"` mode pre-scan. Default 1 scans
  every frame. Higher values reduce scan time. Ignored for `"first"` mode.

**Returns:** `InteractionResult` object containing:

- `interactions`: Dict mapping interaction name to (n_frames, n_residues) boolean array
- `residue_ids`: List of ProLIF ResidueId objects
- `n_frames`: Number of frames processed
- `n_residues`: Number of residues evaluated

### InteractionResult

The result container provides several convenience methods:

```python
# Convert to pandas DataFrame with MultiIndex columns
df = results.to_dataframe()

# Get list of (frame_idx, residue_id, interaction_name) tuples
contacts = results.get_contacts()
contacts = results.get_contacts("HBAcceptor")  # filter by interaction

# Count interactions per residue across all frames
by_residue = results.count_by_residue()
# Returns: {ResidueId: {"HBDonor": 42, "Hydrophobic": 156, ...}}

# Count interactions per frame across all residues
by_frame = results.count_by_frame()
# Returns: {0: {"HBDonor": 3, ...}, 1: {"HBDonor": 2, ...}, ...}
```

### analyze_frame

```python
analyze_frame(
    ligand,
    protein,
    *,
    cutoff=6.0,
) -> dict[ResidueId, dict[str, bool]]
```

**Parameters:**

- `ligand`: ProLIF Molecule for the ligand
- `protein`: ProLIF Molecule for the protein
- `cutoff`: Distance cutoff for residue selection

**Returns:** Dict mapping ResidueId to interaction results.

## Residue Selection Modes

The `residue_mode` parameter controls which protein residues are evaluated:

### first (default)

Selects residues within the cutoff distance in the first frame only.

- Fast setup with no trajectory pre-scan
- Minimal memory footprint (smallest residue set)
- Best for stable, bound systems where the ligand remains in the binding site

### all

Pre-scans the entire trajectory to find any residue that enters the cutoff
at any frame, then evaluates all of them across all frames.

- Captures late-arriving contacts for drifting or unbinding ligands
- Uses MDAnalysis `capped_distance` for efficient, PBC-aware scanning
- Fixed residue set enables stable JAX array shapes (no JIT retracing)
- Early frames return False for residues not yet in contact (correct behavior)

```python
results = analyze_trajectory(
    u,
    ligand_selection="resname LIG",
    protein_selection="protein",
    residue_mode="all",
    scan_stride=1,
)
```

The scan logs the residue count comparison:
```
Residue scan complete: 45 residues within 6.0 A across trajectory (first frame: 23 residues)
```

For very long trajectories, increase `scan_stride` to reduce scan time at the
cost of potentially missing briefly-contacting residues.

## Supported Interactions

The module evaluates nine interaction types:

| Interaction | Description |
|-------------|-------------|
| Hydrophobic | Hydrophobic contacts within 4.5 A |
| Cationic | Cation-anion salt bridges (ligand cation) |
| Anionic | Cation-anion salt bridges (ligand anion) |
| VdWContact | Van der Waals contacts based on atomic radii |
| HBAcceptor | Hydrogen bonds (ligand as acceptor) |
| HBDonor | Hydrogen bonds (ligand as donor) |
| PiStacking | Pi-pi stacking (face-to-face and edge-to-face) |
| CationPi | Cation-pi interactions (ligand cation, protein ring) |
| PiCation | Cation-pi interactions (ligand ring, protein cation) |

## Architecture

The acceleration is achieved through a frame-batching strategy:

1. **SMARTS Matching (Once)**: Pattern matching for functional groups is performed
   once on the molecular graph. This identifies donor/acceptor atoms, rings,
   cations, etc. These indices are fixed across all frames since connectivity
   does not change during MD.

2. **Coordinate Extraction**: Trajectory coordinates are extracted into dense
   arrays with shapes:
   - Ligand: (n_frames, n_ligand_atoms, 3)
   - Residues: (n_frames, n_residues, max_atoms_per_residue, 3)

3. **Vectorized Geometry**: Distance and angle calculations are vectorized
   across all frames using JAX's `vmap`. This enables efficient parallelization
   on both CPU (via XLA) and GPU (via CUDA).

4. **Memory Management**: For GPU execution, the trajectory is automatically
   chunked to fit within available GPU memory. Chunk size is calculated based
   on the number of atoms and available memory.

## Low-Level API

For advanced use cases, the low-level primitives are also exported:

```python
from prolif.interactions._jax import (
    # Geometry functions
    pairwise_distances_frames,
    hbacceptor_frames,
    hbdonor_frames,
    pistacking_frames,
    cationpi_frames,
    xbacceptor_frames,
    xbdonor_frames,

    # Index building
    build_actor_masks,
    build_angle_indices,
    build_ring_cation_indices,
    build_vdw_radii,

    # Device management
    prepare_for_device,
    get_gpu_device,
    get_gpu_memory_info,
    calculate_chunk_size,

    # Core computation
    has_interactions_frames,
    chunked_has_interactions_frames,
)
```

## Limitations

1. **JIT Warmup**: The first call incurs JIT compilation overhead (typically 1-5
   seconds). Subsequent calls with the same array shapes are fast.

2. **Metal Interactions**: MetalDonor and MetalAcceptor interactions are matched
   via SMARTS but use simple distance criteria (no specialized metal geometry).

3. **Halogen Bonds**: XBAcceptor and XBDonor are implemented but not included in
   the default nine-interaction set returned by `analyze_trajectory`.

4. **Residue Selection Trade-offs**: With `residue_mode="first"` (default),
   residues that move into proximity later in the trajectory are not evaluated.
   Use `residue_mode="all"` to capture late-arriving contacts at the cost of
   slower setup and larger memory footprint.

## Checking JAX Availability

```python
from prolif.interactions._jax import JAX_AVAILABLE

if JAX_AVAILABLE:
    from prolif.interactions._jax import analyze_trajectory
    # Use JAX-accelerated path
else:
    # Fall back to standard ProLIF
    pass
```
