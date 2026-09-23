# JAX Interaction Fingerprinting

JAX-backed interaction fingerprinting for ProLIF trajectory and single-frame workflows.

## Requirements

- `jax`
- `jaxlib`

CPU install:

```bash
pip install "jax[cpu]"
```

GPU install depends on your CUDA setup. Use the JAX install guide for the correct wheel.

## Quick Start

### Trajectory analysis

```python
import MDAnalysis as mda
from prolif.interactions._jax import analyze_trajectory

u = mda.Universe("topology.pdb", "trajectory.xtc")

results = analyze_trajectory(
    u,
    ligand_selection="resname LIG",
    protein_selection="protein",
    device="cpu",  # or "gpu"
)

df = results.to_dataframe()
```

### Single frame analysis

```python
from prolif.interactions._jax import analyze_frame

frame_result = analyze_frame(ligand_mol, protein_mol, cutoff=6.0)
```

## Main API

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
    residue_mode="all",
    scan_stride=1,
)
```

Key parameters:

- `device`: `"cpu"` or `"gpu"`
- `chunk_size`: frames per batch; if `None`, CPU uses `256`, GPU uses auto size capped at `256`
- `residue_mode`:
  - `"all"`: select residues seen near ligand across trajectory
  - `"first"`: select residues from first frame only
- `scan_stride`: only used with `residue_mode="all"`

Returns `InteractionResult` with:

- `interactions`: `dict[str, np.ndarray]` of shape `(n_frames, n_residues)`
- `residue_ids`
- `n_frames`
- `n_residues`

## Interaction Types

The high-level trajectory API returns these nine interaction maps:

- `Hydrophobic`
- `Cationic`
- `Anionic`
- `VdWContact`
- `HBAcceptor`
- `HBDonor`
- `PiStacking`
- `CationPi`
- `PiCation`

## Notes

- First call may be slower due to JIT compilation.
- If GPU execution fails due memory pressure, set a smaller `chunk_size` (for example `128`, `64`, or `32`).
- Check availability with:

```python
from prolif.interactions._jax import JAX_AVAILABLE
```
