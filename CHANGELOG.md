# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0.post1] - 2023-09-13

This is a post-release to fix releases not containing the complete test suite.

## [2.0.0.post0] - 2023-09-13

This is a post-release to fix an issue with the conda build not being able to run the
tests.

## [2.0.0] - 2023-09-03

### Added
- Added a `display_residues` function to quickly visualize the residues in a `Molecule`
  object.
- Added a `Complex3D` class for plotting interactions in 3D. Added the corresponding
  `Fingerprint.plot_3d` method to generate the plot directly from an FP object.
- Added a `Barcode` class for plotting interactions. Added the corresponding
  `Fingerprint.plot_barcode` method to generate the plot directly from an FP object.
- Added a `count` argument in `Fingerprint`. If `count=True`, enumerates all groups of
  atoms that satisfy interaction constraints (instead of stopping at the first one),
  allowing users to generate a count-fingerprint. The `Fingerprint.to_dataframe` method
  has been modified accordingly, and a `Fingerprint.to_countvectors` method has been
  added to generate a list of RDKit's `UIntSparseIntVect` from the count-fingerprint.
  The visualisation scripts have been updated to display the occurence with the shortest
  distance when a count-fingerprint is being used.
- Added a `parameters` argument in `Fingerprint` to easily update the parameters used
  by an interaction, instead of defining a new interaction class (Issue #118).
- Added new abstract interaction classes `SingleAngle` and `DoubleAngle` to more easily
  create custom interactions.
- Added the `vdwradii` parameter to the `VdWContact` interaction class to update the
  radii it uses.
- Added the `Fingerprint.plot_lignetwork` method to generate a `LigNetwork` plot directly.
- Added `LigNetwork.from_fingerprint` to generate the ligplot from a `Fingerprint`
  instance. Added a `display_all` parameter for displaying all interactions instead
  of only the shortest one for a given pair of residues. Added `use_coordinates` and
  `flatten_coordinates` to control how the ligand structure is displayed.
- Added support for displaying peptides with the `LigNetwork`.
- Added `Fingerprint.metadata` to generate a dictionary containing metadata about
  interactions between two residues. Replaces `Fingerprint.bitvector_atoms`.
- Added a `vicinity_cutoff` parameter in `Fingerprint` to control the distance cutoff
  used to automatically restrict the IFP calculation to residues within the specified
  range of the ligand.
- Added a `metadata` method to the base `Interaction` class to easily generate metadata
  for custom interactions.
- Added an `Interaction.invert_class` classmethod to easily invert the role of the
  ligand and protein residues in an interaction, e.g. to create a donor class from an
  acceptor class.

### Changed
- The tutorials have been overhauled and should now be much easier to navigate.
- The multiprocessing and pickling backends have been switched to `multiprocess` and
  `dill` respectively, and the parallel implementation has been improved. Users should
  now be able to define custom interactions in Jupyter notebooks, IPython and so on
  without any issue (Issue #117, Issue #86).
- The `LigNetwork` plot now displays the distance for each interaction on mouse hover.
- Changed the format of the `Fingerprint.ifp` attribute to be a dictionary indexed by
  frame/structure index. The values are custom `IFP` dictionaries that can be more
  easily indexed by using residue identifier strings (e.g. `ALA216.A`) rather than
  `ResidueId` objects. Each entry contains complete interaction metadata instead of just
  atom indices.
- All interaction classes now return more complete details about the interaction (e.g.
  distances, angles, atom indices in the residue and parent molecule).
- Changed the default color for `VanDerWaals` interactions in the builtin plots.
- Converting the IFP to a dataframe with atom indices has been optimized and now runs
  about 5 times faster (Issue #112, PR #113 by @ReneHamburger1993). *Note: discarded*
  *by the subsequent updates to the codebase which removed the ability to have*
  *atom indices in the dataframe.*
- Various changes related to packaging, code formatting, linting and CI pipelines
  (PR #114).

### Fixed
- Fixed pickling properties on RDKit molecules for Windows.

### Removed
- Removed the `return_atoms` argument in `Fingerprint.to_dataframe`. Users should
  directly use `Fingerprint.ifp` instead (the documentation's tutorials have been
  updated accordingly).
- Removed the `Fingerprint.bitvector_atoms` method, replaced by `Fingerprint.metadata`.
- Removed the `__wrapped__` attribute on interaction methods that are available
  from the `Fingerprint` object. These methods now accept a `metadata` parameter
  instead.
- Removed `LigNetwork.from_ifp` in favor of `LigNetwork.from_fingerprint`.
- Removed the `match3D` parameter in `LigNetwork`. Replaced by `use_coordinates` and
  `flatten_coordinates` to give users more control and allow them to provide their own
  2D coordinates.


## [1.1.0] - 2022-11-18

### Added
- `Fingerprint.run` now has a `converter_kwargs` parameter that can pass kwargs to the
  underlying RDKitConverter from MDAnalysis (Issue #57).
- Formatting with `black`.

### Changed
- The SMARTS for the following groups have been updated to a more accurate definition
  (Issue #68, PR #73 by @DrrDom, and PR #84):
  - Hydrophobic: excluded F, Cl, tetracoordinated C and S, C connected to N, O or F.
  - HBond donor: exclude charged O, S and charged aromatic N, only accept nitrogen
    that is in valence 3 or ammonium
  - HBond acceptor: exclude amides and some amines, exclude biaryl ethers and alkoxy
    oxygen from esters, include some aromatic oxygen and nitrogen,
  - Anion: include resonance forms of carboxylic, sulfonic and phosphorus acids,
  - Cation: include amidine and guanidine,
  - Metal ligand: exclude amides and some amines.
- The Pi stacking interactions have been changed for a more accurate implementation
  (PR #97, PR #98).
- The Van der Waals contact has been added to the default interactions, and the `tolerance`
  parameter has been set to 0. 
- The `pdbqt_supplier` will not add explicit hydrogen atoms anymore, to avoid detecting
  hydrogen bonds with "random" hydrogens that weren't in the PDBQT file (PR #99).
- When using the `pdbqt_supplier`, irrelevant warnings and logs have been disabled (PR #99).
- Updated the minimal RDKit version to `2021.03.1`
  
### Fixed
- Dead link in the quickstart notebook for the MDAnalysis quickstart (PR #75, @radifar).
- The `pdbqt_supplier` now correctly preserves hydrogens from the input PDBQT file (PR #99).
- If no interaction was detected, `to_dataframe` would error without giving a helpful message. It
  now returns a dataframe with the correct number of frames in the index and no column.


## [1.0.0] - 2022-06-07

### Added
- Support for multiprocessing, enabled by default (Issue #46). The number of processes can
  be controlled through `n_jobs` in `fp.run` and `fp.run_from_iterable`.
- New interaction: van der Waals contact, based on the sum of vdW radii of two atoms.
- Saving/loading the fingerprint object as a pickle with `fp.to_pickle` and
  `Fingerprint.from_pickle` (Issue #40).

### Changed
- Molecule suppliers can now be indexed, reused and can return their length, instead of
  being single-use generators.

### Fixed
- ProLIF can now be installed through pip and conda (Issue #6).
- If no interaction is detected in the first frame, `to_dataframe` will not complain about
  a `KeyError` anymore (Issue #44).
- When creating a `plf.Fingerprint`, unknown interactions will no longer fail silently.


## [0.3.4] - 2021-09-28

### Added
- Added our J. Cheminformatics article to the citation page of the documentation and the
  `CITATION.cff` file. 

### Changed
- Improved the documentation on how to properly restrict interactions to ignore the
  protein backbone (Issue #22), how to fix the empty dataframe issue when no bond
  information is present in the PDB file (Issue #15), how to save the LigNetwork diagram
  (Issue #21), and some clarifications on using `fp.generate`

### Fixed
- Mixing residue type with interaction type in the interactive legend of the LigNetwork
  would incorrectly display/hide some residues on the canvas (#PR 23)
- MOL2 files starting with a comment (`#`) would lead to an error


## [0.3.3] - 2021-06-11

### Changed
- Custom interactions must return three values: a boolean for the interaction,
  and the indices of residue atoms responsible for the interaction

### Fixed
- Custom interactions that only returned a single value instead of three would
  raise an uninformative error message


## [0.3.2] - 2021-06-11

### Added
- LigNetwork: an interaction diagram with atomistic details for the ligand and
  residue-level details for the protein, fully interactive in a browser/notebook, inspired
  from LigPlot (PR #19)
- `fp.generate`: a method to get the IFP between two `prolif.Molecule` objects (PR #19)

### Changed
- Default residue name and number: `UNK` and `0` are now the default values if `None` or
  `''` is given
- The Hydrophobic interaction now uses `+0` (no charge) instead of `!$([+{1-},-{1-}])`
  (not negatively or positively charged) for part of its SMARTS pattern (PR #19)
- Moved the `return_atoms` parameter from the `run` methods to `to_dataframe` to avoid
  recalculating the IFP if one wants to display it with atomic details (PR #19)
- Changed the values returned by `fp.bitvector_atoms`: the atom indices have been
  separated in two lists, one for the ligand and one for the protein (PR #19)

### Fixed
- Residues with a resnumber of `0` are not converted to `None` anymore (Issue #13)
- Fingerprint instantiated with an unknown interaction name will now raise a `NameError`


## [0.3.1] - 2021-02-02

### Added
- Integration with Zenodo to automatically generate a DOI for new releases
- Citation page
- Docking section in the Quickstart notebook (Issue #11)
- PDBQT, MOL2 and SDF molecule suppliers to make it easier for users to use docking
  results as input (Issue #11)
- `Molecule.from_rdkit` classmethod to easily prepare RDKit molecules for ProLIF

### Changed
- The visualisation notebook now displays the protein with py3Dmol. Some examples for
  creating and displaying a graph from the interaction dataframe have been added
- Updated the installation instructions to show how to install a specific release
- The previous repr method of `ResidueId` was easy to confuse with a string, especially
  when trying to access the `Fingerprint.ifp` results by string. The new repr method is
  now more explicit.
- Added the `Fingerprint.run_from_iterable` method, which uses the new supplier functions
  to quickly generate a fingerprint.
- Sorted the output of `Fingerprint.list_available`

### Fixed
- `Fingerprint.to_dataframe` is now much faster (Issue #7)
- `ResidueId.from_string` method now supports 1-letter and 2-letter codes for RNA/DNA
  (Issue #8)


## [0.3.0] - 2020-12-23

### Added
- Reading input directly from RDKit Mol as well as MDAnalysis AtomGroup objects
- Proper documentation and tests
- CI through GitHub Actions
- Publishing to PyPI triggered by GitHub releases

### Changed
- All the API and the underlying code have been modified
- Repository has been moved from GitHub user @cbouy to organisation @chemosim-lab

### Removed
- Custom MOL2 file reader
- Command-line interface

### Fixed
- Interactions not detected properly


## [0.2.1] - 2019-10-02

Base version for this changelog