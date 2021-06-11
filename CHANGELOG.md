# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
### Deprecated
### Removed
### Fixed

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