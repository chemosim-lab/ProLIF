# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
- Added the `Fingerprint.run_from_iterable` method, which uses the new supplier functions
  to quickly generate a fingerprint.
- Sorted the output of `Fingerprint.list_available`
### Deprecated
### Removed
### Fixed

## [0.3.0] - 2020-12-23
### Added
- Reading input directly from RDKit Mol as well as MDAnalysis AtomGroup objects
- Proper documentation and tests
- CI through GitHub Actions
- Publishing to PyPI triggered by GitHub releases
### Changed
- All the API and the underlying code have been modified
- Repository has been moved from GitHub user @cbouy to organisation @chemosim-lab
### Deprecated
### Removed
- Custom MOL2 file reader
- Command-line interface
### Fixed
- Interactions not detected properly

## [0.2.1] - 2019-10-02
Base version for this changelog