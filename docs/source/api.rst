Summary
=======

Input data
----------

Currently, ProLIF only accepts RDKit molecules as input and relies on
MDAnalysis to convert input files for Molecular Dynamics simulations and
docking experiments to this RDKit object.
The documentation can be found in the :ref:`source/modules/input:Molecules` section.

Residues
--------

To decompose interactions by residue, ProLIF needs to split the :class:`prolif.molecule.Molecule`
component in residues. Information on how these residues are stored and accessed
can be found in the :ref:`source/modules/residues:Residues` section.

Interaction fingerprint
-----------------------

An interaction fingerprint decomposes the interactions between two molecules
in a binary vector. The interactions are detected by looking up predefined
molecular patterns (using SMARTS queries in ProLIF) that satisfy geometrical
constraints (distance, angle, dihedral...).
To learn more on how these interactions are defined and how to use prolif to
extract fingerprints, please refer to the
:ref:`source/modules/interaction-fingerprint:Interaction fingerprint` section.

Helper functions
----------------

ProLIF comes with a set of functions that should help users analyse the results
produced by the package more easily. The documentation for these functions can
be found in the :ref:`source/modules/utils:Helper functions` section.
You can also find classes to plot the resulting interaction fingerprints in the
:ref:`source/modules/plotting:Plotting` section

Typing
------
Type aliases that are used throughout the code can be found in the
:ref:`source/modules/types:Custom types` section.