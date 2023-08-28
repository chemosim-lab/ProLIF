Tutorials
=========

In this section you will learn how to generate an interaction fingerprint for different
scenarios, how to extract information from the fingerprint data, and how to visualize
it.

You can get a very brief overview of how ProLIF is implemented in the
:ref:`source/api:Summary` section of this documentation, with links to the different
modules for a more in-depth explanation.

All the examples here showcase a GPCR in complex with a small molecule. For convenience,
the different files used for this tutorial are included with the ProLIF installation,
and you can access these through either:

- `prolif.datafiles.TOP` and `prolif.datafiles.TRAJ` for the topology (PDB file) and
  trajectory (XTC file) used in the molecular dynamics tutorials.
- `prolif.datafiles.datapath`, which points to the directory containing all the other
  tutorial files. This is a `pathlib.Path` object which offers a convenient way to
  navigate filesystems as shown in their :ref:`python:pathlib/basic-use` section.

.. warning::
    Remember to switch any `prolif.datafiles` reference with the actual paths to your
    inputs outside of the tutorials.


Molecular dynamics
------------------

There are two notebooks for this, depending on the type of components that are being
analyzed:

- :ref:`notebooks/md-ligand-protein:Ligand-protein MD`
.. - :ref:`notebooks/md-protein-protein:Protein-protein MD`

Docking
-------

TODO

PDB file
--------

TODO

Advanced usage
--------------

TODO:
- interaction parameters
- custom interaction
- fp.ifp explanation
