Tutorials
=========

In this section you will learn how to generate an interaction fingerprint for different
scenarios, how to extract information from the fingerprint data, and how to visualize
it.

You can get a very brief overview of how ProLIF is implemented in the
:ref:`source/api:Summary` section of this documentation, with links to the different
modules for a more in-depth explanation.

All the examples here showcase a transmembrane protein (GPCR family) in complex with a
small molecule. For convenience, the different files used for this tutorial are included
with the ProLIF installation,
and you can access these through either:

- ``prolif.datafiles.TOP`` and ``prolif.datafiles.TRAJ`` for the topology (PDB file) and
  trajectory (XTC file) used in the molecular dynamics tutorials.
- ``prolif.datafiles.datapath``, which points to the directory containing all the other
  tutorial files. This is a :class:`pathlib.Path` object which offers a convenient way
  to navigate filesystems as shown in their
  `basic use <https://docs.python.org/3/library/pathlib.html#basic-use>`__ section.

.. warning::
    Outside of the tutorials, remember to switch any reference to ``prolif.datafiles``
    with the actual paths to your inputs.

.. tip::
    At the top of each tutorial's page you can find links to either download the
    notebook or run it in Google Colab. You can install the dependencies for the
    tutorials with the command::
      
      pip install prolif[tutorials]


Molecular dynamics
------------------

There are two tutorial notebooks for MD simulations, depending on the type of components
that are being analyzed:

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
