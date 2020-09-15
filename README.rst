ProLIF
======

|pypi-version| |build-status| |docs| |license|

.. |pypi-version| image:: https://img.shields.io/pypi/v/prolif.svg
   :target: https://pypi.python.org/pypi/prolif
   :alt: Pypi Version

.. |build-status| image:: https://img.shields.io/travis/chemosim-lab/ProLIF
    :alt: Travis build status

.. |license| image:: https://img.shields.io/pypi/l/prolif
    :alt: License

.. |docs| image:: https://img.shields.io/readthedocs/prolif
    :alt: Documentation Status

Description
-----------

ProLIF (*Protein-Ligand Interaction Fingerprints*) is a tool designed to generate interaction fingerprints for protein-ligand interactions extracted from molecular dynamics trajectories and docking simulations.

Installing dependencies
-----------------------

Requirements
""""""""""""

* Python 3.6+
* `RDKit <https://www.rdkit.org/docs/>`_ (2020.03+)
* `MDAnalysis <https://www.mdanalysis.org/>`_ (2.0+)
* `Pandas <https://pandas.pydata.org/>`_ (1.0+)
* `NumPy <https://numpy.org/>`_
* `tqdm <https://tqdm.github.io/>`_

.. note::
    | RDKit needs to be installed through `conda`_
    | Once conda is installed, run the following command:
    | ``conda install -c conda-forge rdkit``  

The rest of the dependencies are automatically installed through pip when installing prolif::

    pip install prolif

Alternatively, you can install the latest development version::

    pip install git+https://github.com/chemosim-lab/ProLIF.git@master

.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html

Quickstart guide
----------------

TODO

License
-------

Unless otherwise noted, all files in this directory and all subdirectories are distributed under the Apache License, Version 2.0 ::

    Copyright 2017-2020 Cédric BOUYSSET

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
