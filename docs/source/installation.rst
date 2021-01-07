Installation
------------

Requirements:

* Python 3.6+
* `RDKit <https://www.rdkit.org/docs/>`_ (2020.03+)
* `MDAnalysis <https://www.mdanalysis.org/>`_ (2.0+)
* `Pandas <https://pandas.pydata.org/>`_ (1.0+)
* `NumPy <https://numpy.org/>`_
* `SciPy <https://www.scipy.org/scipylib/index.html>`_
* `tqdm <https://tqdm.github.io/>`_

The simplest way to install ProLIF dependencies is to use `conda`_::

    # create a separate virtual environment
    conda create -n prolif
    # activate it
    conda activate prolif
    # install main dependencies
    conda config --add channels conda-forge
    conda install rdkit cython

We strongly encourage users to install ProLIF in a separate virtual environment, as it currently (and temporarily) depends on a custom fork of MDAnalysis.

The rest of the dependencies are automatically installed through pip when installing prolif::

    pip install git+https://github.com/chemosim-lab/ProLIF.git

Alternatively, you can install a specific release version as follow::

    pip install https://github.com/chemosim-lab/ProLIF/archive/v0.3.0.zip

.. note:: Until MDAnalysis version 2.0.0 is out, ProLIF can only be installed through our GitHub repository. Once MDAnalysis v2.0.0 is out, it will be made available as a standard PyPI package and installable with ``pip install prolif``.

.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html