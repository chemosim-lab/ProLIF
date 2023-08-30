Installation
------------

Requirements:

* Python 3.8+
* `RDKit <https://www.rdkit.org/docs/>`_ (2021.03+)
* `MDAnalysis <https://www.mdanalysis.org/>`_ (2.2+)
* `Pandas <https://pandas.pydata.org/>`_ (1.0+)
* `NumPy <https://numpy.org/>`_
* `SciPy <https://www.scipy.org/scipylib/index.html>`_
* `tqdm <https://tqdm.github.io/>`_
* `multiprocess <https://multiprocess.readthedocs.io/>`_
* `dill <https://dill.readthedocs.io/>`_

The recommanded way to install ProLIF is to use `conda`_.

These first steps are optional and will create a separate environment for ProLIF::

    # create a separate virtual environment
    conda create -n prolif
    # activate it
    conda activate prolif

Then simply run the following command to install the library::

    conda install -c conda-forge prolif

Alternatively, you can install ProLIF with pip instead of conda::

    pip install rdkit prolif

To run the tutorials, you can install the optional dependencies with::

    pip install prolif[tutorials]


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html
