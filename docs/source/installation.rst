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

RDKit needs to be installed through `conda`_
Once conda is installed, run the following command::

    conda install -c conda-forge rdkit cython

The rest of the dependencies are automatically installed through pip when installing prolif::

    pip install git+https://github.com/chemosim-lab/ProLIF.git

Alternatively, you can install the latest development version::

    pip install git+https://github.com/chemosim-lab/ProLIF.git@dev

.. note:: Until MDAnalysis version 2.0.0 is out, ProLIF can only be installed through this GitHub repository. Once v2.0.0 is out, it will be made available as a standard PyPI package and installable with ``pip install prolif``.

.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html