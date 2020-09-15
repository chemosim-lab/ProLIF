Installation
------------

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