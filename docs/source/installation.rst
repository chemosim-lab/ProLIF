Installation
------------

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

You can also install the development version of ProLIF from GitHub. This will contain
the latest features and bugfixes but may also be unstable so use at your own risk::

    pip install git+https://github.com/chemosim-lab/ProLIF.git


To install a development environment for contributing to ProLIF, please check our
`contributing guide <https://github.com/chemosim-lab/ProLIF/blob/master/CONTRIBUTING.md>`_
on GitHub.


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/index.html
