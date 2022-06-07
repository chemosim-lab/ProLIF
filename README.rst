ProLIF
======

.. list-table::
    :widths: 12 35

    * - **Documentation**
      - |docs|
    * - **Tutorials**
      - |binder|
    * - **CI**
      - |tests| |codecov| |lgtm|
    * - **PyPI**
      - |pypi-version| |build|
    * - **Dependencies**
      - |mdanalysis| |rdkit|
    * - **License**
      - |license|

Description
-----------

ProLIF (*Protein-Ligand Interaction Fingerprints*) is a tool designed to generate
interaction fingerprints for complexes made of ligands, protein, DNA or RNA molecules
extracted from molecular dynamics trajectories, docking simulations and experimental
structures.

You can try it out prior to any installation on `Binder <https://mybinder.org/v2/gh/chemosim-lab/ProLIF/HEAD?filepath=docs%2Fnotebooks>`_.

Documentation
-------------

The installation instructions, documentation and tutorials can be found online on `ReadTheDocs <https://prolif.readthedocs.io/en/latest/>`_.

Issues
------

If you have found a bug, please open an issue on the `GitHub Issues <https://github.com/chemosim-lab/ProLIF/issues>`_ page.

Discussion
----------

If you have questions on how to use ProLIF, or if you want to give feedback or share ideas and new features, please head to the `GitHub Discussions <https://github.com/chemosim-lab/ProLIF/discussions>`_ page.

Citing ProLIF
-------------

Please refer to the `citation page <https://prolif.readthedocs.io/en/latest/source/citation.html>`_ on the documentation.

License
-------

Unless otherwise noted, all files in this directory and all subdirectories are distributed under the Apache License, Version 2.0 ::

    Copyright 2017-2022 Cédric BOUYSSET

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


.. |pypi-version| image:: https://img.shields.io/pypi/v/prolif.svg
   :target: https://pypi.python.org/pypi/prolif
   :alt: Pypi Version

.. |build| image:: https://github.com/chemosim-lab/ProLIF/workflows/build/badge.svg
    :target: https://github.com/chemosim-lab/ProLIF/actions?query=workflow%3Abuild
    :alt: Build status

.. |tests| image:: https://github.com/chemosim-lab/ProLIF/workflows/tests/badge.svg?branch=master
    :target: https://github.com/chemosim-lab/ProLIF/actions?query=workflow%3Atests
    :alt: Tests status

.. |codecov| image:: https://codecov.io/gh/chemosim-lab/ProLIF/branch/master/graph/badge.svg?token=2FCHV08G8A
    :target: https://codecov.io/gh/chemosim-lab/ProLIF

.. |docs| image:: https://readthedocs.org/projects/prolif/badge/?version=latest
    :target: https://prolif.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |lgtm| image:: https://img.shields.io/lgtm/grade/python/g/chemosim-lab/ProLIF.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/chemosim-lab/ProLIF/context:python
    :alt: Code quality

.. |license| image:: https://img.shields.io/pypi/l/prolif
    :target: http://www.apache.org/licenses/LICENSE-2.0
    :alt: License

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/chemosim-lab/ProLIF/HEAD?filepath=docs%2Fnotebooks
    :alt: Try it on binder

.. |mdanalysis| image:: https://img.shields.io/badge/Powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org

.. |rdkit| image:: https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC
      :alt: Powered by RDKit
      :target: https://www.rdkit.org/
