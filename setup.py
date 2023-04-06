import os
import re

from setuptools import setup

GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", False)
READTHEDOCS = os.environ.get("READTHEDOCS", False)

# manually check RDKit version
try:
    from rdkit import __version__ as rdkit_version
except ImportError:
    if not (GITHUB_ACTIONS or READTHEDOCS):
        raise ImportError("ProLIF requires RDKit but it is not installed")
else:
    if re.match(r"^(20[0-1][0-9])|(2020)", rdkit_version):
        raise ValueError("ProLIF requires a version of RDKit >= 2021")

setup()
