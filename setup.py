from setuptools import setup
import versioneer
import re
import os

GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", False)

# manually check RDKit version
try:
    from rdkit import __version__ as rdkit_version
except ImportError:
    if not GITHUB_ACTIONS:
        raise ImportError("ProLIF requires RDKit but it is not installed")
else:
    if re.match(r"^20[0-1][0-9]\.", rdkit_version):
        raise ValueError("ProLIF requires a version of RDKit >= 2020")

setup(version=versioneer.get_version())