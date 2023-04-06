from pathlib import Path

import pytest
from MDAnalysis import Universe
from rdkit import Chem

from prolif.datafiles import TOP, TRAJ
from prolif.molecule import Molecule


def pytest_sessionstart(session):
    example_file = Path(TOP)
    if example_file.exists():
        print(f"Example data files found in {example_file!s}, proceeding with tests")
    else:
        pytest.exit(
            f"Example data files are not accessible: {example_file!s} does not exist"
        )


@pytest.fixture(scope="session")
def u():
    return Universe(TOP, TRAJ)


@pytest.fixture(scope="session")
def rdkit_mol():
    return Chem.MolFromPDBFile(TOP, removeHs=False)


@pytest.fixture(scope="session")
def ligand_ag(u):
    return u.select_atoms("resname LIG")


@pytest.fixture(scope="session")
def ligand_rdkit(ligand_ag):
    return ligand_ag.convert_to.rdkit()


@pytest.fixture(scope="session")
def ligand_mol(ligand_ag):
    return Molecule.from_mda(ligand_ag)


@pytest.fixture(scope="session")
def protein_ag(u):
    return u.select_atoms("protein")


@pytest.fixture(scope="session")
def protein_rdkit(protein_ag):
    return protein_ag.convert_to.rdkit()


@pytest.fixture(scope="session")
def protein_mol(protein_ag):
    return Molecule.from_mda(protein_ag)
