import pytest
from MDAnalysis import Universe
from rdkit import Chem

from prolif.datafiles import TOP, TRAJ, datapath
from prolif.molecule import Molecule, sdf_supplier


def pytest_sessionstart(session):
    if not datapath.exists():
        pytest.exit(
            f"Example data files are not accessible: {datapath!s} does not exist"
        )
    vina_path = datapath / "vina"
    if not vina_path.exists():
        pytest.exit(
            f"Example Vina data files are not accessible: {vina_path!s} does not exist"
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
def protein_ag(u, ligand_ag):
    return u.select_atoms("protein and byres around 6.5 group ligand", ligand=ligand_ag)


@pytest.fixture(scope="session")
def protein_rdkit(protein_ag):
    return protein_ag.convert_to.rdkit()


@pytest.fixture(scope="session")
def protein_mol(protein_ag):
    return Molecule.from_mda(protein_ag)


@pytest.fixture(scope="session")
def sdf_suppl():
    path = str(datapath / "vina" / "vina_output.sdf")
    return sdf_supplier(path)
