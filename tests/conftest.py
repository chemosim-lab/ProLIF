import numpy as np
import pytest
from MDAnalysis import Universe
from MDAnalysis.topology.guessers import guess_atom_element
from numpy.testing import assert_array_almost_equal
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

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
    # ugly patch to add Mixin class as attribute to pytest so that we don't have to
    # worry about relative imports in the test codebase
    setattr(pytest, "BaseTestMixinRDKitMol", BaseTestMixinRDKitMol)


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


def from_mol2(f):
    path = str(datapath / f)
    u = Universe(path)
    elements = [guess_atom_element(n) for n in u.atoms.names]
    u.add_TopologyAttr("elements", np.array(elements, dtype=object))
    u.atoms.types = np.array([x.upper() for x in u.atoms.types], dtype=object)
    return Molecule.from_mda(u, force=True)


@pytest.fixture(scope="session")
def benzene():
    return from_mol2("benzene.mol2")


@pytest.fixture(scope="session")
def cation():
    return from_mol2("cation.mol2")


@pytest.fixture(scope="session")
def cation_false():
    return from_mol2("cation_false.mol2")


@pytest.fixture(scope="session")
def anion():
    return from_mol2("anion.mol2")


@pytest.fixture(scope="session")
def ftf():
    return from_mol2("facetoface.mol2")


@pytest.fixture(scope="session")
def etf():
    return from_mol2("edgetoface.mol2")


@pytest.fixture(scope="session")
def chlorine():
    return from_mol2("chlorine.mol2")


@pytest.fixture(scope="session")
def bromine():
    return from_mol2("bromine.mol2")


@pytest.fixture(scope="session")
def hb_donor():
    return from_mol2("donor.mol2")


@pytest.fixture(scope="session")
def hb_acceptor():
    return from_mol2("acceptor.mol2")


@pytest.fixture(scope="session")
def hb_acceptor_false():
    return from_mol2("acceptor_false.mol2")


@pytest.fixture(scope="session")
def xb_donor():
    return from_mol2("xbond_donor.mol2")


@pytest.fixture(scope="session")
def xb_acceptor():
    return from_mol2("xbond_acceptor.mol2")


@pytest.fixture(scope="session")
def xb_acceptor_false_xar():
    return from_mol2("xbond_acceptor_false_xar.mol2")


@pytest.fixture(scope="session")
def xb_acceptor_false_axd():
    return from_mol2("xbond_acceptor_false_axd.mol2")


@pytest.fixture(scope="session")
def ligand():
    return from_mol2("ligand.mol2")


@pytest.fixture(scope="session")
def metal():
    return from_mol2("metal.mol2")


@pytest.fixture(scope="session")
def metal_false():
    return from_mol2("metal_false.mol2")


class BaseTestMixinRDKitMol:
    def test_init(self, mol):
        assert isinstance(mol, Chem.Mol)

    def test_centroid(self, mol):
        expected = ComputeCentroid(mol.GetConformer())
        assert_array_almost_equal(mol.centroid, expected)

    def test_xyz(self, mol):
        expected = mol.GetConformer().GetPositions()
        assert_array_almost_equal(mol.xyz, expected)
