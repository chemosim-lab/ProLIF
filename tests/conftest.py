from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pytest
from MDAnalysis import Universe
from MDAnalysis.topology.guessers import guess_atom_element
from numpy.testing import assert_array_almost_equal
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

from prolif.datafiles import TOP, TRAJ, datapath
from prolif.interactions.base import _INTERACTIONS
from prolif.molecule import Molecule, sdf_supplier

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup

    from prolif.molecule import BaseRDKitMol


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: ARG001
    if not datapath.exists():
        pytest.exit(
            f"Example data files are not accessible: {datapath!s} does not exist",
        )
    vina_path = datapath / "vina"
    if not vina_path.exists():
        pytest.exit(
            f"Example Vina data files are not accessible: {vina_path!s} does not exist",
        )
    # ugly patch to add Mixin class as attribute to pytest so that we don't have to
    # worry about relative imports in the test codebase
    pytest.BaseTestMixinRDKitMol = BaseTestMixinRDKitMol  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def u() -> Universe:
    return Universe(TOP, TRAJ)


@pytest.fixture(scope="session")
def rdkit_mol() -> Chem.Mol:
    return Chem.MolFromPDBFile(TOP, removeHs=False)


@pytest.fixture(scope="session")
def ligand_ag(u: Universe) -> "AtomGroup":
    return u.select_atoms("resname LIG")


@pytest.fixture(scope="session")
def ligand_rdkit(ligand_ag: "AtomGroup") -> Chem.Mol:
    return ligand_ag.convert_to.rdkit()  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def ligand_mol(ligand_ag: "AtomGroup") -> Molecule:
    return Molecule.from_mda(ligand_ag)


@pytest.fixture(scope="session")
def protein_ag(u: Universe, ligand_ag: "AtomGroup") -> "AtomGroup":
    return u.select_atoms("protein and byres around 6.5 group ligand", ligand=ligand_ag)


@pytest.fixture(scope="session")
def protein_rdkit(protein_ag: "AtomGroup") -> Chem.Mol:
    return protein_ag.convert_to.rdkit()  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def protein_mol(protein_ag: "AtomGroup") -> Molecule:
    return Molecule.from_mda(protein_ag)


@pytest.fixture(scope="session")
def sdf_suppl() -> sdf_supplier:
    path = str(datapath / "vina" / "vina_output.sdf")
    return sdf_supplier(path)


def from_mol2(filename: str) -> Molecule:
    path = str(datapath / filename)
    u = Universe(path)
    elements = [guess_atom_element(n) for n in u.atoms.names]
    u.add_TopologyAttr("elements", np.array(elements, dtype=object))
    u.atoms.types = np.array([x.upper() for x in u.atoms.types], dtype=object)
    return Molecule.from_mda(u, force=True)


@pytest.fixture(scope="session")
def benzene() -> Molecule:
    return from_mol2("benzene.mol2")


@pytest.fixture(scope="session")
def cation() -> Molecule:
    return from_mol2("cation.mol2")


@pytest.fixture(scope="session")
def cation_false() -> Molecule:
    return from_mol2("cation_false.mol2")


@pytest.fixture(scope="session")
def anion() -> Molecule:
    return from_mol2("anion.mol2")


@pytest.fixture(scope="session")
def ftf() -> Molecule:
    return from_mol2("facetoface.mol2")


@pytest.fixture(scope="session")
def etf() -> Molecule:
    return from_mol2("edgetoface.mol2")


@pytest.fixture(scope="session")
def chlorine() -> Molecule:
    return from_mol2("chlorine.mol2")


@pytest.fixture(scope="session")
def bromine() -> Molecule:
    return from_mol2("bromine.mol2")


@pytest.fixture(scope="session")
def hb_donor() -> Molecule:
    return from_mol2("donor.mol2")


@pytest.fixture(scope="session")
def hb_acceptor() -> Molecule:
    return from_mol2("acceptor.mol2")


@pytest.fixture(scope="session")
def hb_acceptor_false() -> Molecule:
    return from_mol2("acceptor_false.mol2")


@pytest.fixture(scope="session")
def xb_donor() -> Molecule:
    return from_mol2("xbond_donor.mol2")


@pytest.fixture(scope="session")
def xb_acceptor() -> Molecule:
    return from_mol2("xbond_acceptor.mol2")


@pytest.fixture(scope="session")
def xb_acceptor_false_xar() -> Molecule:
    return from_mol2("xbond_acceptor_false_xar.mol2")


@pytest.fixture(scope="session")
def xb_acceptor_false_axd() -> Molecule:
    return from_mol2("xbond_acceptor_false_axd.mol2")


@pytest.fixture(scope="session")
def ligand() -> Molecule:
    return from_mol2("ligand.mol2")


@pytest.fixture(scope="session")
def metal() -> Molecule:
    return from_mol2("metal.mol2")


@pytest.fixture(scope="session")
def metal_false() -> Molecule:
    return from_mol2("metal_false.mol2")


@pytest.fixture
def cleanup_dummy() -> Iterator[None]:
    yield
    _INTERACTIONS.pop("Dummy", None)


@pytest.fixture(scope="session")
def water_u() -> Universe:
    top_path = (datapath / "water_m2.pdb").as_posix()
    traj_path = (datapath / "water_m2.xtc").as_posix()
    return Universe(top_path, traj_path)


@pytest.fixture(scope="session")
def water_params(water_u: Universe) -> tuple["AtomGroup", "AtomGroup", "AtomGroup"]:
    ligand = water_u.select_atoms("resname QNB")
    protein = water_u.select_atoms(
        "protein and byres around 4 group ligand", ligand=ligand
    )
    water = water_u.select_atoms(
        "resname TIP3 and byres around 6 (group ligand or group pocket)",
        ligand=ligand,
        pocket=protein,
    )
    return ligand, protein, water


class BaseTestMixinRDKitMol:
    def test_init(self, mol: "BaseRDKitMol") -> None:
        assert isinstance(mol, Chem.Mol)

    def test_centroid(self, mol: "BaseRDKitMol") -> None:
        expected = ComputeCentroid(mol.GetConformer())
        assert_array_almost_equal(mol.centroid, expected)

    def test_xyz(self, mol: "BaseRDKitMol") -> None:
        expected = mol.GetConformer().GetPositions()
        assert_array_almost_equal(mol.xyz, expected)
