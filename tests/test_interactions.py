from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, cast

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.transformations import rotateby, translate
from rdkit import Chem, RDLogger

import prolif
from prolif.fingerprint import Fingerprint
from prolif.interactions import VdWContact
from prolif.interactions.base import _INTERACTIONS, Interaction
from prolif.interactions.constants import VDW_PRESETS
from prolif.interactions.utils import get_mapindex

# Disable RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from MDAnalysis.core.groups import AtomGroup
    from MDAnalysis.core.universe import Universe

    from prolif.molecule import Molecule
    from prolif.residue import Residue
    from prolif.typeshed import InteractionMetadata


@pytest.fixture(scope="module")
def benzene_universe() -> "Universe":
    benzene = mda.Universe(prolif.datafiles.datapath / "benzene.mol2")
    elements = mda.topology.guessers.guess_types(benzene.atoms.names)
    benzene.add_TopologyAttr("elements", elements)
    benzene.segments.segids = np.array(["U1"], dtype=object)
    benzene.transfer_to_memory()
    return benzene


@pytest.fixture(scope="module")
def interaction_instances() -> dict[str, Interaction]:
    return {
        name: cls()
        for name, cls in _INTERACTIONS.items()
        if name != "Interaction" and not name.startswith("_")
    }


@pytest.fixture(scope="session")
def any_mol(request: pytest.FixtureRequest) -> "Molecule":
    return cast("Molecule", request.getfixturevalue(request.param))


@pytest.fixture(scope="session")
def any_other_mol(request: pytest.FixtureRequest) -> "Molecule":
    return cast("Molecule", request.getfixturevalue(request.param))


@pytest.fixture(scope="module")
def interaction_qmol(
    request: pytest.FixtureRequest, interaction_instances: dict[str, Interaction]
) -> Chem.Mol | list[Chem.Mol]:
    int_name, parameter = cast(str, request.param).split(".")
    return cast(
        Chem.Mol | list[Chem.Mol], getattr(interaction_instances[int_name], parameter)
    )


class TestInteractions:
    @pytest.fixture(scope="class")
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("all")

    @pytest.mark.parametrize(
        ("func_name", "any_mol", "any_other_mol", "expected"),
        [
            ("cationic", "cation", "anion", True),
            ("cationic", "anion", "cation", False),
            ("cationic", "cation", "benzene", False),
            ("anionic", "cation", "anion", False),
            ("anionic", "anion", "cation", True),
            ("anionic", "anion", "benzene", False),
            ("cationpi", "cation", "benzene", True),
            ("cationpi", "cation_false", "benzene", False),
            ("cationpi", "benzene", "cation", False),
            ("cationpi", "cation", "cation", False),
            ("cationpi", "benzene", "benzene", False),
            ("pication", "benzene", "cation", True),
            ("pication", "benzene", "cation_false", False),
            ("pication", "cation", "benzene", False),
            ("pication", "cation", "cation", False),
            ("pication", "benzene", "benzene", False),
            ("pistacking", "benzene", "etf", True),
            ("pistacking", "etf", "benzene", True),
            ("pistacking", "ftf", "benzene", True),
            ("pistacking", "benzene", "ftf", True),
            ("facetoface", "benzene", "ftf", True),
            ("facetoface", "ftf", "benzene", True),
            ("facetoface", "benzene", "etf", False),
            ("facetoface", "etf", "benzene", False),
            ("edgetoface", "benzene", "etf", True),
            ("edgetoface", "etf", "benzene", True),
            ("edgetoface", "benzene", "ftf", False),
            ("edgetoface", "ftf", "benzene", False),
            ("hydrophobic", "benzene", "etf", True),
            ("hydrophobic", "benzene", "ftf", True),
            ("hydrophobic", "benzene", "chlorine", False),
            ("hydrophobic", "benzene", "bromine", True),
            ("hydrophobic", "benzene", "anion", False),
            ("hydrophobic", "benzene", "cation", False),
            ("hbdonor", "hb_donor", "hb_acceptor", True),
            ("hbdonor", "hb_donor", "hb_acceptor_false", False),
            ("hbdonor", "hb_acceptor", "hb_donor", False),
            ("hbacceptor", "hb_acceptor", "hb_donor", True),
            ("hbacceptor", "hb_acceptor_false", "hb_donor", False),
            ("hbacceptor", "hb_donor", "hb_acceptor", False),
            ("xbdonor", "xb_donor", "xb_acceptor", True),
            ("xbdonor", "xb_donor", "xb_acceptor_false_xar", False),
            ("xbdonor", "xb_donor", "xb_acceptor_false_axd", False),
            ("xbdonor", "xb_acceptor", "xb_donor", False),
            ("xbacceptor", "xb_acceptor", "xb_donor", True),
            ("xbacceptor", "xb_acceptor_false_xar", "xb_donor", False),
            ("xbacceptor", "xb_acceptor_false_axd", "xb_donor", False),
            ("xbacceptor", "xb_donor", "xb_acceptor", False),
            ("metaldonor", "metal", "ligand", True),
            ("metaldonor", "metal_false", "ligand", False),
            ("metaldonor", "ligand", "metal", False),
            ("metalacceptor", "ligand", "metal", True),
            ("metalacceptor", "ligand", "metal_false", False),
            ("metalacceptor", "metal", "ligand", False),
            ("vdwcontact", "benzene", "etf", True),
            ("vdwcontact", "hb_acceptor", "metal_false", False),
        ],
        indirect=["any_mol", "any_other_mol"],
    )
    def test_interaction(
        self,
        fingerprint: Fingerprint,
        func_name: str,
        any_mol: "Molecule",
        any_other_mol: "Molecule",
        expected: bool,
    ) -> None:
        interaction = getattr(fingerprint, func_name)
        assert next(interaction(any_mol[0], any_other_mol[0]), False) is expected

    @pytest.mark.usefixtures("cleanup_dummy")
    def test_warning_supersede(self) -> None:
        class Dummy(Interaction):
            @abstractmethod
            def detect(
                self, lig_res: "Residue", prot_res: "Residue"
            ) -> "Iterator[InteractionMetadata]":
                pass

        old = id(_INTERACTIONS["Dummy"])
        with pytest.warns(UserWarning, match="interaction has been superseded"):

            class Dummy(Interaction):  # type: ignore[no-redef]
                @abstractmethod
                def detect(
                    self, lig_res: "Residue", prot_res: "Residue"
                ) -> "Iterator[InteractionMetadata]":
                    pass

        new = id(_INTERACTIONS["Dummy"])
        assert old != new

    @pytest.mark.usefixtures("cleanup_dummy")
    def test_error_no_detect(self) -> None:
        class Dummy(Interaction):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class Dummy"):
            Fingerprint(["Dummy"])

    @pytest.mark.parametrize("index", [0, 1, 3, 42, 78])
    def test_get_mapindex(self, index: int, ligand_mol: "Molecule") -> None:
        parent_index = get_mapindex(ligand_mol[0], index)
        assert parent_index == index

    def test_vdwcontact_tolerance_error(self) -> None:
        with pytest.raises(ValueError, match="`tolerance` must be 0 or positive"):
            VdWContact(tolerance=-1)

    @pytest.mark.parametrize(
        ("any_mol", "any_other_mol"),
        [("benzene", "cation")],
        indirect=["any_mol", "any_other_mol"],
    )
    def test_vdwcontact_cache(
        self, any_mol: "Molecule", any_other_mol: "Molecule"
    ) -> None:
        vdw = VdWContact()
        assert vdw._vdw_cache == {}
        vdw.detect(any_mol[0], any_other_mol[0])
        for (lig, res), value in vdw._vdw_cache.items():
            vdw_dist = vdw.vdwradii[lig] + vdw.vdwradii[res] + vdw.tolerance
            assert vdw_dist == value

    @pytest.mark.parametrize(
        ("any_mol", "any_other_mol"),
        [("benzene", "cation")],
        indirect=["any_mol", "any_other_mol"],
    )
    def test_vdwcontact_vdwradii_update(
        self, any_mol: "Molecule", any_other_mol: "Molecule"
    ) -> None:
        vdw = VdWContact(vdwradii={"Na": 0})
        metadata = vdw.detect(any_mol[0], any_other_mol[0])
        assert next(metadata, None) is None

    @pytest.mark.parametrize(
        ("any_mol", "any_other_mol"),
        [("benzene", "cation")],
        indirect=["any_mol", "any_other_mol"],
    )
    @pytest.mark.parametrize("preset", ["mdanalysis", "rdkit", "csd"])
    def test_vdwcontact_preset(
        self,
        any_mol: "Molecule",
        any_other_mol: "Molecule",
        preset: 'Literal["mdanalysis", "rdkit", "csd"]',
    ) -> None:
        vdw = VdWContact(preset=preset)
        metadata = vdw.detect(any_mol[0], any_other_mol[0])
        assert next(metadata, None) is not None
        assert vdw.vdwradii == VDW_PRESETS[preset]

    def test_vdwcontact_radii_missing(self) -> None:
        vdw = VdWContact(preset="mdanalysis")
        with pytest.raises(
            ValueError, match=r"van der Waals radius for atom .+ not found"
        ):
            vdw._get_radii_sum("X", "Y")

    @pytest.mark.parametrize(
        ("interaction_qmol", "smiles", "expected"),
        [
            ("Hydrophobic.lig_pattern", "C", 0),
            ("Hydrophobic.lig_pattern", "C=[SH2]", 0),
            ("Hydrophobic.lig_pattern", "c1cscc1", 5),
            ("Hydrophobic.lig_pattern", "c1cocc1", 2),
            ("Hydrophobic.lig_pattern", "[*]SC", 1),
            ("Hydrophobic.lig_pattern", "[*]CC", 1),
            ("Hydrophobic.lig_pattern", "[*]C=C", 1),
            ("Hydrophobic.lig_pattern", "[*]=C=C", 1),
            ("Hydrophobic.lig_pattern", "[*]C(=C)C", 1),
            ("Hydrophobic.lig_pattern", "[*]C(C)C", 1),
            ("Hydrophobic.lig_pattern", "CS(C)(C)C", 0),
            ("Hydrophobic.lig_pattern", "FC(F)(F)F", 0),
            ("Hydrophobic.lig_pattern", "BrI", 2),
            ("Hydrophobic.lig_pattern", "C=O", 0),
            ("Hydrophobic.lig_pattern", "C=N", 0),
            ("Hydrophobic.lig_pattern", "CF", 0),
            ("Hydrophobic.lig_pattern", "Nc1ccccc1", 5),
            ("Hydrophobic.lig_pattern", "[*]C(C)(C)C", 1),
            ("HBAcceptor.prot_pattern", "[OH2]", 2),
            ("HBAcceptor.prot_pattern", "[NH3]", 3),
            ("HBAcceptor.prot_pattern", "[NH4+]", 4),
            ("HBAcceptor.prot_pattern", "[SH2]", 2),
            ("HBAcceptor.prot_pattern", "O=C=O", 0),
            ("HBAcceptor.prot_pattern", "c1c[nH+]ccc1", 0),
            ("HBAcceptor.prot_pattern", "c1c[nH]cc1", 1),
            ("HBAcceptor.prot_pattern", "c1c[nH+]c[nH]1", 2),
            ("HBAcceptor.lig_pattern", "O", 1),
            ("HBAcceptor.lig_pattern", "N", 1),
            ("HBAcceptor.lig_pattern", "[NH4+]", 0),
            ("HBAcceptor.lig_pattern", "N-C=O", 1),
            ("HBAcceptor.lig_pattern", "N-C=[SH2]", 0),
            ("HBAcceptor.lig_pattern", "[nH+]1ccccc1", 0),
            ("HBAcceptor.lig_pattern", "n1ccccc1", 1),
            ("HBAcceptor.lig_pattern", "n(C)1cccc1", 0),
            ("HBAcceptor.lig_pattern", "[nH]1cccc1", 0),
            ("HBAcceptor.lig_pattern", "[nH+]1nc[nH]c1", 0),
            ("HBAcceptor.lig_pattern", "c12c([nH]cc1)cccc2", 0),
            ("HBAcceptor.lig_pattern", "Nc1ccccc1", 0),
            ("HBAcceptor.lig_pattern", "C(=N)(N)N", 0),
            ("HBAcceptor.lig_pattern", "N#C", 1),
            ("HBAcceptor.lig_pattern", "o1cccc1", 1),
            ("HBAcceptor.lig_pattern", "[o+](C)1cccc1", 0),
            ("HBAcceptor.lig_pattern", "[*]-[N+](=O)-[O-]", 0),
            ("HBAcceptor.lig_pattern", "COC=O", 1),
            ("HBAcceptor.lig_pattern", "c1ccccc1Oc1ccccc1", 0),
            ("HBAcceptor.lig_pattern", "FC", 1),
            ("HBAcceptor.lig_pattern", "Fc1ccccc1", 1),
            ("HBAcceptor.lig_pattern", "FCF", 0),
            ("XBAcceptor.prot_pattern", "CCl", 1),
            ("XBAcceptor.prot_pattern", "c1ccccc1Cl", 1),
            ("XBAcceptor.prot_pattern", "NCl", 1),
            ("XBAcceptor.prot_pattern", "c1cccc[n+]1Cl", 1),
            ("XBAcceptor.lig_pattern", "[NH3]", 3),
            ("XBAcceptor.lig_pattern", "[NH+]C", 0),
            ("XBAcceptor.lig_pattern", "c1ccccc1", 12),
            ("XBAcceptor.lig_pattern", "C(=O)C", 1),
            ("XBAcceptor.lig_pattern", "C#N", 0),
            ("Cationic.lig_pattern", "[NH4+]", 1),
            ("Cationic.lig_pattern", "[Ca+2]", 1),
            ("Cationic.lig_pattern", "CC(=[NH2+])N", 2),
            ("Cationic.lig_pattern", "NC(=[NH2+])N", 3),
            ("Cationic.prot_pattern", "[Cl-]", 1),
            ("Cationic.prot_pattern", "CC(=O)[O-]", 2),
            ("Cationic.prot_pattern", "CS(=O)[O-]", 2),
            ("Cationic.prot_pattern", "CP(=O)[O-]", 2),
            ("CationPi.cation", "[NH4+]", 1),
            ("CationPi.cation", "[Ca+2]", 1),
            ("CationPi.cation", "CC(=[NH2+])N", 2),
            ("CationPi.cation", "NC(=[NH2+])N", 3),
            ("CationPi.pi_ring", "c1ccccc1", 1),
            ("CationPi.pi_ring", "c1cocc1", 1),
            ("EdgeToFace.pi_ring", "c1ccccc1", 1),
            ("EdgeToFace.pi_ring", "c1cocc1", 1),
            ("FaceToFace.pi_ring", "c1ccccc1", 1),
            ("FaceToFace.pi_ring", "c1cocc1", 1),
            ("MetalDonor.lig_pattern", "[Mg]", 1),
            ("MetalDonor.prot_pattern", "O", 1),
            ("MetalDonor.prot_pattern", "N", 1),
            ("MetalDonor.prot_pattern", "[NH+]", 0),
            ("MetalDonor.prot_pattern", "N-C=[SH2]", 0),
            ("MetalDonor.prot_pattern", "[nH+]1ccccc1", 0),
            ("MetalDonor.prot_pattern", "Nc1ccccc1", 0),
            ("MetalDonor.prot_pattern", "o1cccc1", 0),
            ("MetalDonor.prot_pattern", "COC=O", 2),
        ],
        indirect=["interaction_qmol"],
    )
    def test_smarts_matches(
        self, interaction_qmol: Chem.Mol | list[Chem.Mol], smiles: str, expected: int
    ) -> None:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if isinstance(interaction_qmol, list):
            n_matches = sum(
                len(mol.GetSubstructMatches(qmol)) for qmol in interaction_qmol
            )
        else:
            n_matches = len(mol.GetSubstructMatches(interaction_qmol))
        assert n_matches == expected

    @pytest.mark.parametrize(
        ("xyz", "rotation", "pi_type", "expected"),
        [
            ([0, 2.5, 4.0], [0, 0, 0], "facetoface", True),
            ([0, 3, 4.5], [0, 0, 0], "facetoface", False),
            ([0, 2, 4.5], [30, 0, 0], "facetoface", True),
            ([0, 2, 4.5], [150, 0, 0], "facetoface", True),
            ([0, 2, -4.5], [30, 0, 0], "facetoface", True),
            ([0, 2, -4.5], [150, 0, 0], "facetoface", True),
            ([1, 1.5, 3.5], [30, 15, 80], "facetoface", True),
            ([1, 2.5, 4.5], [30, 15, 65], "facetoface", True),
            ([0, 1.5, 4.5], [60, 0, 0], "edgetoface", True),
            ([0, 2, 5], [60, 0, 0], "edgetoface", True),
            ([0, 1.5, 4.5], [90, 0, 0], "edgetoface", True),
            ([0, 1.5, -4.5], [90, 0, 0], "edgetoface", True),
            ([0, 6, -0.5], [110, 0, 0], "edgetoface", True),
            ([0, 4.5, -0.5], [105, 0, 0], "edgetoface", True),
            ([0, 1.5, 4.5], [105, 0, 0], "edgetoface", False),
            ([0, 1.5, -4.5], [75, 0, 0], "edgetoface", False),
        ],
    )
    def test_pi_stacking(
        self,
        benzene_universe: "Universe",
        xyz: list[float],
        rotation: list[float],
        pi_type: str,
        expected: bool,
        fingerprint: Fingerprint,
    ) -> None:
        r1, r2 = self.create_rings(benzene_universe, xyz, rotation)

        def evaluate(pistacking_type: str, r1: "Molecule", r2: "Molecule") -> bool:
            return getattr(fingerprint, pistacking_type).any(r1, r2) or False

        assert evaluate(pi_type, r1, r2) is expected
        if expected is True:
            other = "edgetoface" if pi_type == "facetoface" else "facetoface"
            assert evaluate(other, r1, r2) is not expected
            assert evaluate("pistacking", r1, r2) is expected

    @staticmethod
    def create_rings(
        benzene_universe: "Universe", xyz: list[float], rotation: list[float]
    ) -> tuple["Molecule", "Molecule"]:
        r2 = benzene_universe.copy()
        r2.segments.segids = np.array(["U2"], dtype=object)
        tr = translate(xyz)
        rotx = rotateby(rotation[0], [1, 0, 0], ag=r2.atoms)
        roty = rotateby(rotation[1], [0, 1, 0], ag=r2.atoms)
        rotz = rotateby(rotation[2], [0, 0, 1], ag=r2.atoms)
        r2.trajectory.add_transformations(tr, rotx, roty, rotz)
        return (
            prolif.Molecule.from_mda(benzene_universe)[0],
            prolif.Molecule.from_mda(r2)[0],
        )

    def test_edgetoface_phe331(
        self, ligand_mol: "Molecule", protein_mol: "Molecule", fingerprint: Fingerprint
    ) -> None:
        lig, phe331 = ligand_mol[0], protein_mol["PHE331.B"]
        assert fingerprint.edgetoface.any(lig, phe331)  # type: ignore[attr-defined]
        assert not fingerprint.facetoface.any(lig, phe331)  # type: ignore[attr-defined]
        assert fingerprint.pistacking.any(lig, phe331)  # type: ignore[attr-defined]


class TestBridgedInteractions:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"order": 0}, "order must be greater than 0"),
            ({"order": 1, "min_order": 2}, "min_order cannot be greater than order"),
        ],
    )
    def test_water_bridge_validation(
        self,
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
        kwargs: dict,
        match: str,
    ) -> None:
        *_, water = water_atomgroups
        with pytest.raises(ValueError, match=match):
            Fingerprint(
                ["WaterBridge"],
                parameters={"WaterBridge": {"water": water, **kwargs}},
            )

    def test_direct_water_bridge(
        self,
        water_u: "Universe",
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
    ) -> None:
        ligand, protein, water = water_atomgroups
        fp = Fingerprint(["WaterBridge"], parameters={"WaterBridge": {"water": water}})
        fp.run(water_u.trajectory[:1], ligand, protein)
        int_data = next(fp.ifp[0].interactions())

        assert int_data.interaction == "WaterBridge"
        assert str(int_data.protein) == "TRP400.X"

    @pytest.mark.parametrize(
        ("kwargs", "num_expected"),
        [
            ({}, 3),
            ({"min_order": 2}, 2),
        ],
    )
    def test_higher_order_water_bridge(
        self,
        water_u: "Universe",
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
        kwargs: dict,
        num_expected: int,
    ) -> None:
        ligand, protein, water = water_atomgroups
        fp = Fingerprint(
            ["WaterBridge"],
            parameters={"WaterBridge": {"water": water, "order": 2, **kwargs}},
        )
        fp.run(water_u.trajectory[:1], ligand, protein)
        all_int_data = list(fp.ifp[0].interactions())

        assert len(all_int_data) == num_expected
        int_data = all_int_data[-1]
        assert "distance_TIP383.X_TIP317.X" in int_data.metadata

    def test_water_bridge_with_updating_atomgroup(
        self,
        water_u: "Universe",
        water_atomgroups: tuple["AtomGroup", "AtomGroup", "AtomGroup"],
    ) -> None:
        ligand, protein, water = water_atomgroups
        water = water_u.select_atoms(
            "segid WAT and byres around 4 (group ligand or group pocket)",
            ligand=ligand,
            pocket=protein,
            updating=True,
        )
        fp = Fingerprint(
            ["WaterBridge"],
            parameters={"WaterBridge": {"water": water, "order": 2}},
        )
        fp.run(water_u.trajectory[:1], ligand, protein)
        all_int_data = list(fp.ifp[0].interactions())

        assert len(all_int_data) == 3
        int_data = all_int_data[-1]
        assert "distance_TIP383.X_TIP317.X" in int_data.metadata

    def test_run_iter_water_bridge(
        self, water_mols: tuple["Molecule", "Molecule", "Molecule"]
    ) -> None:
        ligand, protein, water = water_mols
        fp = Fingerprint(["WaterBridge"], parameters={"WaterBridge": {"water": water}})
        # mimick multiple poses
        fp.run_from_iterable([ligand, ligand], protein)
        int_data = next(fp.ifp[1].interactions())

        assert int_data.interaction == "WaterBridge"
        assert str(int_data.protein) == "TRP400.X"

    def test_higher_order_run_iter_water_bridge(
        self, water_mols: tuple["Molecule", "Molecule", "Molecule"]
    ) -> None:
        ligand, protein, water = water_mols
        fp = Fingerprint(
            ["WaterBridge"], parameters={"WaterBridge": {"water": water, "order": 2}}
        )
        # mimick multiple poses
        fp.run_from_iterable([ligand, ligand], protein)
        all_int_data = list(fp.ifp[0].interactions())

        assert len(all_int_data) == 3
        int_data = all_int_data[-1]
        assert "distance_TIP383.X_TIP317.X" in int_data.metadata
