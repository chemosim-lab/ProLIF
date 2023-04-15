import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.transformations import rotateby, translate
from rdkit import Chem, RDLogger

import prolif
from prolif.fingerprint import Fingerprint
from prolif.interactions import _INTERACTIONS, Interaction, VdWContact, get_mapindex

from . import mol2factory

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


@pytest.fixture(scope="module")
def benzene():
    benzene = mda.Universe(prolif.datafiles.datapath / "benzene.mol2")
    elements = mda.topology.guessers.guess_types(benzene.atoms.names)
    benzene.add_TopologyAttr("elements", elements)
    benzene.segments.segids = np.array(["U1"], dtype=object)
    benzene.transfer_to_memory()
    return benzene


@pytest.fixture(scope="module")
def interaction_instances():
    return {
        name: cls()
        for name, cls in _INTERACTIONS.items()
        if name not in ["Interaction", "_Distance"]
    }


@pytest.fixture(scope="module")
def lig_mol(request):
    return getattr(mol2factory, request.param)()


@pytest.fixture(scope="module")
def prot_mol(request):
    return getattr(mol2factory, request.param)()


@pytest.fixture(scope="module")
def interaction_qmol(request, interaction_instances):
    int_name, parameter = request.param.split(".")
    return getattr(interaction_instances[int_name], parameter)


class TestInteractions:
    @pytest.fixture(scope="class")
    def fingerprint(self):
        return Fingerprint()

    @pytest.mark.parametrize(
        "func_name, lig_mol, prot_mol, expected",
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
        indirect=["lig_mol", "prot_mol"],
    )
    def test_interaction(self, fingerprint, func_name, lig_mol, prot_mol, expected):
        interaction = getattr(fingerprint, func_name)
        assert interaction(lig_mol[0], prot_mol[0]) is expected

    def test_warning_supersede(self):
        old = id(_INTERACTIONS["Hydrophobic"])
        with pytest.warns(UserWarning, match="interaction has been superseded"):

            class Hydrophobic(Interaction):
                def detect(self):
                    pass

        new = id(_INTERACTIONS["Hydrophobic"])
        assert old != new
        # fix dummy Hydrophobic class being reused in later unrelated tests

        class Hydrophobic(prolif.interactions.Hydrophobic):
            __doc__ = prolif.interactions.Hydrophobic.__doc__

    def test_error_no_detect(self):
        with pytest.raises(
            TypeError, match="Can't instantiate interaction class _Dummy"
        ):

            class _Dummy(Interaction):
                pass

    @pytest.mark.parametrize("index", [0, 1, 3, 42, 78])
    def test_get_mapindex(self, index, ligand_mol):
        parent_index = get_mapindex(ligand_mol[0], index)
        assert parent_index == index

    def test_vdwcontact_tolerance_error(self):
        with pytest.raises(ValueError, match="`tolerance` must be 0 or positive"):
            VdWContact(tolerance=-1)

    @pytest.mark.parametrize(
        "lig_mol, prot_mol", [("benzene", "cation")], indirect=["lig_mol", "prot_mol"]
    )
    def test_vdwcontact_cache(self, lig_mol, prot_mol):
        vdw = VdWContact()
        assert vdw._vdw_cache == {}
        vdw.detect(lig_mol[0], prot_mol[0])
        for (lig, res), value in vdw._vdw_cache.items():
            vdw_dist = vdw.vdwradii[lig] + vdw.vdwradii[res] + vdw.tolerance
            assert vdw_dist == value

    @pytest.mark.parametrize(
        "lig_mol, prot_mol", [("benzene", "cation")], indirect=["lig_mol", "prot_mol"]
    )
    def test_vdwcontact_vdwradii_update(self, lig_mol, prot_mol):
        class CustomVdW(VdWContact):
            def __init__(self, tolerance=0, vdwradii={"Na": 0}):
                super().__init__(tolerance, vdwradii)

        metadata = CustomVdW().detect(lig_mol[0], prot_mol[0])
        assert metadata is None
        _INTERACTIONS.pop("CustomVdW")

    @pytest.mark.parametrize(
        ["interaction_qmol", "smiles", "expected"],
        [
            ("Hydrophobic.lig_pattern", "C", 1),
            ("Hydrophobic.lig_pattern", "C=[SH2]", 1),
            ("Hydrophobic.lig_pattern", "c1cscc1", 5),
            ("Hydrophobic.lig_pattern", "CSC", 3),
            ("Hydrophobic.lig_pattern", "CS(C)(C)C", 4),
            ("Hydrophobic.lig_pattern", "FC(F)(F)F", 0),
            ("Hydrophobic.lig_pattern", "BrI", 2),
            ("Hydrophobic.lig_pattern", "C=O", 0),
            ("Hydrophobic.lig_pattern", "C=N", 0),
            ("Hydrophobic.lig_pattern", "CF", 0),
            ("_BaseHBond.donor", "[OH2]", 2),
            ("_BaseHBond.donor", "[NH3]", 3),
            ("_BaseHBond.donor", "[NH4+]", 4),
            ("_BaseHBond.donor", "[SH2]", 2),
            ("_BaseHBond.donor", "O=C=O", 0),
            ("_BaseHBond.donor", "c1c[nH+]ccc1", 0),
            ("_BaseHBond.donor", "c1c[nH]cc1", 1),
            ("_BaseHBond.acceptor", "O", 1),
            ("_BaseHBond.acceptor", "N", 1),
            ("_BaseHBond.acceptor", "[NH4+]", 0),
            ("_BaseHBond.acceptor", "N-C=O", 1),
            ("_BaseHBond.acceptor", "N-C=[SH2]", 0),
            ("_BaseHBond.acceptor", "[nH+]1ccccc1", 0),
            ("_BaseHBond.acceptor", "n1ccccc1", 1),
            ("_BaseHBond.acceptor", "Nc1ccccc1", 0),
            ("_BaseHBond.acceptor", "o1cccc1", 1),
            ("_BaseHBond.acceptor", "COC=O", 1),
            ("_BaseHBond.acceptor", "c1ccccc1Oc1ccccc1", 0),
            ("_BaseHBond.acceptor", "FC", 1),
            ("_BaseHBond.acceptor", "Fc1ccccc1", 1),
            ("_BaseHBond.acceptor", "FCF", 0),
            ("_BaseXBond.donor", "CCl", 1),
            ("_BaseXBond.donor", "c1ccccc1Cl", 1),
            ("_BaseXBond.donor", "NCl", 1),
            ("_BaseXBond.donor", "c1cccc[n+]1Cl", 1),
            ("_BaseXBond.acceptor", "[NH3]", 3),
            ("_BaseXBond.acceptor", "[NH+]C", 0),
            ("_BaseXBond.acceptor", "c1ccccc1", 12),
            ("Cationic.lig_pattern", "[NH4+]", 1),
            ("Cationic.lig_pattern", "[Ca+2]", 1),
            ("Cationic.lig_pattern", "CC(=[NH2+])N", 2),
            ("Cationic.lig_pattern", "NC(=[NH2+])N", 3),
            ("Cationic.prot_pattern", "[Cl-]", 1),
            ("Cationic.prot_pattern", "CC(=O)[O-]", 2),
            ("Cationic.prot_pattern", "CS(=O)[O-]", 2),
            ("Cationic.prot_pattern", "CP(=O)[O-]", 2),
            ("_BaseCationPi.cation", "[NH4+]", 1),
            ("_BaseCationPi.cation", "[Ca+2]", 1),
            ("_BaseCationPi.cation", "CC(=[NH2+])N", 2),
            ("_BaseCationPi.cation", "NC(=[NH2+])N", 3),
            ("_BaseCationPi.pi_ring", "c1ccccc1", 1),
            ("_BaseCationPi.pi_ring", "c1cocc1", 1),
            ("EdgeToFace.pi_ring", "c1ccccc1", 1),
            ("EdgeToFace.pi_ring", "c1cocc1", 1),
            ("FaceToFace.pi_ring", "c1ccccc1", 1),
            ("FaceToFace.pi_ring", "c1cocc1", 1),
            ("_BaseMetallic.lig_pattern", "[Mg]", 1),
            ("_BaseMetallic.prot_pattern", "O", 1),
            ("_BaseMetallic.prot_pattern", "N", 1),
            ("_BaseMetallic.prot_pattern", "[NH+]", 0),
            ("_BaseMetallic.prot_pattern", "N-C=[SH2]", 0),
            ("_BaseMetallic.prot_pattern", "[nH+]1ccccc1", 0),
            ("_BaseMetallic.prot_pattern", "Nc1ccccc1", 0),
            ("_BaseMetallic.prot_pattern", "o1cccc1", 0),
            ("_BaseMetallic.prot_pattern", "COC=O", 2),
        ],
        indirect=["interaction_qmol"],
    )
    def test_smarts_matches(self, interaction_qmol, smiles, expected):
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
        ["xyz", "rotation", "pi_type", "expected"],
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
    def test_pi_stacking(self, benzene, xyz, rotation, pi_type, expected, fingerprint):
        r1, r2 = self.create_rings(benzene, xyz, rotation)
        assert getattr(fingerprint, pi_type)(r1, r2) is expected
        if expected is True:
            other = "edgetoface" if pi_type == "facetoface" else "facetoface"
            assert getattr(fingerprint, other)(r1, r2) is not expected
            assert getattr(fingerprint, "pistacking")(r1, r2) is expected

    @staticmethod
    def create_rings(benzene, xyz, rotation):
        r2 = benzene.copy()
        r2.segments.segids = np.array(["U2"], dtype=object)
        tr = translate(xyz)
        rotx = rotateby(rotation[0], [1, 0, 0], ag=r2.atoms)
        roty = rotateby(rotation[1], [0, 1, 0], ag=r2.atoms)
        rotz = rotateby(rotation[2], [0, 0, 1], ag=r2.atoms)
        r2.trajectory.add_transformations(tr, rotx, roty, rotz)
        return prolif.Molecule.from_mda(benzene)[0], prolif.Molecule.from_mda(r2)[0]

    def test_edgetoface_phe331(self, ligand_mol, protein_mol):
        fp = Fingerprint()
        lig, phe331 = ligand_mol[0], protein_mol["PHE331.B"]
        assert fp.edgetoface(lig, phe331) is True
        assert fp.pistacking(lig, phe331) is True

    def test_copy_parameters(self):
        class Dummy1(VdWContact):
            def __init__(self, tolerance=1):
                super().__init__(tolerance)

        class Dummy2(Dummy1):
            def __init__(self, tolerance=2):
                super().__init__(tolerance)

        assert Dummy1().tolerance == 1
        assert Dummy2().tolerance == 2

        DummyUpdated = Dummy2.update_parameters(Dummy1)
        assert DummyUpdated().tolerance == 1
        assert _INTERACTIONS["Dummy2"] is DummyUpdated

        DummyUpdated = Dummy1.update_parameters(Dummy2)
        assert DummyUpdated().tolerance == 2
        assert _INTERACTIONS["Dummy1"] is DummyUpdated

        for name in ["Dummy1", "Dummy2"]:
            _INTERACTIONS.pop(name)
