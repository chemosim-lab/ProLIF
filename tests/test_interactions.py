import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.transformations import rotateby, translate
from rdkit import Chem, RDLogger

import prolif
from prolif.fingerprint import Fingerprint
from prolif.interactions import VdWContact
from prolif.interactions.base import _INTERACTIONS, Interaction, get_mapindex

# disable rdkit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


@pytest.fixture(scope="module")
def benzene_universe():
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
        if name != "Interaction" and not name.startswith("_")
    }


@pytest.fixture(scope="session")
def any_mol(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def any_other_mol(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def interaction_qmol(request, interaction_instances):
    int_name, parameter = request.param.split(".")
    return getattr(interaction_instances[int_name], parameter)


class TestInteractions:
    @pytest.fixture(scope="class")
    def fingerprint(self):
        return Fingerprint()

    @pytest.mark.parametrize(
        "func_name, any_mol, any_other_mol, expected",
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
        self, fingerprint, func_name, any_mol, any_other_mol, expected
    ):
        interaction = getattr(fingerprint, func_name)
        assert next(interaction(any_mol[0], any_other_mol[0]), False) is expected

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
        "any_mol, any_other_mol",
        [("benzene", "cation")],
        indirect=["any_mol", "any_other_mol"],
    )
    def test_vdwcontact_cache(self, any_mol, any_other_mol):
        vdw = VdWContact()
        assert vdw._vdw_cache == {}
        vdw.detect(any_mol[0], any_other_mol[0])
        for (lig, res), value in vdw._vdw_cache.items():
            vdw_dist = vdw.vdwradii[lig] + vdw.vdwradii[res] + vdw.tolerance
            assert vdw_dist == value

    @pytest.mark.parametrize(
        "any_mol, any_other_mol",
        [("benzene", "cation")],
        indirect=["any_mol", "any_other_mol"],
    )
    def test_vdwcontact_vdwradii_update(self, any_mol, any_other_mol):
        vdw = VdWContact(vdwradii={"Na": 0})
        metadata = vdw.detect(any_mol[0], any_other_mol[0])
        assert next(metadata, None) is None

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
            ("HBAcceptor.prot_pattern", "[OH2]", 2),
            ("HBAcceptor.prot_pattern", "[NH3]", 3),
            ("HBAcceptor.prot_pattern", "[NH4+]", 4),
            ("HBAcceptor.prot_pattern", "[SH2]", 2),
            ("HBAcceptor.prot_pattern", "O=C=O", 0),
            ("HBAcceptor.prot_pattern", "c1c[nH+]ccc1", 0),
            ("HBAcceptor.prot_pattern", "c1c[nH]cc1", 1),
            ("HBAcceptor.lig_pattern", "O", 1),
            ("HBAcceptor.lig_pattern", "N", 1),
            ("HBAcceptor.lig_pattern", "[NH4+]", 0),
            ("HBAcceptor.lig_pattern", "N-C=O", 1),
            ("HBAcceptor.lig_pattern", "N-C=[SH2]", 0),
            ("HBAcceptor.lig_pattern", "[nH+]1ccccc1", 0),
            ("HBAcceptor.lig_pattern", "n1ccccc1", 1),
            ("HBAcceptor.lig_pattern", "Nc1ccccc1", 0),
            ("HBAcceptor.lig_pattern", "o1cccc1", 1),
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
    def test_pi_stacking(
        self, benzene_universe, xyz, rotation, pi_type, expected, fingerprint
    ):
        r1, r2 = self.create_rings(benzene_universe, xyz, rotation)
        evaluate = lambda pistacking_type, r1, r2: next(
            getattr(fingerprint, pistacking_type)(r1, r2), False
        )
        assert evaluate(pi_type, r1, r2) is expected
        if expected is True:
            other = "edgetoface" if pi_type == "facetoface" else "facetoface"
            assert evaluate(other, r1, r2) is not expected
            assert evaluate("pistacking", r1, r2) is expected

    @staticmethod
    def create_rings(benzene_universe, xyz, rotation):
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

    def test_edgetoface_phe331(self, ligand_mol, protein_mol):
        fp = Fingerprint()
        lig, phe331 = ligand_mol[0], protein_mol["PHE331.B"]
        assert next(fp.edgetoface(lig, phe331)) is True
        assert next(fp.pistacking(lig, phe331)) is True
