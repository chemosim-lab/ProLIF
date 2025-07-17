"""
Detecting interactions between residues --- :mod:`prolif.interactions.interactions`
===================================================================================

Note that some of the SMARTS patterns used in the interaction classes are inspired from
`Pharmit`_ and `RDKit`_.

.. _Pharmit: https://sourceforge.net/p/pharmit/code/ci/master/tree/src/pharmarec.cpp
.. _RDKit: https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef

"""

from collections.abc import Iterator
from itertools import product
from math import degrees, radians
from typing import TYPE_CHECKING, Literal

from rdkit import Geometry
from rdkit.Chem import MolFromSmarts

from prolif.interactions.base import (
    BasePiStacking,
    Distance,
    DoubleAngle,
    Interaction,
    SingleAngle,
)
from prolif.interactions.constants import VDW_PRESETS, VDWRADII  # noqa
from prolif.utils import angle_between_limits, get_centroid, get_ring_normal_vector

if TYPE_CHECKING:
    from prolif.residue import Residue
    from prolif.typeshed import Angles, InteractionMetadata

__all__ = [
    "Anionic",
    "CationPi",
    "Cationic",
    "EdgeToFace",
    "FaceToFace",
    "HBAcceptor",
    "HBDonor",
    "Hydrophobic",
    "MetalAcceptor",
    "MetalDonor",
    "PiCation",
    "PiStacking",
    "VdWContact",
    "XBAcceptor",
    "XBDonor",
]


class Hydrophobic(Distance):
    """Hydrophobic interaction

    Parameters
    ----------
    hydrophobic : str
        SMARTS query for hydrophobic atoms
    distance : float
        Distance threshold for the interaction


    .. versionchanged:: 1.1.0
        The initial SMARTS pattern was too broad.

    .. versionchanged:: 2.1.0
        Properly excluded all carbon linked to nitrogen/oxygen/fluoride from being
        hydrophobic, previous versions were allowing such carbons if they were aromatic.
        Fixed patterns taken from RDKit that were not made for mols with explicit H.
    """

    def __init__(
        self,
        hydrophobic: str = (
            "[c,s,Br,I,S&H0&v2"
            # equivalent to RDKit's ChainTwoWayAttach with explicit H support
            ",$([C&R0;$([CH0](=*)=*),$([CH1](=*)-[!#1]),$([CH2](-[!#1])-[!#1])])"
            # equivalent to RDKit's ThreeWayAttach
            ",$([C;$([CH0](=*)(-[!#1])-[!#1]),$([CH1](-[!#1])(-[!#1])-[!#1])])"
            # tButyl
            ",$([C&D4!R](-[CH3])(-[CH3])-[CH3])"
            # not carbon connected to N/O/F and not charged
            ";!$([#6]~[#7,#8,#9]);+0]"
        ),
        distance: float = 4.5,
    ) -> None:
        super().__init__(
            lig_pattern=hydrophobic,
            prot_pattern=hydrophobic,
            distance=distance,
        )


class HBAcceptor(SingleAngle):
    """Hbond interaction between a ligand (acceptor) and a residue (donor)

    Parameters
    ----------
    acceptor : str
        SMARTS for ``[Acceptor]``
    donor : str
        SMARTS for ``[Donor]-[Hydrogen]``
    distance : float
        Distance threshold between the acceptor and donor atoms
    DHA_angle : tuple
        Min and max values for the ``[Acceptor]...[Hydrogen]-[Donor]`` angle


    .. versionchanged:: 1.1.0
        The initial SMARTS pattern was too broad.

    .. versionchanged:: 2.0.0
        ``angles`` parameter renamed to ``DHA_angle``.

    .. versionchanged:: 2.1.0
        Removed charged aromatic nitrogen, triazolium, and guanidine/anidine-like from
        acceptors. Added charged nitrogen from histidine as donor.

    """

    def __init__(
        self,
        acceptor: str = (
            "[$([N&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4+1])&!$(N=C(-[C,N])-N)])"
            ",$([n+0&!X3&!$([n&r5]:[n+&r5])])"
            ",$([O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O)])"
            ",$([o+0])"
            ",$([F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])])]"
        ),
        donor: str = "[$([O,S,#7;+0]),$([Nv4+1]),$([n+]c[nH])]-[H]",
        distance: float = 3.5,
        DHA_angle: "Angles" = (130, 180),
    ) -> None:
        super().__init__(
            lig_pattern=acceptor,
            prot_pattern=donor,
            distance=distance,
            angle=DHA_angle,
            distance_atom="P1",
            metadata_mapping={"angle": "DHA_angle"},
        )


HBDonor = HBAcceptor.invert_role(
    "HBDonor",
    "Hbond interaction between a ligand (donor) and a residue (acceptor)",
)


class XBAcceptor(DoubleAngle):
    """Halogen bonding between a ligand (acceptor) and a residue (donor).

    Parameters
    ----------
    acceptor : str
        SMARTS for ``[Acceptor]-[R]``
    donor : str
        SMARTS for ``[Donor]-[Halogen]``
    distance : float
        Cutoff distance between the acceptor and halogen atoms
    AXD_angle : tuple
        Min and max values for the ``[Acceptor]...[Halogen]-[Donor]`` angle
    XAR_angle : tuple
        Min and max values for the ``[Halogen]...[Acceptor]-[R]`` angle

    Notes
    -----
    Distance and angle adapted from Auffinger et al. PNAS 2004


    .. versionchanged:: 2.0.0
        ``axd_angles`` and ``xar_angles`` parameters renamed to ``AXD_angle`` and
        ``XAR_angle``.

    .. versionchanged:: 2.0.3
        Fixed the SMARTS pattern for acceptors that was not allowing carbonyles and
        other groups with double bonds to match.

    """

    def __init__(
        self,
        acceptor: str = "[#7,#8,P,S,Se,Te,a;!+{1-}]!#[*]",
        donor: str = "[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]",
        distance: float = 3.5,
        AXD_angle: "Angles" = (130, 180),
        XAR_angle: "Angles" = (80, 140),
    ) -> None:
        super().__init__(
            lig_pattern=acceptor,
            prot_pattern=donor,
            distance=distance,
            L1P2P1_angle=AXD_angle,
            L2L1P2_angle=XAR_angle,
            distance_atoms=("L1", "P2"),
            metadata_mapping={
                "L1P2P1_angle": "AXD_angle",
                "L2L1P2_angle": "XAR_angle",
            },
        )


XBDonor = XBAcceptor.invert_role(
    "XBDonor",
    "Halogen bonding between a ligand (donor) and a residue (acceptor)",
)


class Cationic(Distance):
    """Ionic interaction between a ligand (cation) and a residue (anion)

    .. versionchanged:: 1.1.0
        Handles resonance forms for common acids, amidine and guanidine.

    """

    def __init__(
        self,
        cation: str = "[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
        anion: str = "[-{1-},$(O=[C,S,P]-[O-])]",
        distance: float = 4.5,
    ) -> None:
        super().__init__(lig_pattern=cation, prot_pattern=anion, distance=distance)


Anionic = Cationic.invert_role(
    "Anionic",
    "Ionic interaction between a ligand (anion) and a residue (cation)",
)


class CationPi(Interaction):
    """Cation-Pi interaction between a ligand (cation) and a residue (aromatic ring)

    Parameters
    ----------
    cation : str
        SMARTS for cation
    pi_ring : tuple
        SMARTS for aromatic rings (5 and 6 membered rings only)
    distance : float
        Cutoff distance between the centroid and the cation
    angle : tuple
        Min and max values for the angle between the vector normal to the ring
        plane and the vector going from the centroid to the cation


    .. versionchanged:: 1.1.0
        Handles resonance forms for amidine and guanidine as cations.

    .. versionchanged:: 2.0.0
        ``angles`` parameter renamed to ``angle``.

    """

    def __init__(
        self,
        cation: str = "[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
        pi_ring: tuple[str, ...] = (
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
        distance: float = 4.5,
        angle: "Angles" = (0, 30),
    ) -> None:
        self.cation = MolFromSmarts(cation)
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.angle = tuple(radians(i) for i in angle)

    def detect(
        self, cation: "Residue", pi: "Residue"
    ) -> Iterator["InteractionMetadata"]:
        cation_matches = cation.GetSubstructMatches(self.cation)
        for pi_ring in self.pi_ring:
            pi_matches = pi.GetSubstructMatches(pi_ring)
            if not (cation_matches and pi_matches):
                continue
            for cation_match, pi_match in product(cation_matches, pi_matches):
                cat = Geometry.Point3D(*cation.xyz[cation_match[0]])
                # get coordinates of atoms matching pi-system
                pi_coords = pi.xyz[list(pi_match)]
                # centroid of pi-system as 3d point
                centroid = Geometry.Point3D(*get_centroid(pi_coords))
                # distance between cation and centroid
                dist = cat.Distance(centroid)
                if dist > self.distance:
                    continue
                # vector normal to ring plane
                normal = get_ring_normal_vector(centroid, pi_coords)
                # vector between the centroid and the charge
                centroid_cation = centroid.DirectionVector(cat)
                # compute angle between normal to ring plane and
                # centroid-cation
                angle = normal.AngleTo(centroid_cation)
                if angle_between_limits(angle, *self.angle, ring=True):
                    yield self.metadata(
                        cation,
                        pi,
                        cation_match,
                        pi_match,
                        distance=dist,
                        angle=degrees(angle),
                    )


PiCation = CationPi.invert_role(
    "PiCation",
    "Cation-Pi interaction between a ligand (aromatic ring) and a residue (cation)",
)


class FaceToFace(BasePiStacking):
    """Face-to-face Pi-Stacking interaction between a ligand and a residue

    Parameters
    ----------
    distance : float
        Cutoff distance between each rings centroid
    plane_angles : tuple
        Min and max values for the angle between the ring planes
    normal_centroids_angles : tuple
        Min and max angles allowed between the vector normal to a ring's plane,
        and the vector between the centroid of both rings.
    pi_ring : list
        List of SMARTS for aromatic rings


    .. versionchanged:: 2.0.0
        Renamed ``centroid_distance`` to distance.

    """

    def __init__(
        self,
        distance: float = 5.5,
        plane_angle: "Angles" = (0, 35),
        normal_to_centroid_angle: "Angles" = (0, 33),
        pi_ring: tuple[str, ...] = (
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
    ) -> None:
        super().__init__(
            distance=distance,
            plane_angle=plane_angle,
            normal_to_centroid_angle=normal_to_centroid_angle,
            pi_ring=pi_ring,
        )


class EdgeToFace(BasePiStacking):
    """Edge-to-face Pi-Stacking interaction between a ligand and a residue

    Parameters
    ----------
    distance : float
        Cutoff distance between each rings centroid
    plane_angles : tuple
        Min and max values for the angle between the ring planes
    normal_centroids_angles : tuple
        Min and max angles allowed between the vector normal to a ring's plane,
        and the vector between the centroid of both rings.
    pi_ring : list
        List of SMARTS for aromatic rings
    intersect_radius : float
        Used to check whether the intersect point between ring planes falls within
        ``intersect_radius`` of the opposite ring's centroid.


    .. versionchanged:: 1.1.0
        In addition to the changes made to the base pi-stacking interaction, this
        implementation makes sure that the intersection between the perpendicular ring's
        plane and the other's plane falls inside the ring.

    .. versionchanged:: 2.0.0
        Renamed ``centroid_distance`` to distance, added ``intersect_radius`` parameter.

    """

    def __init__(
        self,
        distance: float = 6.5,
        plane_angle: "Angles" = (50, 90),
        normal_to_centroid_angle: "Angles" = (0, 30),
        pi_ring: tuple[str, ...] = (
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
        intersect_radius: float = 1.5,
    ) -> None:
        super().__init__(
            distance=distance,
            plane_angle=plane_angle,
            normal_to_centroid_angle=normal_to_centroid_angle,
            pi_ring=pi_ring,
            intersect=True,
            intersect_radius=intersect_radius,
        )


class PiStacking(Interaction):
    """Pi-Stacking interaction between a ligand and a residue

    Parameters
    ----------
    ftf_kwargs : dict
        Parameters to pass to the underlying FaceToFace class
    etf_kwargs : dict
        Parameters to pass to the underlying EdgeToFace class


    .. versionchanged:: 0.3.4
        `shortest_distance` has been replaced by `angle_normal_centroid`

    .. versionchanged:: 1.1.0
        The implementation now directly calls :class:`EdgeToFace` and
        :class:`FaceToFace` instead of overwriting the default parameters with more
        generic ones.

    """

    def __init__(
        self, ftf_kwargs: dict | None = None, etf_kwargs: dict | None = None
    ) -> None:
        self.ftf = FaceToFace(**ftf_kwargs or {})
        self.etf = EdgeToFace(**etf_kwargs or {})

    def detect(
        self, ligand: "Residue", residue: "Residue"
    ) -> Iterator["InteractionMetadata"]:
        yield from self.ftf.detect(ligand, residue)
        yield from self.etf.detect(ligand, residue)


class MetalDonor(Distance):
    """Metal complexation interaction between a ligand (metal) and a residue (chelated)

    Parameters
    ----------
    metal : str
        SMARTS for a transition metal
    ligand : str
        SMARTS for a ligand
    distance : float
        Cutoff distance


    .. versionchanged:: 1.1.0
        The initial SMARTS pattern was too broad.

    """

    def __init__(
        self,
        metal: str = "[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]",
        ligand: str = (
            "[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]"
        ),
        distance: float = 2.8,
    ) -> None:
        super().__init__(lig_pattern=metal, prot_pattern=ligand, distance=distance)


MetalAcceptor = MetalDonor.invert_role(
    "MetalAcceptor",
    "Metal complexation interaction between a ligand (chelated) and a residue (metal)",
)


class VdWContact(Interaction):
    """Interaction based on the van der Waals radii of interacting atoms.

    Parameters
    ----------
    tolerance : float, optional
        Tolerance added to the sum of vdW radii of atoms before comparing to
        the interatomic distance. If ``distance <= sum_vdw + tolerance`` the
        atoms are identified as a contact.
    vdwradii : dict, optional
        Updates to the vdW radii dictionary, with elements (first letter uppercase) as a
        key and the radius as a value.
    preset : str, optional
        Which preset of vdW radii to use. ``mdanalysis`` and ``rdkit`` correspond to the
        values used by the corresponding package, ``csd`` uses values calculated from
        the Cambridge Structural Database in doi.org/10.1039/C3DT50599E. ``rdkit`` and
        ``csd`` should contain almost all elements, as opposed to ``mdanalysis`` which
        only defines a limited subset.

    Raises
    ------
    ValueError
        ``tolerance`` parameter cannot be negative


    .. versionchanged:: 2.0.0
        Added the ``vdwradii`` parameter.

    .. versionchanged:: 2.1.0
        Added the ``preset`` parameter.

    """

    def __init__(
        self,
        tolerance: float = 0.0,
        vdwradii: dict[str, float] | None = None,
        preset: Literal["mdanalysis", "rdkit", "csd"] = "mdanalysis",
    ) -> None:
        if tolerance >= 0:
            self.tolerance = tolerance
        else:
            raise ValueError("`tolerance` must be 0 or positive")
        self._vdw_cache: dict[frozenset[str], float] = {}
        self.preset = preset.lower()
        preset_vdw = VDW_PRESETS[self.preset]
        self.vdwradii = {**preset_vdw, **vdwradii} if vdwradii else preset_vdw

    def _get_radii_sum(self, atom1: str, atom2: str) -> float:
        try:
            return self.vdwradii[atom1] + self.vdwradii[atom2]
        except KeyError:
            missing = []
            if atom1 not in self.vdwradii:
                missing.append(f"{atom1!r}")
            if atom2 not in self.vdwradii:
                missing.append(f"{atom2!r}")
            raise ValueError(
                f"van der Waals radius for atom {' and '.join(missing)} not found."
                " Either specify the missing radii in the `vdwradii` parameter for the"
                " VdWContact interaction, or use a preset different from the current"
                f" {self.preset!r}."
            ) from None

    def detect(
        self, ligand: "Residue", residue: "Residue"
    ) -> Iterator["InteractionMetadata"]:
        lxyz = ligand.GetConformer()
        rxyz = residue.GetConformer()
        for la, ra in product(ligand.GetAtoms(), residue.GetAtoms()):
            lig = la.GetSymbol()
            res = ra.GetSymbol()
            elements = frozenset((lig, res))
            try:
                vdw = self._vdw_cache[elements]
            except KeyError:
                vdw = self._get_radii_sum(lig, res) + self.tolerance
                self._vdw_cache[elements] = vdw
            dist = lxyz.GetAtomPosition(la.GetIdx()).Distance(
                rxyz.GetAtomPosition(ra.GetIdx()),
            )
            if dist <= vdw:
                yield self.metadata(
                    ligand,
                    residue,
                    (la.GetIdx(),),
                    (ra.GetIdx(),),
                    distance=dist,
                )
