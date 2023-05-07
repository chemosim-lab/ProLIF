"""
Detecting interactions between residues --- :mod:`prolif.interactions.interactions`
===================================================================================

Note that some of the SMARTS patterns used in the interaction classes are inspired from
`Pharmit`_ and `RDKit`_.

.. _Pharmit: https://sourceforge.net/p/pharmit/code/ci/master/tree/src/pharmarec.cpp
.. _RDKit: https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef

"""

from itertools import product
from math import degrees, radians

from MDAnalysis.topology.tables import vdwradii
from rdkit import Geometry
from rdkit.Chem import MolFromSmarts

from prolif.interactions.base import (
    BasePiStacking,
    Distance,
    DoubleAngle,
    Interaction,
    SingleAngle,
)
from prolif.utils import angle_between_limits, get_centroid, get_ring_normal_vector

__all__ = [
    "Hydrophobic",
    "HBAcceptor",
    "HBDonor",
    "XBAcceptor",
    "XBDonor",
    "Cationic",
    "Anionic",
    "CationPi",
    "PiCation",
    "FaceToFace",
    "EdgeToFace",
    "PiStacking",
    "MetalDonor",
    "MetalAcceptor",
    "VdWContact",
]
VDWRADII = {symbol.capitalize(): radius for symbol, radius in vdwradii.items()}


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

    """

    def __init__(
        self,
        hydrophobic=(
            "[c,s,Br,I,S&H0&v2," "$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]"
        ),
        distance=4.5,
    ):
        super().__init__(
            lig_pattern=hydrophobic, prot_pattern=hydrophobic, distance=distance
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

    """

    def __init__(
        self,
        acceptor=(
            "[#7&!$([nX3])&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4&+1]),"
            "O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O),o+0,"
            "F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])]"
        ),
        donor="[$([O,S;+0]),$([N;v3,v4&+1]),n+0]-[H]",
        distance=3.5,
        DHA_angle=(130, 180),
    ):
        super().__init__(
            lig_pattern=acceptor,
            prot_pattern=donor,
            distance=distance,
            angle=DHA_angle,
            distance_atom="P1",
            metadata_mapping={"angle": "DHA_angle"},
        )


HBDonor = HBAcceptor.invert_role(
    "HBDonor", "Hbond interaction between a ligand (donor) and a residue (acceptor)"
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

    """

    def __init__(
        self,
        acceptor="[#7,#8,P,S,Se,Te,a;!+{1-}][*]",
        donor="[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]",
        distance=3.5,
        AXD_angle=(130, 180),
        XAR_angle=(80, 140),
    ):
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
    "XBDonor", "Halogen bonding between a ligand (donor) and a residue (acceptor)"
)


class Cationic(Distance):
    """Ionic interaction between a ligand (cation) and a residue (anion)

    .. versionchanged:: 1.1.0
        Handles resonance forms for common acids, amidine and guanidine.

    """

    def __init__(
        self,
        cation="[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
        anion="[-{1-},$(O=[C,S,P]-[O-])]",
        distance=4.5,
    ):
        super().__init__(lig_pattern=cation, prot_pattern=anion, distance=distance)


Anionic = Cationic.invert_role(
    "Anionic", "Ionic interaction between a ligand (anion) and a residue (cation)"
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
        cation="[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
        pi_ring=(
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
        distance=4.5,
        angle=(0, 30),
    ):
        self.cation = MolFromSmarts(cation)
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.angle = tuple(radians(i) for i in angle)

    def detect(self, cation, pi):
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
        distance=5.5,
        plane_angle=(0, 35),
        normal_to_centroid_angle=(0, 33),
        pi_ring=(
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
    ):
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
        distance=6.5,
        plane_angle=(50, 90),
        normal_to_centroid_angle=(0, 30),
        pi_ring=(
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
        intersect_radius=1.5,
    ):
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
        The implementation now directly calls :class:`EdgeToFace` and :class:`FaceToFace`
        instead of overwriting the default parameters with more generic ones.

    """

    def __init__(self, ftf_kwargs=None, etf_kwargs=None):
        self.ftf = FaceToFace(**ftf_kwargs or {})
        self.etf = EdgeToFace(**etf_kwargs or {})

    def detect(self, ligand, residue):
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
        metal="[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]",
        ligand="[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]",
        distance=2.8,
    ):
        super().__init__(lig_pattern=metal, prot_pattern=ligand, distance=distance)


MetalAcceptor = MetalDonor.invert_role(
    "MetalAcceptor",
    "Metal complexation interaction between a ligand (chelated) and a residue (metal)",
)


class VdWContact(Interaction):
    """Interaction based on the van der Waals radii of interacting atoms.

    Parameters
    ----------
    tolerance : float
        Tolerance added to the sum of vdW radii of atoms before comparing to
        the interatomic distance. If ``distance <= sum_vdw + tolerance`` the
        atoms are identified as a contact
    vdwradii : dict, optional
        Updates to the vdW radii dictionary, with elements (first letter uppercase) as a
        key and the radius as a value.

    Raises
    ------
    ValueError
        ``tolerance`` parameter cannot be negative


    .. versionchanged:: 2.0.0
        Added the ``vdwradii`` parameter.

    """

    def __init__(self, tolerance=0.0, vdwradii=None):
        if tolerance >= 0:
            self.tolerance = tolerance
        else:
            raise ValueError("`tolerance` must be 0 or positive")
        self._vdw_cache = {}
        self.vdwradii = {**VDWRADII, **vdwradii} if vdwradii else VDWRADII

    def detect(self, ligand, residue):
        lxyz = ligand.GetConformer()
        rxyz = residue.GetConformer()
        for la, ra in product(ligand.GetAtoms(), residue.GetAtoms()):
            lig = la.GetSymbol()
            res = ra.GetSymbol()
            elements = frozenset((lig, res))
            try:
                vdw = self._vdw_cache[elements]
            except KeyError:
                vdw = self.vdwradii[lig] + self.vdwradii[res] + self.tolerance
                self._vdw_cache[elements] = vdw
            dist = lxyz.GetAtomPosition(la.GetIdx()).Distance(
                rxyz.GetAtomPosition(ra.GetIdx())
            )
            if dist <= vdw:
                yield self.metadata(
                    ligand, residue, (la.GetIdx(),), (ra.GetIdx(),), distance=dist
                )
