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
from math import degrees, pi, radians
from typing import TYPE_CHECKING, Literal

from rdkit import Geometry
from rdkit.Chem import MolFromSmarts
from rdkit.Chem.rdchem import HybridizationType

from prolif.interactions.base import (
    BasePiStacking,
    Distance,
    DoubleAngle,
    Interaction,
    SingleAngle,
)
from prolif.interactions.constants import IDEAL_ATOM_ANGLES, VDW_PRESETS
from prolif.io.constants import RESNAME_ALIASES
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
    "ImplicitHBAcceptor",
    "ImplicitHBDonor",
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


class ImplicitHBAcceptor(Distance, VdWContact):
    """Implicit Hbond interaction between a ligand (acceptor) and a residue (donor).

    .. versionadded:: 2.1.0

    Parameters
    ----------
    acceptor : str
        SMARTS for ``[Acceptor]``.
    donor : str
        SMARTS for ``[Donor]-[Implicit Hydrogen]``.
    distance : float
        Distance threshold between the acceptor and donor atoms.
    include_water : bool
        Whether to include water residues in the detection of interactions.
    tolerance_dev_daa : float
        Tolerance for the deviation from the ideal donor atom's angle (degrees).
        If the deviation is larger than this value, the interaction will not be
        considered valid (geometry checks).
    tolerance_dev_dpa : float
        Tolerance for the deviation from the ideal donor plane angle (degrees).
        If the deviation is larger than this value, the interaction will not be
        considered valid (geometry checks).
    vina_potential_max : float
        Distance (calculated as the donor-acceptor distance minus the sum of
        van der Waals radii) beyond which the piecewise linear potential term reaches
        its maximum value (1.0).
    vina_potential_min : float
        Distance (calculated as the donor-acceptor distance minus the sum of
        van der Waals radii) beyond which the piecewise linear potential term reaches
        its minimum value (0.0).
    ignore_geometry_checks : bool
        If True, the geometry checks for the interaction will be skipped. This is useful
        for cases where the geometry is not relevant or when the user wants to skip the
        geometry checks for performance reasons. Defaults to False.

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
        donor: str = (
            # implicit or explicit hydrogen bond donors
            "[$([O,S,#7;+0&h,+0&H,+0&H2,+0H3])"
            ",$([N;v4&+1&h,v4&+1&H,v4&+1&H2,v4&+1&H3,v4&+1&H4])"
            ",$([n+]c[nh]),$([n+]c[nH])]"
        ),
        distance: float = 3.5,
        include_water: bool = False,
        tolerance_dev_daa: float = 25,
        tolerance_dev_dpa: float = 30,
        vina_potential_max: float = -0.425,
        vina_potential_min: float = 0.565,
        ignore_geometry_checks: bool = False,
    ) -> None:
        super().__init__(lig_pattern=acceptor, prot_pattern=donor, distance=distance)
        VdWContact.__init__(self, preset="csd")
        self.include_water = include_water
        self.acceptor = acceptor
        self.donor = donor
        self.tolerance_dev_aaa = 90.0  # tolerance deviation for acceptor atom angle
        self.tolerance_dev_daa = tolerance_dev_daa
        self.tolerance_dev_apa = 90.0  # tolerance deviation for acceptor plane angle
        self.tolerance_dev_dpa = tolerance_dev_dpa

        # Specify where the piecewise linear terms become one (good interaction)
        self.vina_p_max = vina_potential_max
        # Specify where the piecewise linear terms become zero (bad interaction)
        self.vina_p_min = vina_potential_min
        self.ignore_geometry_checks = ignore_geometry_checks

    def detect(
        self, lig_res: "Residue", prot_res: "Residue"
    ) -> Iterator["InteractionMetadata"]:
        """Detect implicit hydrogen bond acceptor interactions.

        Parameters
        ----------
        lig_res : Residue
            Ligand residue.
        prot_res : Residue
            Protein residue.

        Yields
        ------
        InteractionMetadata
            Metadata for the detected interaction.

        """
        # Check if the interaction including water residues
        water_in_prot_res = self.check_water_residue(prot_res)
        water_in_lig_res = self.check_water_residue(lig_res)

        for interaction_data in super().detect(lig_res, prot_res):
            if (
                # If the interaction involves water residues
                (water_in_prot_res or water_in_lig_res)
                # Check if the user wants to include water residues
                and self.include_water
                and (
                    (
                        # If both residues are water, skip the geometry checks
                        water_in_prot_res and water_in_lig_res
                    )
                    or (
                        # If water is only in protein residue,
                        # either ignore geometry checks or check geometry on ligand
                        water_in_prot_res
                        and (
                            self.ignore_geometry_checks
                            or self.check_geometry(
                                interaction_data,
                                lig_res=lig_res,
                                prot_res=prot_res,
                                on="ligand",
                            )
                        )
                    )
                    or (
                        # If water is only in ligand residue,
                        # either ignore geometry checks or check geometry on protein
                        water_in_lig_res
                        and (
                            self.ignore_geometry_checks
                            or self.check_geometry(
                                interaction_data,
                                lig_res=lig_res,
                                prot_res=prot_res,
                                on="protein",
                            )
                        )
                    )
                )
            ) or (
                # If the interaction not involves water residues:
                (not water_in_prot_res and not water_in_lig_res)
                and (
                    # either skip geometry checks (first condition)
                    # or pass the geometry checks (second condition)
                    self.ignore_geometry_checks
                    or self.check_geometry(
                        interaction_data,
                        lig_res=lig_res,
                        prot_res=prot_res,
                        on="both",
                    )
                )
            ):
                yield self.add_vina_hbond_potential(
                    interaction_data, lig_res=lig_res, prot_res=prot_res
                )

    def check_geometry(
        self,
        interaction_data: "InteractionMetadata",
        lig_res: "Residue",
        prot_res: "Residue",
        on: Literal["ligand", "protein", "both"] = "both",
    ) -> bool:
        """Check the geometry of the interaction.

        Parameters
        ----------
        interaction_data : InteractionMetadata
            Metadata for the detected interaction.
        lig_res : Residue
            Ligand residue.
        prot_res : Residue
            Protein residue.
        on : Literal["ligand", "protein", "both"]
            Specifies which residue to check the geometry on. Defaults to "both".

        Returns
        -------
        bool
            True if the geometry is valid, False otherwise.

        """

        # Get the atoms involved in the interaction
        lig_atom_idx = interaction_data["indices"]["ligand"][0]
        lig_atom = lig_res.GetAtomWithIdx(lig_atom_idx)
        prot_atom_idx = interaction_data["indices"]["protein"][0]
        prot_atom = prot_res.GetAtomWithIdx(prot_atom_idx)

        # Initialize the interaction data
        ideal_acceptor_atom_angle = None
        acceptor_atom_angles = None
        deviation_aaa = None
        acceptor_plane_angle = None
        ideal_donor_atom_angle = None
        donor_atom_angles = None
        deviation_daa = None
        donor_plane_angle = None

        # Check acceptor (ligand-centered)
        if on in {"both", "ligand"}:
            # Check acceptor atom's angle
            ideal_acceptor_atom_angle = IDEAL_ATOM_ANGLES[lig_atom.GetHybridization()]
            acceptor_atom_angles = self._get_atom_angles(
                res=lig_res,
                res_atom_idx=lig_atom_idx,
                remote_res=prot_res,
                remote_res_atom_idx=prot_atom_idx,
            )
            deviation_aaa = min(
                abs(each_atom_angle - ideal_acceptor_atom_angle)
                for each_atom_angle in acceptor_atom_angles
            )
            if deviation_aaa > self.tolerance_dev_aaa:
                return False

            # Save geometry features in interaction data
            interaction_data["ideal_acceptor_angle"] = ideal_acceptor_atom_angle
            interaction_data["acceptor_atom_angles"] = acceptor_atom_angles
            interaction_data["acceptor_atom_angle_deviation"] = deviation_aaa

            # Check acceptor plane angle (if applicable, sp2)
            if lig_atom.GetHybridization() == HybridizationType.SP2:
                acceptor_plane_angle = self._get_plane_angle(
                    res=lig_res,
                    res_atom_idx=lig_atom_idx,
                    remote_res=prot_res,
                    remote_res_atom_idx=prot_atom_idx,
                )
                # Ideal acceptor's plane angle is 0 degrees
                if acceptor_plane_angle > self.tolerance_dev_apa:
                    return False

                # Save geometry features in interaction data
                interaction_data["acceptor_plane_angle"] = acceptor_plane_angle

        # Check donor atom (protein-centered)
        if on in {"both", "protein"}:
            # Check donor atom's angle
            ideal_donor_atom_angle = IDEAL_ATOM_ANGLES[prot_atom.GetHybridization()]
            donor_atom_angles = self._get_atom_angles(
                res=prot_res,
                res_atom_idx=prot_atom_idx,
                remote_res=lig_res,
                remote_res_atom_idx=lig_atom_idx,
            )
            deviation_daa = min(
                abs(each_atom_angle - ideal_donor_atom_angle)
                for each_atom_angle in donor_atom_angles
            )
            if deviation_daa > self.tolerance_dev_daa:
                return False

            # Save geometry features in interaction data
            interaction_data["ideal_donor_angle"] = ideal_donor_atom_angle
            interaction_data["donor_atom_angles"] = donor_atom_angles
            interaction_data["donor_atom_angle_deviation"] = deviation_daa

            # Check donor plane angle (if applicable, sp2)
            if prot_atom.GetHybridization() == HybridizationType.SP2:
                donor_plane_angle = self._get_plane_angle(
                    res=prot_res,
                    res_atom_idx=prot_atom_idx,
                    remote_res=lig_res,
                    remote_res_atom_idx=lig_atom_idx,
                )
                # Ideal donor's plane angle is 0 degrees
                if donor_plane_angle > self.tolerance_dev_dpa:
                    return False

                interaction_data["donor_plane_angle"] = donor_plane_angle

        # Return the result of the geometry checks
        return True

    def add_vina_hbond_potential(
        self,
        interaction_data: dict,
        lig_res: "Residue",
        prot_res: "Residue",
    ) -> dict:
        """Add hydrogen bond potential (derived from `Autodock Vina`_) to the
        interaction metadata.

        .. _Autodock Vina: https://github.com/ccsb-scripps/AutoDock-Vina/blob/develop/src/lib/potentials.h#L217


        Parameters
        ----------
        interaction_data : dict
            Metadata for the detected interaction.
        lig_res : Residue
            Ligand residue.
        prot_res : Residue
            Protein residue.

        Returns
        -------
        Dict
            Updated metadata with hydrogen bond probability.

        """
        # Hbond probability is based on the Autodock Vina Hbond interaction term
        lig_atom = lig_res.GetAtomWithIdx(interaction_data["indices"]["ligand"][0])
        prot_atom = prot_res.GetAtomWithIdx(interaction_data["indices"]["protein"][0])
        vdw_sum = self._get_radii_sum(lig_atom.GetSymbol(), prot_atom.GetSymbol())
        d_diff = interaction_data["distance"] - vdw_sum

        if d_diff <= self.vina_p_max:
            interaction_data["vina_hbond_potential"] = 1.0
        elif d_diff >= self.vina_p_min:
            interaction_data["vina_hbond_potential"] = 0.0
        else:
            # Piecewise linear function for the hydrogen bond potential
            interaction_data["vina_hbond_potential"] = (d_diff - self.vina_p_min) / (
                self.vina_p_max - self.vina_p_min
            )
        return interaction_data

    def check_water_residue(self, res: "Residue") -> bool:
        """Check if the residue is a water molecule.

        Parameters
        ----------
        res : Residue
            The residue to check.

        Returns
        -------
        bool
            True if the residue is a water molecule, False otherwise.
        """
        resname: str = RESNAME_ALIASES.get(res.resid.name, res.resid.name)
        return resname == "HOH"

    def _get_atom_angles(
        self,
        res: "Residue",
        res_atom_idx: int,
        remote_res: "Residue",
        remote_res_atom_idx: int,
    ) -> list[float]:
        """Get the angle of the atom in the residue (relative to the far atom).
        The angle is defined as follows:
        [nearby heavy atom]-[res_atom] ... [remote_res_atom]

        Parameters
        ----------
        res : Residue
            The residue containing the atom.
        res_atom_idx : int
            The index of the atom in the residue.
        remote_res : Residue
            The remote residue containing the far atom.
        remote_res_atom_idx : int
            The index of the far atom in the remote residue.

        Returns
        -------
        float
            The angle in degrees."""

        res_atom = res.GetAtomWithIdx(res_atom_idx)
        nearby_heavy_atoms = [
            atom for atom in res_atom.GetNeighbors() if atom.GetAtomicNum() != 1
        ]
        if not nearby_heavy_atoms:
            raise ValueError(
                f"No nearby heavy atoms found in residue {res.resid} "
                f"for atom {res_atom.GetSymbol()!r} at index {res_atom_idx}."
            )

        angles = []
        for nearby_heavy_atom in nearby_heavy_atoms:
            # Get the coordinates of the atoms
            nearby_coords = Geometry.Point3D(*res.xyz[nearby_heavy_atom.GetIdx()])
            res_atom_coords = Geometry.Point3D(*res.xyz[res_atom_idx])
            far_atom_coords = Geometry.Point3D(*remote_res.xyz[remote_res_atom_idx])

            # Calculate the angle
            res2nearby = res_atom_coords.DirectionVector(nearby_coords)
            res2far = res_atom_coords.DirectionVector(far_atom_coords)
            angle = res2nearby.AngleTo(res2far)
            angles.append(degrees(angle))

        return angles

    def _get_plane_angle(
        self,
        res: "Residue",
        res_atom_idx: int,
        remote_res: "Residue",
        remote_res_atom_idx: int,
    ) -> float:
        """Get the plane angle of the sp2 atom in the residue
        (relative to the far atom).

        Parameters
        ----------
        res : Residue
            The residue containing the atom.
        res_atom_idx : int
            The index of the atom in the residue.
        remote_res : Residue
            The remote residue containing the far atom.
        remote_res_atom_idx : int
            The index of the far atom in the remote residue.

        Returns
        -------
        float
            The angle in degrees.
        """
        res_atom = res.GetAtomWithIdx(res_atom_idx)
        nearby_heavy_atoms = [
            atom for atom in res_atom.GetNeighbors() if atom.GetAtomicNum() != 1
        ]
        # If there are less than 2 nearby heavy atoms, we cannot calculate the plane.
        # We will need to extend the search of nearby atoms.
        if len(nearby_heavy_atoms) < 2:
            nearby_nearby_atoms = []
            for each_atom in nearby_heavy_atoms:
                nearby_nearby_atoms.extend(
                    [
                        nearby_nearby_atom
                        for nearby_nearby_atom in each_atom.GetNeighbors()
                        if nearby_nearby_atom.GetAtomicNum() != 1
                        and nearby_nearby_atom.GetIdx() != res_atom_idx
                    ]
                )
            nearby_heavy_atoms.extend(nearby_nearby_atoms)

        # Get the coordinates of the atoms
        nearby_atom_1_idx = nearby_heavy_atoms[0].GetIdx()
        nearby_atom_2_idx = nearby_heavy_atoms[1].GetIdx()
        res_atom_coords = Geometry.Point3D(*res.xyz[res_atom_idx])
        nearby_atom_1_coords = Geometry.Point3D(*res.xyz[nearby_atom_1_idx])
        nearby_atom_2_coords = Geometry.Point3D(*res.xyz[nearby_atom_2_idx])
        remote_atom_coords = Geometry.Point3D(*remote_res.xyz[remote_res_atom_idx])

        # Calculate the normal vector of the plane
        atom2nearby_1 = res_atom_coords.DirectionVector(nearby_atom_1_coords)
        atom2nearby_2 = res_atom_coords.DirectionVector(nearby_atom_2_coords)
        normal_vector = atom2nearby_1.CrossProduct(atom2nearby_2)

        # Calculate the angle between the normal vector and
        # the vector to the remote atom
        atom2remote_atom = res_atom_coords.DirectionVector(remote_atom_coords)
        angle = normal_vector.AngleTo(atom2remote_atom)

        # Convert it to plane angle in degrees
        return abs(degrees(angle - pi / 2))


ImplicitHBDonor = ImplicitHBAcceptor.invert_role(
    "ImplicitHBDonor",
    "Implicit Hbond interaction between a ligand (donor) and a residue (acceptor)",
)
