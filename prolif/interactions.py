"""
Detecting interactions between residues --- :mod:`prolif.interactions`
======================================================================

You can declare your own interaction class like this::

    from scipy.spatial import distance_matrix

    class CloseContact(prolif.Interaction):
        def detect(self, res1, res2, threshold=2.0):
            dist_matrix = distance_matrix(res1.xyz, res2.xyz)
            if (dist_matrix <= threshold).any():
                # return format: bool, ligand atom index, protein atom index
                return True, None, None
            return False, None, None

.. warning:: Your custom class must inherit from :class:`prolif.interactions.Interaction`

The new "CloseContact" class is then automatically added to the list of
interactions available to the fingerprint generator::

    >>> u = mda.Universe(prolif.datafiles.TOP, prolif.datafiles.TRAJ)
    >>> prot = u.select_atoms("protein")
    >>> lig = u.select_atoms("resname LIG")
    >>> fp = prolif.Fingerprint(interactions="all")
    >>> df = fp.run(u.trajectory[0:1], lig, prot).to_dataframe()
    >>> df.xs("CloseContact", level=1, axis=1)
       ASP129.A  VAL201.A
    0         1         1
    >>> lmol = prolif.Molecule.from_mda(lig)
    >>> pmol = prolif.Molecule.from_mda(prot)
    >>> fp.closecontact(lmol, pmol["ASP129.A"])
    True

Note that some of the SMARTS patterns used in the interaction classes are inspired from
`Pharmit`_ and `RDKit`_.

.. _Pharmit: https://sourceforge.net/p/pharmit/code/ci/master/tree/src/pharmarec.cpp
.. _RDKit: https://github.com/rdkit/rdkit/blob/master/Data/BaseFeatures.fdef

"""

import warnings
from abc import ABC, ABCMeta, abstractmethod
from itertools import product
from math import radians

import numpy as np
from MDAnalysis.topology.tables import vdwradii
from rdkit import Geometry
from rdkit.Chem import MolFromSmarts

from .utils import angle_between_limits, get_centroid, get_ring_normal_vector

_INTERACTIONS = {}


class _InteractionMeta(ABCMeta):
    """Metaclass to register interactions automatically"""
    def __init__(cls, name, bases, classdict):
        type.__init__(cls, name, bases, classdict)
        if name in _INTERACTIONS.keys():
            warnings.warn(f"The {name!r} interaction has been superseded by a "
                          f"new class with id {id(cls):#x}")
        _INTERACTIONS[name] = cls


class Interaction(ABC, metaclass=_InteractionMeta):
    """Abstract class for interactions

    All interaction classes must inherit this class and define a
    :meth:`~detect` method
    """
    @abstractmethod
    def detect(self, **kwargs):
        pass


def get_mapindex(res, index):
    """Get the index of the atom in the original molecule

    Parameters
    ----------
    res : prolif.residue.Residue
        The residue in the protein or ligand
    index : int
        The index of the atom in the :class:`~prolif.residue.Residue`

    Returns
    -------
    mapindex : int
        The index of the atom in the :class:`~prolif.molecule.Molecule`
    """
    return res.GetAtomWithIdx(index).GetUnsignedProp("mapindex")


class _Distance(Interaction):
    """Generic class for distance-based interactions

    Parameters
    ----------
    lig_pattern : str
        SMARTS pattern for atoms in ligand residues
    prot_pattern : str
        SMARTS pattern for atoms in protein residues
    distance : float
        Cutoff distance, measured between the first atom of each pattern
    """
    def __init__(self, lig_pattern, prot_pattern, distance):
        self.lig_pattern = MolFromSmarts(lig_pattern)
        self.prot_pattern = MolFromSmarts(prot_pattern)
        self.distance = distance

    def detect(self, lig_res, prot_res):
        lig_matches = lig_res.GetSubstructMatches(self.lig_pattern)
        prot_matches = prot_res.GetSubstructMatches(self.prot_pattern)
        if lig_matches and prot_matches:
            for lig_match, prot_match in product(lig_matches,
                                                           prot_matches):
                alig = Geometry.Point3D(*lig_res.xyz[lig_match[0]])
                aprot = Geometry.Point3D(*prot_res.xyz[prot_match[0]])
                if alig.Distance(aprot) <= self.distance:
                    return True, lig_match[0], prot_match[0]
        return False, None, None


class Hydrophobic(_Distance):
    """Hydrophobic interaction

    Parameters
    ----------
    hydrophobic : str
        SMARTS query for hydrophobic atoms
    distance : float
        Cutoff distance for the interaction


    .. versionchanged:: 1.1.0
        The initial SMARTS pattern was too broad.

    """
    def __init__(
        self,
        hydrophobic=(
            "[c,s,Br,I,S&H0&v2,"
            "$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]"
        ),
        distance=4.5
    ):
        super().__init__(hydrophobic, hydrophobic, distance)


class _BaseHBond(Interaction):
    """Base class for Hydrogen bond interactions

    Parameters
    ----------
    donor : str
        SMARTS for ``[Donor]-[Hydrogen]``
    acceptor : str
        SMARTS for ``[Acceptor]``
    distance : float
        Cutoff distance between the donor and acceptor atoms
    angles : tuple
        Min and max values for the ``[Donor]-[Hydrogen]...[Acceptor]`` angle


    .. versionchanged:: 1.1.0
        The initial SMARTS pattern was too broad.

    """
    def __init__(
        self,         
        donor="[$([O,S;+0]),$([N;v3,v4&+1]),n+0]-[H]",
        acceptor=(
            "[#7&!$([nX3])&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4&+1]),"
            "O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O),o+0,"
            "F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])]"
        ),
        distance=3.5,
        angles=(130, 180)
    ):
        self.donor = MolFromSmarts(donor)
        self.acceptor = MolFromSmarts(acceptor)
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, acceptor, donor):
        acceptor_matches = acceptor.GetSubstructMatches(self.acceptor)
        donor_matches = donor.GetSubstructMatches(self.donor)
        if acceptor_matches and donor_matches:
            for donor_match, acceptor_match in product(donor_matches,
                                                       acceptor_matches):
                # D-H ... A
                d = Geometry.Point3D(*donor.xyz[donor_match[0]])
                h = Geometry.Point3D(*donor.xyz[donor_match[1]])
                a = Geometry.Point3D(*acceptor.xyz[acceptor_match[0]])
                if d.Distance(a) <= self.distance:
                    hd = h.DirectionVector(d)
                    ha = h.DirectionVector(a)
                    # get DHA angle
                    angle = hd.AngleTo(ha)
                    if angle_between_limits(angle, *self.angles):
                        return True, acceptor_match[0], donor_match[1]
        return False, None, None


class HBDonor(_BaseHBond):
    """Hbond interaction between a ligand (donor) and a residue (acceptor)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class HBAcceptor(_BaseHBond):
    """Hbond interaction between a ligand (acceptor) and a residue (donor)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class _BaseXBond(Interaction):
    """Base class for Halogen bond interactions

    Parameters
    ----------
    donor : str
        SMARTS for ``[Donor]-[Halogen]``
    acceptor : str
        SMARTS for ``[Acceptor]-[R]``
    distance : float
        Cutoff distance between the halogen and acceptor atoms
    axd_angles : tuple
        Min and max values for the ``[Acceptor]...[Halogen]-[Donor]`` angle
    xar_angles : tuple
        Min and max values for the ``[R]-[Acceptor]...[Halogen]`` angle

    Notes
    -----
    Distance and angle adapted from Auffinger et al. PNAS 2004
    """
    def __init__(self,
                 donor="[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]",
                 acceptor="[#7,#8,P,S,Se,Te,a;!+{1-}][*]",
                 distance=3.5,
                 axd_angles=(130, 180),
                 xar_angles=(80, 140)):
        self.donor = MolFromSmarts(donor)
        self.acceptor = MolFromSmarts(acceptor)
        self.distance = distance
        self.axd_angles = tuple(radians(i) for i in axd_angles)
        self.xar_angles = tuple(radians(i) for i in xar_angles)

    def detect(self, acceptor, donor):
        acceptor_matches = acceptor.GetSubstructMatches(self.acceptor)
        donor_matches = donor.GetSubstructMatches(self.donor)
        if acceptor_matches and donor_matches:
            for donor_match, acceptor_match in product(donor_matches,
                                                       acceptor_matches):
                # D-X ... A distance
                d = Geometry.Point3D(*donor.xyz[donor_match[0]])
                x = Geometry.Point3D(*donor.xyz[donor_match[1]])
                a = Geometry.Point3D(*acceptor.xyz[acceptor_match[0]])
                if x.Distance(a) <= self.distance:
                    # D-X ... A angle
                    xd = x.DirectionVector(d)
                    xa = x.DirectionVector(a)
                    angle = xd.AngleTo(xa)
                    if angle_between_limits(angle, *self.axd_angles):
                        # X ... A-R angle
                        r = Geometry.Point3D(*acceptor.xyz[acceptor_match[1]])
                        ax = a.DirectionVector(x)
                        ar = a.DirectionVector(r)
                        angle = ax.AngleTo(ar)
                        if angle_between_limits(angle, *self.xar_angles):
                            return True, acceptor_match[0], donor_match[1]
        return False, None, None


class XBAcceptor(_BaseXBond):
    """Halogen bonding between a ligand (acceptor) and a residue (donor)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class XBDonor(_BaseXBond):
    """Halogen bonding between a ligand (donor) and a residue (acceptor)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class _BaseIonic(_Distance):
    """Base class for ionic interactions

    .. versionchanged:: 1.1.0
        Handles resonance forms for common acids, amidine and guanidine.

    """
    def __init__(self,
                 cation="[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
                 anion="[-{1-},$(O=[C,S,P]-[O-])]",
                 distance=4.5):
        super().__init__(cation, anion, distance)


class Cationic(_BaseIonic):
    """Ionic interaction between a ligand (cation) and a residue (anion)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class Anionic(_BaseIonic):
    """Ionic interaction between a ligand (anion) and a residue (cation)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class _BaseCationPi(Interaction):
    """Base class for cation-pi interactions

    Parameters
    ----------
    cation : str
        SMARTS for cation
    pi_ring : tuple
        SMARTS for aromatic rings (5 and 6 membered rings only)
    distance : float
        Cutoff distance between the centroid and the cation
    angles : tuple
        Min and max values for the angle between the vector normal to the ring
        plane and the vector going from the centroid to the cation


    .. versionchanged:: 1.1.0
        Handles resonance forms for amidine and guanidine as cations.

    """
    def __init__(self,
                 cation="[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]",
                 pi_ring=("[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1", "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1"),
                 distance=4.5,
                 angles=(0, 30)):
        self.cation = MolFromSmarts(cation)
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

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
                if cat.Distance(centroid) > self.distance:
                    continue
                # vector normal to ring plane
                normal = get_ring_normal_vector(centroid, pi_coords)
                # vector between the centroid and the charge
                centroid_cation = centroid.DirectionVector(cat)
                # compute angle between normal to ring plane and
                # centroid-cation
                angle = normal.AngleTo(centroid_cation)
                if angle_between_limits(angle, *self.angles, ring=True):
                    return True, cation_match[0], pi_match[0]
        return False, None, None


class PiCation(_BaseCationPi):
    """Cation-Pi interaction between a ligand (aromatic ring) and a residue
    (cation)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class CationPi(_BaseCationPi):
    """Cation-Pi interaction between a ligand (cation) and a residue
    (aromatic ring)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class _BasePiStacking(Interaction):
    """Base class for Pi-Stacking interactions
    
    Parameters
    ----------
    centroid_distance : float
        Cutoff distance between each rings centroid
    plane_angles : tuple
        Min and max values for the angle between the ring planes
    normal_centroids_angles : tuple
        Min and max angles allowed between the vector normal to a ring's plane,
        and the vector between the centroid of both rings.
    pi_ring : list
        List of SMARTS for aromatic rings


    .. versionchanged:: 1.1.0
        The implementation now relies on the angle between the vector normal to a ring's
        plane and the vector between centroids (``normal_centroids_angles``) instead of
        the ``shortest_distance`` parameter.

    """
    def __init__(self,
                 centroid_distance=5.5,
                 plane_angle=(0, 35),
                 normal_to_centroid_angle=(0, 30),
                 pi_ring=("[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1", "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1")):
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.centroid_distance = centroid_distance
        self.plane_angle = tuple(radians(i) for i in plane_angle) 
        self.normal_to_centroid_angle = tuple(radians(i) for i in normal_to_centroid_angle)
        self.edge = False
        self.ring_radius = 1.7

    def detect(self, ligand, residue):
        for pi_rings in product(self.pi_ring, repeat=2):
            res_matches = residue.GetSubstructMatches(pi_rings[0])
            lig_matches = ligand.GetSubstructMatches(pi_rings[1])
            if not (lig_matches and res_matches):
                continue
            for lig_match, res_match in product(lig_matches, res_matches):
                lig_pi_coords = ligand.xyz[list(lig_match)]
                lig_centroid = Geometry.Point3D(*get_centroid(lig_pi_coords))
                res_pi_coords = residue.xyz[list(res_match)]
                res_centroid = Geometry.Point3D(*get_centroid(res_pi_coords))
                cdist = lig_centroid.Distance(res_centroid)
                if cdist > self.centroid_distance:
                    continue
                # ligand
                lig_normal = get_ring_normal_vector(lig_centroid,
                                                    lig_pi_coords)
                # residue
                res_normal = get_ring_normal_vector(res_centroid,
                                                    res_pi_coords)
                plane_angle = lig_normal.AngleTo(res_normal)
                if not angle_between_limits(
                    plane_angle, *self.plane_angle, ring=True
                ):
                    continue
                c1c2 = lig_centroid.DirectionVector(res_centroid)
                c2c1 = res_centroid.DirectionVector(lig_centroid)
                n1c1c2 = lig_normal.AngleTo(c1c2)
                n2c2c1 = res_normal.AngleTo(c2c1)
                if not (
                    angle_between_limits(
                        n1c1c2, *self.normal_to_centroid_angle, ring=True
                    ) or angle_between_limits(
                        n2c2c1, *self.normal_to_centroid_angle, ring=True
                    )
                ):
                    continue
                if self.edge:
                    # look for point of intersection between both ring planes
                    intersect = self._get_intersect_point(
                        lig_normal, lig_centroid, res_normal, res_centroid
                    )
                    if intersect is None:
                        continue
                    # check if intersection point falls ~within plane ring
                    intersect_dist = min(
                        lig_centroid.Distance(intersect),
                        res_centroid.Distance(intersect)
                    )
                    if intersect_dist > self.ring_radius:
                        continue
                return True, lig_match[0], res_match[0]
        return False, None, None

    @staticmethod
    def _get_intersect_point(
        plane_normal, plane_centroid, tilted_normal, tilted_centroid,
    ):
        # intersect line is orthogonal to both planes normal vectors 
        intersect_direction = plane_normal.CrossProduct(tilted_normal)
        # setup system of linear equations to solve
        A = np.array([list(plane_normal), list(tilted_normal), list(intersect_direction)])
        if np.linalg.det(A) == 0:
            return None
        tilted_offset = tilted_normal.DotProduct(
            Geometry.Point3D(*tilted_centroid)
        )
        plane_offset = plane_normal.DotProduct(
            Geometry.Point3D(*plane_centroid)
        )
        d = np.array([[plane_offset], [tilted_offset], [0.]])
        # point on intersect line
        point = np.linalg.solve(A, d).T[0]
        point = Geometry.Point3D(*point)
        # find projection of centroid on intersect line using vector projection
        vec = plane_centroid - point
        intersect_direction.Normalize()
        scalar_proj = intersect_direction.DotProduct(vec)
        return point + intersect_direction * scalar_proj


class FaceToFace(_BasePiStacking):
    """Face-to-face Pi-Stacking interaction between a ligand and a residue"""
    def __init__(self,
                 centroid_distance=5.5,
                 plane_angle=(0, 35),
                 normal_to_centroid_angle=(0, 33),
                 pi_ring=("[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1", "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1")):
        super().__init__(
            centroid_distance=centroid_distance,
            plane_angle=plane_angle,
            normal_to_centroid_angle=normal_to_centroid_angle,
            pi_ring=pi_ring
        )


class EdgeToFace(_BasePiStacking):
    """Edge-to-face Pi-Stacking interaction between a ligand and a residue
    
    .. versionchanged:: 1.1.0
        In addition to the changes made to the base pi-stacking interaction, this
        implementation makes sure that the intersection between the perpendicular ring's
        plane and the other's plane falls inside the ring.

    """
    def __init__(self,
                 centroid_distance=6.5,
                 plane_angle=(50, 90),
                 normal_to_centroid_angle=(0, 30),
                 ring_radius=1.5,
                 pi_ring=("[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1", "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1")):
        super().__init__(
            centroid_distance=centroid_distance,
            plane_angle=plane_angle,
            normal_to_centroid_angle=normal_to_centroid_angle,
            pi_ring=pi_ring
        )
        self.edge = True
        self.ring_radius = ring_radius
    
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


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
        ftf = self.ftf.detect(ligand, residue)
        if ftf[0] is True:
            return ftf
        return self.etf.detect(ligand, residue)


class _BaseMetallic(_Distance):
    """Base class for metal complexation

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
    def __init__(self,
                 metal="[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]",
                 ligand="[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]",
                 distance=2.8):
        super().__init__(metal, ligand, distance)


class MetalDonor(_BaseMetallic):
    """Metallic interaction between a metal and a residue (chelated)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class MetalAcceptor(_BaseMetallic):
    """Metallic interaction between a ligand (chelated) and a metal residue"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class VdWContact(Interaction):
    """Interaction based on the van der Waals radii of interacting atoms.

    Parameters
    ----------
    tolerance : float
        Tolerance added to the sum of vdW radii of atoms before comparing to
        the interatomic distance. If ``distance <= sum_vdw + tolerance`` the
        atoms are identified as a contact

    Raises
    ------
    ValueError : ``tolerance`` parameter cannot be negative
    """
    def __init__(self, tolerance=.5):
        if tolerance >= 0:
            self.tolerance = tolerance
        else:
            raise ValueError("`tolerance` must be 0 or positive")
        self._vdw_cache = {}

    def detect(self, ligand, residue):
        lxyz = ligand.GetConformer()
        rxyz = residue.GetConformer()
        for la, ra in product(ligand.GetAtoms(), residue.GetAtoms()):
            lig = la.GetSymbol().upper()
            res = ra.GetSymbol().upper()
            try:
                vdw = self._vdw_cache[(lig, res)]
            except KeyError:
                vdw = vdwradii[lig] + vdwradii[res] + self.tolerance
                self._vdw_cache[(lig, res)] = vdw
            dist = (lxyz.GetAtomPosition(la.GetIdx())
                        .Distance(rxyz.GetAtomPosition(ra.GetIdx())))
            if dist <= vdw:
                return True, la.GetIdx(), ra.GetIdx()
        return False, None, None
