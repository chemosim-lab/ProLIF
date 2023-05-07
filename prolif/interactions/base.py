"""
Base interaction classes --- :mod:`prolif.interactions.base`
============================================================

This module contains the base classes used to build most of the interactions.
"""
import warnings
from itertools import product
from math import degrees, radians

import numpy as np
from rdkit import Geometry
from rdkit.Chem import MolFromSmarts

from prolif.interactions.utils import DISTANCE_FUNCTIONS, get_mapindex
from prolif.utils import angle_between_limits, get_centroid, get_ring_normal_vector

_INTERACTIONS = {}
_BASE_INTERACTIONS = {}


class Interaction:
    """Base class for interactions

    All interaction classes must inherit this class and define a :meth:`~detect` method.

    .. versionchanged:: 2.0.0
        Changed the return type of interactions. Added some helper methods to easily
        update/derive interaction classes.
    """

    def __init_subclass__(cls, is_abstract=False):
        super().__init_subclass__()
        name = cls.__name__
        register = _BASE_INTERACTIONS if is_abstract else _INTERACTIONS
        if not hasattr(cls, "detect"):
            raise TypeError(
                f"Can't instantiate interaction class {name} without a `detect` method."
            )
        if name in register:
            warnings.warn(
                f"The {name!r} interaction has been superseded by a "
                f"new class with id {id(cls):#x}"
            )
        register[name] = cls

    def __call__(self, lig_res, prot_res, metadata=False):
        for int_data in self.detect(lig_res, prot_res):
            yield int_data if metadata else True

    def __repr__(self):  # pragma: no cover
        cls = self.__class__
        return f"<{cls.__module__}.{cls.__name__} at {id(self):#x}>"

    def metadata(self, lig_res, prot_res, lig_indices, prot_indices, **data):
        """Returns a dict containing the indices of atoms responsible for the
        interaction, alongside any other metrics (e.g. distance, angle...etc.)."""
        if hasattr(self, "_metadata_mapping"):
            mapping = self._metadata_mapping or {}
            data = {mapping.get(key, key): value for key, value in data.items()}
        return {
            "indices": {
                "ligand": lig_indices,
                "protein": prot_indices,
            },
            "parent_indices": {
                "ligand": tuple(
                    [get_mapindex(lig_res, index) for index in lig_indices]
                ),
                "protein": tuple(
                    [get_mapindex(prot_res, index) for index in prot_indices]
                ),
            },
            **data,
        }

    @staticmethod
    def _invert_metadata(metadata):
        """Invert the role of the ligand and protein components in the dict returned
        by :meth:`~Interaction.metadata`."""
        if metadata is not None:
            metadata["indices"]["protein"], metadata["indices"]["ligand"] = (
                metadata["indices"]["ligand"],
                metadata["indices"]["protein"],
            )
            (
                metadata["parent_indices"]["protein"],
                metadata["parent_indices"]["ligand"],
            ) = (
                metadata["parent_indices"]["ligand"],
                metadata["parent_indices"]["protein"],
            )
        return metadata

    @classmethod
    def invert_role(cls, name, docstring):
        """Creates a new interaction class where the role of the ligand and protein
        residues have been swapped. Usefull to create e.g. an acceptor class from a
        donor class.
        """
        cls_docstring = cls.__doc__ or "\n"
        parameters_doc = cls_docstring.split("\n", maxsplit=1)[1]
        __doc__ = f"{docstring}\n{parameters_doc}"
        inverted = type(name, (cls,), {"__doc__": __doc__})

        def detect(self, ligand, residue):
            for metadata in super(inverted, self).detect(residue, ligand):
                yield self._invert_metadata(metadata)

        inverted.detect = detect
        return inverted


class Distance(Interaction, is_abstract=True):
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
            for lig_match, prot_match in product(lig_matches, prot_matches):
                alig = Geometry.Point3D(*lig_res.xyz[lig_match[0]])
                aprot = Geometry.Point3D(*prot_res.xyz[prot_match[0]])
                dist = alig.Distance(aprot)
                if dist <= self.distance:
                    yield self.metadata(
                        lig_res, prot_res, lig_match, prot_match, distance=dist
                    )


class SingleAngle(Interaction, is_abstract=True):
    """Generic class for interactions using constraints on a distance and an angle.

    Parameters
    ----------
    lig_pattern : str
        SMARTS pattern for atoms in ligand residues.
    prot_pattern : str
        SMARTS pattern for atoms in protein residues, should include (at least) two
        atoms, the first two atoms will be used for the angle calculation.
    distance : float
        Distance threshold.
    angle : tuple
        Min and max values for the ``[P1]-[P2]...[L1]`` angle, where L1 is the first
        atom in the ``lig_pattern`` SMARTS, and P1 and P2 are the first two atoms in the
        ``prot_pattern`` SMARTS.
    distance_atom : str
        Which atom from the protein (P1 or P2) to use for the distance calculation with
        L1.
    metadata_mapping : dict, optional
        Mapping for names used in the metadata dict for the distance and angle variables


    .. versionadded:: 2.0.0

    """

    def __init__(
        self,
        lig_pattern,
        prot_pattern,
        distance,
        angle,
        distance_atom,
        metadata_mapping=None,
    ):
        self.lig_pattern = MolFromSmarts(lig_pattern)
        self.prot_pattern = MolFromSmarts(prot_pattern)
        self.distance = distance
        self.angle = tuple(radians(i) for i in angle)
        self._measure_distance = DISTANCE_FUNCTIONS[distance_atom]
        self._metadata_mapping = metadata_mapping

    def detect(self, lig_res, prot_res):
        lig_matches = lig_res.GetSubstructMatches(self.lig_pattern)
        prot_matches = prot_res.GetSubstructMatches(self.prot_pattern)
        if lig_matches and prot_matches:
            for lig_match, prot_match in product(lig_matches, prot_matches):
                l1 = Geometry.Point3D(*lig_res.xyz[lig_match[0]])
                p1 = Geometry.Point3D(*prot_res.xyz[prot_match[0]])
                p2 = Geometry.Point3D(*prot_res.xyz[prot_match[1]])
                dist = self._measure_distance(l1, p1, p2)
                if dist <= self.distance:
                    # P1-P2 ... L1
                    p2p1 = p2.DirectionVector(p1)
                    p2l1 = p2.DirectionVector(l1)
                    angle = p2p1.AngleTo(p2l1)
                    if angle_between_limits(angle, *self.angle):
                        yield self.metadata(
                            lig_res,
                            prot_res,
                            lig_match,
                            prot_match,
                            distance=dist,
                            angle=degrees(angle),
                        )


class DoubleAngle(Interaction, is_abstract=True):
    """Generic class for interactions using constraints on a distance and two angles.

    Parameters
    ----------
    lig_pattern : str
        SMARTS for ``[L1]-[L2]``
    prot_pattern : str
        SMARTS for ``[P1]-[P2]``
    distance : float
        Distance threshold
    L1P2P1_angle : tuple
        Min and max values for the ``[L1]...[P2]-[P1]`` angle
    L2L1P2_angle : tuple
        Min and max values for the ``[L2]-[L1]...[P2]`` angle
    distance_atoms : tuple[str, str]
        Which atoms to use for the distance calculation: L1 or L2, and P1 or P2
    metadata_mapping : dict, optional
        Mapping for names used in the metadata dict for the distance and angle variables


    .. versionadded:: 2.0.0

    """

    def __init__(
        self,
        lig_pattern,
        prot_pattern,
        distance,
        L1P2P1_angle,
        L2L1P2_angle,
        distance_atoms=("L1", "P2"),
        metadata_mapping=None,
    ):
        self.lig_pattern = MolFromSmarts(lig_pattern)
        self.prot_pattern = MolFromSmarts(prot_pattern)
        self.distance = distance
        self.L1P2P1_angle = tuple(radians(i) for i in L1P2P1_angle)
        self.L2L1P2_angle = tuple(radians(i) for i in L2L1P2_angle)
        self._measure_distance = DISTANCE_FUNCTIONS[distance_atoms]
        self._metadata_mapping = metadata_mapping

    def detect(self, lig_res, prot_res):
        lig_matches = lig_res.GetSubstructMatches(self.lig_pattern)
        prot_matches = prot_res.GetSubstructMatches(self.prot_pattern)
        if lig_matches and prot_matches:
            for lig_match, prot_match in product(lig_matches, prot_matches):
                l1 = Geometry.Point3D(*lig_res.xyz[lig_match[0]])
                l2 = Geometry.Point3D(*lig_res.xyz[lig_match[1]])
                p1 = Geometry.Point3D(*prot_res.xyz[prot_match[0]])
                p2 = Geometry.Point3D(*prot_res.xyz[prot_match[1]])
                dist = self._measure_distance(l1, l2, p1, p2)
                if dist <= self.distance:
                    p2p1 = p2.DirectionVector(p1)
                    p2l1 = p2.DirectionVector(l1)
                    l1p2p1 = p2p1.AngleTo(p2l1)
                    if angle_between_limits(l1p2p1, *self.L1P2P1_angle):
                        l1p2 = l1.DirectionVector(p2)
                        l1l2 = l1.DirectionVector(l2)
                        l2l1p2 = l1p2.AngleTo(l1l2)
                        if angle_between_limits(l2l1p2, *self.L2L1P2_angle):
                            yield self.metadata(
                                lig_res,
                                prot_res,
                                lig_match,
                                prot_match,
                                distance=dist,
                                L1P2P1_angle=degrees(l1p2p1),
                                L2L1P2_angle=degrees(l2l1p2),
                            )


class BasePiStacking(Interaction, is_abstract=True):
    """Base class for Pi-Stacking interactions

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
    intersect : bool
        Whether to look for the point of intersection between both rings (for T-shaped
        interaction).
    intersect_radius : float
        If ``intersect=True``, used to check whether the intersect point falls within
        ``intersect_radius`` of the opposite ring's centroid.


    .. versionchanged:: 1.1.0
        The implementation now relies on the angle between the vector normal to a ring's
        plane and the vector between centroids (``normal_centroids_angles``) instead of
        the ``shortest_distance`` parameter.

    .. versionchanged:: 2.0.0
        Renamed ``centroid_distance`` to distance. Added ``intersect`` and
        ``ring_radius`` parameters.

    """

    def __init__(
        self,
        distance,
        plane_angle,
        normal_to_centroid_angle,
        pi_ring=(
            "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1",
            "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1",
        ),
        intersect=False,
        intersect_radius=1.5,
    ):
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.plane_angle = tuple(radians(i) for i in plane_angle)
        self.normal_to_centroid_angle = tuple(
            radians(i) for i in normal_to_centroid_angle
        )
        self.intersect = intersect
        self.intersect_radius = intersect_radius

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
                centroid_dist = lig_centroid.Distance(res_centroid)
                if centroid_dist > self.distance:
                    continue
                # ligand
                lig_normal = get_ring_normal_vector(lig_centroid, lig_pi_coords)
                # residue
                res_normal = get_ring_normal_vector(res_centroid, res_pi_coords)
                plane_angle = lig_normal.AngleTo(res_normal)
                if not angle_between_limits(plane_angle, *self.plane_angle, ring=True):
                    continue
                c1c2 = lig_centroid.DirectionVector(res_centroid)
                c2c1 = res_centroid.DirectionVector(lig_centroid)
                n1c1c2 = lig_normal.AngleTo(c1c2)
                n2c2c1 = res_normal.AngleTo(c2c1)
                ncc_angle = None
                if angle_between_limits(
                    n1c1c2, *self.normal_to_centroid_angle, ring=True
                ):
                    ncc_angle = n1c1c2
                elif angle_between_limits(
                    n2c2c1, *self.normal_to_centroid_angle, ring=True
                ):
                    ncc_angle = n2c2c1
                if ncc_angle is None:
                    continue
                if self.intersect:
                    # look for point of intersection between both ring planes
                    intersect = self._get_intersect_point(
                        lig_normal, lig_centroid, res_normal, res_centroid
                    )
                    if intersect is None:
                        continue
                    # check if intersection point falls ~within plane ring
                    intersect_dist = min(
                        lig_centroid.Distance(intersect),
                        res_centroid.Distance(intersect),
                    )
                    if intersect_dist <= self.intersect_radius:
                        yield self.metadata(
                            ligand,
                            residue,
                            lig_match,
                            res_match,
                            distance=centroid_dist,
                            plane_angle=degrees(plane_angle),
                            normal_to_centroid_angle=degrees(ncc_angle),
                            intersect_distance=intersect_dist,
                        )
                else:
                    yield self.metadata(
                        ligand,
                        residue,
                        lig_match,
                        res_match,
                        distance=centroid_dist,
                        plane_angle=degrees(plane_angle),
                        normal_to_centroid_angle=degrees(ncc_angle),
                    )

    @staticmethod
    def _get_intersect_point(
        plane_normal,
        plane_centroid,
        tilted_normal,
        tilted_centroid,
    ):
        # intersect line is orthogonal to both planes normal vectors
        intersect_direction = plane_normal.CrossProduct(tilted_normal)
        # setup system of linear equations to solve
        A = np.array(
            [list(plane_normal), list(tilted_normal), list(intersect_direction)]
        )
        if np.linalg.det(A) == 0:
            return None
        tilted_offset = tilted_normal.DotProduct(Geometry.Point3D(*tilted_centroid))
        plane_offset = plane_normal.DotProduct(Geometry.Point3D(*plane_centroid))
        d = np.array([[plane_offset], [tilted_offset], [0.0]])
        # point on intersect line
        point = np.linalg.solve(A, d).T[0]
        point = Geometry.Point3D(*point)
        # find projection of centroid on intersect line using vector projection
        vec = plane_centroid - point
        intersect_direction.Normalize()
        scalar_proj = intersect_direction.DotProduct(vec)
        return point + intersect_direction * scalar_proj
