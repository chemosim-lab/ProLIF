"""
Detecting interactions between residues --- :mod:`prolif.interactions`
======================================================================

You can declare your own interaction class like this::

    class CloseContact(prolif.Interaction):
        def detect(self, res1, res2, threshold=2.0):
            A = res1.xyz
            B = res2.xyz
            squared_dist_matrix = np.add.outer((A*A).sum(axis=-1), 
                                               (B*B).sum(axis=-1)
                                               ) - 2*np.dot(A, B.T)
            if (squared_dist_matrix <= threshold**2).any():
                return True

.. warning:: Your custom class must inherit from :class:`prolif.interactions.Interaction`

The new "CloseContact" class is then automatically added to the list of
interactions available to the fingerprint generator::

    >>> u = mda.Universe(prolif.datafiles.TOP, prolif.datafiles.TRAJ)
    >>> prot = u.select_atoms("protein")
    >>> lig = u.select_atoms("resname ERM")
    >>> fp = prolif.Fingerprint(interactions="all")
    >>> df = fp.run(u.trajectory[0:1], lig, prot).to_dataframe()
    >>> df.xs("CloseContact", level=1, axis=1)
       ASP129.0  VAL201.0
    0         1         1
    >>> lig_mol = prolif.Molecule.from_mda(lig)
    >>> prot_mol = prolif.Molecule.from_mda(prot)
    >>> fp.closecontact(lig_mol[0], prot_mol["ASP129.0"])
    True

"""
import copy
from math import radians
from abc import ABC, ABCMeta, abstractmethod
from .utils import (
    angle_between_limits,
    get_centroid,
    get_ring_normal_vector)
import numpy as np
from rdkit import Chem
from rdkit import Geometry as rdGeometry


_INTERACTIONS = {}


class _InteractionMeta(ABCMeta):
    """Metaclass to register interactions automatically"""
    def __init__(cls, name, bases, classdict):
        type.__init__(cls, name, bases, classdict)
        _INTERACTIONS[name] = cls


class Interaction(ABC, metaclass=_InteractionMeta):
    """Abstract class for interactions

    All interaction classes must inherit this class and define a
    :meth:`~detect` method
    """
    @abstractmethod
    def detect(self, **kwargs):
        raise NotImplementedError(
            "This method must be defined by the subclass")


def get_mapindex(res, index):
    """Get the index of the atom in the original molecule

    Parameters
    ----------
    res : prolif.residue.Residue
        The ligand or residue
    index : int
        The index of the atom in the :class:`~prolif.residue.Residue`

    Returns
    -------
    mapindex : int
        The index of the atom in the :class:`~prolif.molecule.Molecule`
    """
    return res.GetAtomWithIdx(index).GetUnsignedProp("mapindex")


class Hydrophobic(Interaction):
    """Hydrophobic interaction
    
    Parameters
    ----------
    hydrophobic : str
        SMARTS query for hydrophobic atoms
    distance : float
        Cutoff distance for the interaction
    """
    def __init__(self,
                 hydrophobic="[C&!$(C=O)&!$(C#N),S&^3,#17,#35,#53;!+;!-]",
                 distance=4.5):
        self.hydrophobic = Chem.MolFromSmarts(hydrophobic)
        self.distance = distance

    def detect(self, ligand, residue):
        # get atom tuples matching query
        lig_matches = ligand.GetSubstructMatches(self.hydrophobic)
        res_matches = residue.GetSubstructMatches(self.hydrophobic)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                # define ligand atom matching query as 3d point
                lig_atom = rdGeometry.Point3D(*ligand.xyz[lig_match[0]])
                for res_match in res_matches:
                    # define residue atom matching query as 3d point
                    res_atom = rdGeometry.Point3D(*residue.xyz[res_match[0]])
                    # compute distance between points
                    dist = lig_atom.Distance(res_atom)
                    if dist <= self.distance:
                        return (True, get_mapindex(ligand, lig_match[0]), 
                                      get_mapindex(residue, res_match[0]))
        return False, None, None


class _BaseHBond(Interaction):
    """Base class for Hydrogen bond interactions
    
    Parameters
    ----------
    donor : str
        SMARTS for ``[Donor]-[Hydrogen]``
    acceptor : str
        SMARTS for ``[Acceptor]``
    distance : float
        Cutoff distance
    angles : tuple
        Min and max values for the ``[Donor]-[Hydrogen]...[Acceptor]`` angle
    """
    def __init__(self, donor="[O,N,S]-[H]", acceptor="[O,N,F,*-;!+]",
                 distance=3.1, angles=(130, 180)):
        self.donor = Chem.MolFromSmarts(donor)
        self.acceptor = Chem.MolFromSmarts(acceptor)
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, acceptor, donor):
        acceptor_matches = acceptor.GetSubstructMatches(self.acceptor)
        donor_matches = donor.GetSubstructMatches(self.donor)
        if acceptor_matches and donor_matches:
            for donor_match in donor_matches:
                # D-H ... A
                d = rdGeometry.Point3D(*donor.xyz[donor_match[0]])
                h = rdGeometry.Point3D(*donor.xyz[donor_match[1]])
                for acceptor_match in acceptor_matches:
                    a = rdGeometry.Point3D(*acceptor.xyz[acceptor_match[0]])
                    dist = h.Distance(a)
                    if dist <= self.distance:
                        hd = h.DirectionVector(d)
                        ha = h.DirectionVector(a)
                        # get DHA angle
                        angle = hd.AngleTo(ha)
                        if angle_between_limits(angle, *self.angles):
                            return (True, get_mapindex(acceptor,
                                    acceptor_match[0]), get_mapindex(donor,
                                    donor_match[1]))
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
        Cutoff distance
    axd_angles : tuple
        Min and max values for the ``[Acceptor]...[Halogen]-[Donor]`` angle
    xar_angles : tuple
        Min and max values for the ``[R]-[Acceptor]...[Halogen]`` angle
    """
    def __init__(self, donor="[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]",
                 acceptor="[F-,Cl-,Br-,I-,#7,O,P,S,Se,Te,a;!+][*]",
                 distance=3.2, axd_angles=(160, 180), xar_angles=(90, 130)):
        self.donor = Chem.MolFromSmarts(donor)
        self.acceptor = Chem.MolFromSmarts(acceptor)
        self.distance = distance
        self.axd_angles = tuple(radians(i) for i in axd_angles)
        self.xar_angles = tuple(radians(i) for i in xar_angles)

    def detect(self, acceptor, donor):
        acceptor_matches = acceptor.GetSubstructMatches(self.acceptor)
        donor_matches = donor.GetSubstructMatches(self.donor)
        if acceptor_matches and donor_matches:
            for donor_match in donor_matches:
                # D-X ... A distance
                d = rdGeometry.Point3D(*donor.xyz[donor_match[0]])
                x = rdGeometry.Point3D(*donor.xyz[donor_match[1]])
                for acceptor_match in acceptor_matches:
                    a = rdGeometry.Point3D(*acceptor.xyz[acceptor_match[0]])
                    dist = x.Distance(a)
                    if dist <= self.distance:
                        # D-X ... A angle
                        xd = x.DirectionVector(d)
                        xa = x.DirectionVector(a)
                        angle = xd.AngleTo(xa)
                        if angle_between_limits(angle, *self.axd_angles):
                            # X ... A-R angle
                            r = rdGeometry.Point3D(*acceptor.xyz[acceptor_match[1]])
                            ax = a.DirectionVector(x)
                            ar = a.DirectionVector(r)
                            angle = ax.AngleTo(ar)
                            if angle_between_limits(angle, *self.xar_angles):
                                return (True, get_mapindex(acceptor,
                                        acceptor_match[0]), get_mapindex(donor,
                                        donor_match[1]))
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


class _BaseIonic(Interaction):
    """Base class for ionic interactions
    
    Parameters
    ----------
    cation : str
        SMARTS for cation
    anion : str
        SMARTS for anion
    distance : float
        Cutoff distance
    """
    def __init__(self, cation="[*+]", anion="[*-]", distance=5.0):
        self.cation = Chem.MolFromSmarts(cation)
        self.anion = Chem.MolFromSmarts(anion)
        self.distance = distance

    def detect(self, cation, anion):
        anion_matches = anion.GetSubstructMatches(self.anion)
        cation_matches = cation.GetSubstructMatches(self.cation)
        if anion_matches and cation_matches:
            for anion_match in anion_matches:
                a = rdGeometry.Point3D(*anion.xyz[anion_match[0]])
                for cation_match in cation_matches:
                    c = rdGeometry.Point3D(*cation.xyz[cation_match[0]])
                    dist = a.Distance(c)
                    if dist <= self.distance:
                        return (True, get_mapindex(cation, cation_match[0]),
                                      get_mapindex(anion, anion_match[0]))
        return False, None, None


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
        Cutoff distance
    angles : tuple
        Min and max values for the angle between the vector normal to the ring
        plane and the vector going from the centroid to the cation
    """
    def __init__(self, cation="[*+]", pi_ring=("a1:a:a:a:a:a:1",
                 "a1:a:a:a:a:1"), distance=5.0, angles=(0, 30)):
        self.cation = Chem.MolFromSmarts(cation)
        self.pi_ring = [Chem.MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, cation, pi):
        cation_matches = cation.GetSubstructMatches(self.cation)
        for pi_ring in self.pi_ring:
            pi_matches = pi.GetSubstructMatches(pi_ring)
            if cation_matches and pi_matches:
                for cation_match in cation_matches:
                    cat = rdGeometry.Point3D(*cation.xyz[cation_match[0]])
                    for pi_match in pi_matches:
                        # get coordinates of atoms matching pi-system
                        pi_coords = pi.xyz[list(pi_match)]
                        # centroid of pi-system as 3d point
                        centroid  = rdGeometry.Point3D(*get_centroid(pi_coords))
                        # distance between cation and centroid
                        dist = cat.Distance(centroid)
                        if dist <= self.distance:
                            # vector normal to ring plane
                            normal = get_ring_normal_vector(centroid, pi_coords)
                            # vector between the centroid and the charge
                            centroid_cation = centroid.DirectionVector(cat)
                            # compute angle between normal to ring plane and centroid-cation
                            angle = normal.AngleTo(centroid_cation)
                            if angle_between_limits(angle, *self.angles, ring=True):
                                return (True, get_mapindex(cation,
                                        cation_match[0]), get_mapindex(pi,
                                        pi_match[0]))
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
    distance : float
        Cutoff distance
    angles : tuple
        Min and max values for the angle between both vectors normal to the
        ring plane
    """
    def __init__(self, distance, angles, pi_ring=["a1:a:a:a:a:a:1",
                 "a1:a:a:a:a:1"]):
        self.pi_ring = [Chem.MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, ligand, residue):
        for pi_ring in self.pi_ring:
            res_matches = residue.GetSubstructMatches(pi_ring)
            lig_matches = ligand.GetSubstructMatches(pi_ring)
            if lig_matches and res_matches:
                for lig_match in lig_matches:
                    lig_pi_coords = ligand.xyz[list(lig_match)]
                    lig_centroid  = rdGeometry.Point3D(*get_centroid(lig_pi_coords))
                    for res_match in res_matches:
                        res_pi_coords = residue.xyz[list(res_match)]
                        res_centroid  = rdGeometry.Point3D(*get_centroid(res_pi_coords))
                        dist = lig_centroid.Distance(res_centroid)
                        if dist <= self.distance:
                            # ligand
                            lig_normal = get_ring_normal_vector(lig_centroid, lig_pi_coords)
                            # residue
                            res_normal = get_ring_normal_vector(res_centroid, res_pi_coords)
                            # angle
                            angle = res_normal.AngleTo(lig_normal)
                            if angle_between_limits(angle, *self.angles, ring=True):
                                return (True, get_mapindex(ligand,
                                        lig_match[0]), get_mapindex(residue,
                                        res_match[0]))
        return False, None, None


class FaceToFace(_BasePiStacking):
    """Face-to-face Pi-Stacking interaction between a ligand and a residue"""
    def __init__(self, distance=4.4, angles=(0, 30)):
        super().__init__(distance=distance, angles=angles)


class EdgeToFace(_BasePiStacking):
    """Edge-to-face Pi-Stacking interaction between a ligand and a residue"""
    def __init__(self, distance=5.5, angles=(60, 90)):
        super().__init__(distance=distance, angles=angles)


class PiStacking(Interaction):
    """Pi-Stacking interaction (either edge-to-face or face-to-face) between a
    ligand and a residue"""
    def __init__(self):
        self.ftf = FaceToFace()
        self.etf = EdgeToFace()

    def detect(self, ligand, residue):
        ftf = self.ftf.detect(ligand, residue)
        if ftf[0]:
            return ftf
        return self.etf.detect(ligand, residue)


class _BaseMetallic(Interaction):
    """Base class for metal complexation
    
    Parameters
    ----------
    metal : str
        SMARTS for a transition metal
    ligand : str
        SMARTS for a ligand
    distance : float
        Cutoff distance
    """
    def __init__(self, metal="[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]", 
                 ligand="[O,N,*-;!+]", distance=2.8):
        self.metal = Chem.MolFromSmarts(metal)
        self.ligand = Chem.MolFromSmarts(ligand)
        self.distance = distance

    def detect(self, metal, ligand):
        ligand_matches = ligand.GetSubstructMatches(self.ligand)
        metal_matches = metal.GetSubstructMatches(self.metal)
        if ligand_matches and metal_matches:
            for ligand_match in ligand_matches:
                ligand_atom = rdGeometry.Point3D(*ligand.xyz[ligand_match[0]])
                for metal_match in metal_matches:
                    metal_atom = rdGeometry.Point3D(*metal.xyz[metal_match[0]])
                    dist = ligand_atom.Distance(metal_atom)
                    if dist <= self.distance:
                        return (True, get_mapindex(metal, metal_match[0]),
                                      get_mapindex(ligand, ligand_match[0]))
        return False, None, None


class MetalDonor(_BaseMetallic):
    """Metallic interaction between a ligand (metal) and a residue (chelated)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class MetalAcceptor(_BaseMetallic):
    """Metallic interaction between a ligand (chelated) and a residue (metal)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires