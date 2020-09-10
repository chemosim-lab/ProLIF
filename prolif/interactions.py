import copy
from math import radians
from .utils import (
    angle_between_limits,
    get_centroid,
    get_ring_normal_vector)
from rdkit import Chem
from rdkit import Geometry as rdGeometry


_INTERACTIONS = {}


class _InteractionMeta(type):
    def __init__(cls, name, bases, classdict):
        type.__init__(cls, name, bases, classdict)
        if not (name.startswith("_") or name == "Interaction"):
            _INTERACTIONS[name] = cls


class Interaction(metaclass=_InteractionMeta):
    """Base class for all interactions"""
    def detect(self, **kwargs):
        raise NotImplementedError("This method must be defined by the subclass")


class Hydrophobic(Interaction):
    """Get the presence or absence of an hydrophobic interaction between
    a ResidueFrame and a ligand Frame"""
    def __init__(self, hydrophobic="[#6,S,F,Cl,Br,I;!+;!-]", distance=4.5):
        self.hydrophobic = Chem.MolFromSmarts(hydrophobic)
        self.distance = distance

    def detect(self, ligand, residue):
        """Get the presence or absence of an hydrophobic interaction between
        a ResidueFrame and a ligand Frame"""
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
                        return True
        return False


class _BaseHBond(Interaction):
    """Base class for Hydrogen bond interactions"""
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
                            return True
        return False


class HBDonor(_BaseHBond):
    """Get the presence or absence of a H-bond interaction between
    a residue as an acceptor and a ligand as a donor"""
    def detect(self, ligand, residue):
        return super().detect(residue, ligand)


class HBAcceptor(_BaseHBond):
    """Get the presence or absence of a H-bond interaction between
    a residue as a donor and a ligand as an acceptor"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class _BaseXBond(Interaction):
    """Base class for Halogen bond interactions"""
    def __init__(self, donor="[#6,#7,Si,F,Cl,Br,I][F,Cl,Br,I,At]", 
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
                # D-X ... A
                d = rdGeometry.Point3D(*donor.xyz[donor_match[0]])
                x = rdGeometry.Point3D(*donor.xyz[donor_match[1]])
                for acceptor_match in acceptor_matches:
                    a = rdGeometry.Point3D(*acceptor.xyz[acceptor_match[0]])
                    dist = x.Distance(a)
                    if dist <= self.distance:
                        xd = x.DirectionVector(d)
                        xa = x.DirectionVector(a)
                        angle = xd.AngleTo(xa)
                        if angle_between_limits(angle, *self.axd_angles):
                            r = rdGeometry.Point3D(*acceptor.xyz[acceptor_match[1]])
                            ax = a.DirectionVector(x)
                            ar = a.DirectionVector(r)
                            angle = ax.AngleTo(ar)
                            if angle_between_limits(angle, *self.xar_angles):
                                return True
        return False


class XBAcceptor(_BaseXBond):
    """Get the presence or absence of a Halogen Bond where the residue acts as
    a donor"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class XBDonor(_BaseXBond):
    """Get the presence or absence of a Halogen Bond where the ligand acts as
    a donor"""
    def detect(self, ligand, residue):
        return super().detect(residue, ligand)


class _BaseIonic(Interaction):
    """Base class for ionic interactions"""
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
                        return True
        return False


class Cationic(_BaseIonic):
    """Get the presence or absence of an ionic interaction between a residue
    as an anion and a ligand as a cation"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class Anionic(_BaseIonic):
    """Get the presence or absence of an ionic interaction between a residue
    as a cation and a ligand as an anion"""
    def detect(self, ligand, residue):
        return super().detect(residue, ligand)


class _BaseCationPi(Interaction):
    """Base class for cation-pi interactions"""
    def __init__(self, cation="[*+]", aromatic=["[a]1:[a]:[a]:[a]:[a]:[a]:1", 
                 "[a]1:[a]:[a]:[a]:[a]:1"], distance=5.0, angles=(0, 30)):
        self.cation = Chem.MolFromSmarts(cation)
        self.aromatic = [Chem.MolFromSmarts(s) for s in aromatic]
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, cation, pi):
        cation_matches = cation.GetSubstructMatches(self.cation)
        for aromatic in self.aromatic:
            pi_matches = pi.GetSubstructMatches(aromatic)
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
                                return True
        return False


class PiCation(_BaseCationPi):
    """Get the presence or absence of an interaction between a residue as
    a cation and a ligand as a pi system"""
    def detect(self, ligand, residue):
        return super().detect(residue, ligand)


class CationPi(_BaseCationPi):
    """Get the presence or absence of an interaction between a residue as a
    pi system and a ligand as a cation"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class _BasePiStacking(Interaction):
    """Base class for Pi-Stacking interactions"""
    def __init__(self, distance, angles, aromatic=[
                 "[a]1:[a]:[a]:[a]:[a]:[a]:1", "[a]1:[a]:[a]:[a]:[a]:1"]):
        self.aromatic = [Chem.MolFromSmarts(s) for s in aromatic]
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, ligand, residue):
        for aromatic in self.aromatic:
            res_matches = residue.GetSubstructMatches(aromatic)
            lig_matches = ligand.GetSubstructMatches(aromatic)
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
                                return True
        return False


class FaceToFace(_BasePiStacking):
    """Get the presence or absence of an aromatic face to face interaction
    between a residue and a ligand"""
    def __init__(self, distance=4.4, angles=(0, 30)):
        super().__init__(distance=distance, angles=angles)


class EdgeToFace(_BasePiStacking):
    """Get the presence or absence of an aromatic face to edge interaction
    between a residue and a ligand"""
    def __init__(self, distance=5.5, angles=(60, 90)):
        super().__init__(distance=distance, angles=angles)


class PiStacking(Interaction):
    """Get the presence of any kind of pi-stacking interaction"""
    def __init__(self):
        self.ftf = FaceToFace()
        self.etf = EdgeToFace()

    def detect(self, ligand, residue):
        return self.ftf.detect(ligand, residue) or self.etf.detect(ligand, residue)


class _BaseMetallic(Interaction):
    """Base class for metal complexation"""
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
                        return True
        return False


class MetalDonor(_BaseMetallic):
    """Get the presence or absence of a metal complexation where the ligand is a metal"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class MetalAcceptor(_BaseMetallic):
    """Get the presence or absence of a metal complexation where the residue is a metal"""
    def detect(self, ligand, residue):
        return super().detect(residue, ligand)