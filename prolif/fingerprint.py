import logging, copy
from os import path
from math import radians
from rdkit import Chem, DataStructs
from rdkit import Geometry as rdGeometry
from .parameters import RULES
from .utils import (
    angle_between_limits,
    get_resnumber,
    get_centroid,
    get_ring_normal_vector)

logger = logging.getLogger("prolif")

class FingerprintFactory:
    """Class that generates an interaction fingerprint between a protein and a ligand"""

    def __init__(self, rules=None,
        interactions=['HBdonor','HBacceptor','Cation','Anion','PiStacking','Hydrophobic']):
        # read parameters
        if rules:
            logger.debug("Using supplied geometric rules")
            self.rules = rules
        else:
            _path = path.join(path.dirname(__file__), 'parameters.py')
            logger.debug(f"Using default geometric rules from {_path}")
            self.rules = copy.deepcopy(RULES)
        # convert angles from degrees to radian
        for i in range(2):
            for key in ["HBond", "Cation-Pi"]:
                self.rules[key]["angle"][i] = radians(self.rules[key]["angle"][i])
            for key in ["AXD","XAR"]:
                self.rules["XBond"]["angle"][key][i] = radians(self.rules["XBond"]["angle"][key][i])
            for key in ["FaceToFace", "EdgeToFace"]:
                self.rules["Aromatic"][key]["angle"][i] = radians(self.rules["Aromatic"][key]["angle"][i])
        # create SMARTS
        self.SMARTS = {
            "Aromatic": [Chem.MolFromSmarts(s) for s in self.rules["Aromatic"]["smarts"]],
            "Hydrophobic": Chem.MolFromSmarts(self.rules["Hydrophobic"]["smarts"]),
            "HBdonor": Chem.MolFromSmarts(self.rules["HBond"]["donor"]),
            "HBacceptor": Chem.MolFromSmarts(self.rules["HBond"]["acceptor"]),
            "XBdonor": Chem.MolFromSmarts(self.rules["XBond"]["donor"]),
            "XBacceptor": Chem.MolFromSmarts(self.rules["XBond"]["acceptor"]),
            "Cation": Chem.MolFromSmarts(self.rules["Ionic"]["cation"]),
            "Anion": Chem.MolFromSmarts(self.rules["Ionic"]["anion"]),
            "Metal": Chem.MolFromSmarts(self.rules["Metallic"]["metal"]),
            "Ligand": Chem.MolFromSmarts(self.rules["Metallic"]["ligand"]),
        }
        # read interactions to compute
        self.interactions = interactions
        logger.info('The fingerprint factory will generate the following bitstring: {}'.format(' '.join(self.interactions)))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{len(self.interactions)} interactions"
        return f"<{name}: {params} at 0x{id(self):02x}>"

    def get_hydrophobic(self, ligand, residue):
        """Get the presence or absence of an hydrophobic interaction between
        a ResidueFrame and a ligand Frame"""
        # define SMARTS query as hydrophobic
        hydrophobic = self.SMARTS["Hydrophobic"]
        # get atom tuples matching query
        lig_matches = ligand.GetSubstructMatches(hydrophobic)
        res_matches = residue.GetSubstructMatches(hydrophobic)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                # define ligand atom matching query as 3d point
                lig_atom = rdGeometry.Point3D(*ligand.xyz[lig_match[0]])
                for res_match in res_matches:
                    # define residue atom matching query as 3d point
                    res_atom = rdGeometry.Point3D(*residue.xyz[res_match[0]])
                    # compute distance between points
                    dist = lig_atom.Distance(res_atom)
                    if dist <= self.rules["Hydrophobic"]["distance"]:
                        return 1
        return 0

    def get_hbond_donor(self, ligand, residue):
        """Get the presence or absence of a H-bond interaction between
        a residue as an acceptor and a ligand as a donor"""
        return self._get_hbond(residue, ligand)

    def get_hbond_acceptor(self, ligand, residue):
        """Get the presence or absence of a H-bond interaction between
        a residue as a donor and a ligand as an acceptor"""
        return self._get_hbond(ligand, residue)

    def _get_hbond(self, acceptor_frame, donor_frame):
        """Get the presence or absence of a Hydrogen Bond between two frames"""
        donor = self.SMARTS["HBdonor"]
        acceptor = self.SMARTS["HBacceptor"]
        acceptor_matches = acceptor_frame.GetSubstructMatches(acceptor)
        donor_matches = donor_frame.GetSubstructMatches(donor)
        if acceptor_matches and donor_matches:
            for donor_match in donor_matches:
                # D-H ... A
                d = rdGeometry.Point3D(*donor_frame.xyz[donor_match[0]])
                h = rdGeometry.Point3D(*donor_frame.xyz[donor_match[1]])
                for acceptor_match in acceptor_matches:
                    a = rdGeometry.Point3D(*acceptor_frame.xyz[acceptor_match[0]])
                    dist = h.Distance(a)
                    if dist <= self.rules["HBond"]["distance"]:
                        dh = d.DirectionVector(h)
                        ha = h.DirectionVector(a)
                        # get angle between dh and ha
                        angle = dh.AngleTo(ha)
                        if angle_between_limits(angle, *self.rules["HBond"]["angle"]):
                            return 1
        return 0

    def get_xbond_donor(self, ligand, residue):
        """Get the presence or absence of a Halogen Bond where the ligand acts as
        a donor"""
        return self._get_xbond(residue, ligand)

    def get_xbond_acceptor(self, ligand, residue):
        """Get the presence or absence of a Halogen Bond where the residue acts as
        a donor"""
        return self._get_xbond(ligand, residue)

    def _get_xbond(self, acceptor_frame, donor_frame):
        """Get the presence or absence of a Halogen Bond between two frames"""
        donor = self.SMARTS["XBdonor"]
        acceptor = self.SMARTS["XBacceptor"]
        acceptor_matches = acceptor_frame.GetSubstructMatches(acceptor)
        donor_matches = donor_frame.GetSubstructMatches(donor)
        if acceptor_matches and donor_matches:
            for donor_match in donor_matches:
                # D-X ... A
                d = rdGeometry.Point3D(*donor_frame.xyz[donor_match[0]])
                x = rdGeometry.Point3D(*donor_frame.xyz[donor_match[1]])
                for acceptor_match in acceptor_matches:
                    a = rdGeometry.Point3D(*acceptor_frame.xyz[acceptor_match[0]])
                    dist = x.Distance(a)
                    if dist <= self.rules["XBond"]["distance"]:
                        xd = x.DirectionVector(d)
                        xa = x.DirectionVector(a)
                        angle = xd.AngleTo(xa)
                        if angle_between_limits(angle, *self.rules["XBond"]["angle"]["AXD"]):
                            r = rdGeometry.Point3D(*acceptor_frame.xyz[acceptor_match[1]])
                            ax = a.DirectionVector(x)
                            ar = a.DirectionVector(r)
                            angle = ax.AngleTo(ar)
                            if angle_between_limits(angle, *self.rules["XBond"]["angle"]["XAR"]):
                                return 1
        return 0

    def get_cationic(self, ligand, residue):
        """Get the presence or absence of an ionic interaction between a residue
        as an anion and a ligand as a cation"""
        return self._get_ionic(ligand, residue)

    def get_anionic(self, ligand, residue):
        """Get the presence or absence of an ionic interaction between a residue
        as a cation and a ligand as an anion"""
        return self._get_ionic(residue, ligand)

    def _get_ionic(self, cation_frame, anion_frame):
        """Get the presence or absence of an ionic interaction between two frames"""
        cation = self.SMARTS["Cation"]
        anion  = self.SMARTS["Anion"]
        anion_matches = anion_frame.GetSubstructMatches(anion)
        cation_matches = cation_frame.GetSubstructMatches(cation)
        if anion_matches and cation_matches:
            for anion_match in anion_matches:
                a = rdGeometry.Point3D(*anion_frame.xyz[anion_match[0]])
                for cation_match in cation_matches:
                    c = rdGeometry.Point3D(*cation_frame.xyz[cation_match[0]])
                    dist = a.Distance(c)
                    if dist <= self.rules["Ionic"]["distance"]:
                        return 1
        return 0

    def get_pi_cation(self, ligand, residue):
        """Get the presence or absence of an interaction between a residue as
        a cation and a ligand as a pi system"""
        return self._get_cation_pi(residue, ligand)

    def get_cation_pi(self, ligand, residue):
        """Get the presence or absence of an interaction between a residue as a
        pi system and a ligand as a cation"""
        return self._get_cation_pi(ligand, residue)

    def _get_cation_pi(self, cation_frame, pi_frame):
        """Get the presence or absence of a cation-pi interaction between two frames"""
        # check for matches with cation smarts query
        cation = self.SMARTS["Cation"]
        cation_matches = cation_frame.GetSubstructMatches(cation)
        for pi in self.SMARTS["Aromatic"]:
            pi_matches = pi_frame.GetSubstructMatches(pi)
            if cation_matches and pi_matches:
                for cation_match in cation_matches:
                    cat = rdGeometry.Point3D(*cation_frame.xyz[cation_match[0]])
                    for pi_match in pi_matches:
                        # get coordinates of atoms matching pi-system
                        pi_coords = pi_frame.xyz[list(pi_match)]
                        # centroid of pi-system as 3d point
                        centroid  = rdGeometry.Point3D(*get_centroid(pi_coords))
                        # distance between cation and centroid
                        dist = cat.Distance(centroid)
                        if dist <= self.rules["Cation-Pi"]["distance"]:
                            # vector normal to ring plane
                            normal = get_ring_normal_vector(centroid, pi_coords)
                            # vector between the centroid and the charge
                            centroid_cation = centroid.DirectionVector(cat)
                            # compute angle between normal to ring plane and centroid-cation
                            angle = normal.AngleTo(centroid_cation)
                            if angle_between_limits(angle, *self.rules["Cation-Pi"]["angle"], ring=True):
                                return 1
        return 0

    def get_pi_stacking(self, ligand, residue):
        """Get the presence of any kind of pi-stacking interaction"""
        return self.get_face_to_face(ligand, residue) or self.get_edge_to_face(ligand, residue)

    def get_face_to_face(self, ligand, residue):
        """Get the presence or absence of an aromatic face to face interaction
        between a residue and a ligand"""
        return self._get_pi_stacking(ligand, residue, kind="FaceToFace")

    def get_edge_to_face(self, ligand, residue):
        """Get the presence or absence of an aromatic face to edge interaction
        between a residue and a ligand"""
        return self._get_pi_stacking(ligand, residue, kind="EdgeToFace")

    def _get_pi_stacking(self, ligand, residue, kind="FaceToFace"):
        """Get the presence or absence of pi-stacking interaction
        between a residue and a ligand"""
        for pi in self.SMARTS["Aromatic"]:
            res_matches = residue.GetSubstructMatches(pi)
            lig_matches = ligand.GetSubstructMatches(pi)
            if lig_matches and res_matches:
                for lig_match in lig_matches:
                    lig_pi_coords = ligand.xyz[list(lig_match)]
                    lig_centroid  = rdGeometry.Point3D(*get_centroid(lig_pi_coords))
                    for res_match in res_matches:
                        res_pi_coords = residue.xyz[list(res_match)]
                        res_centroid  = rdGeometry.Point3D(*get_centroid(res_pi_coords))
                        dist = lig_centroid.Distance(res_centroid)
                        if dist <= self.rules["Aromatic"][kind]["distance"]:
                            # ligand
                            lig_normal = get_ring_normal_vector(lig_centroid, lig_pi_coords)
                            # residue
                            res_normal = get_ring_normal_vector(res_centroid, res_pi_coords)
                            # angle
                            angle = res_normal.AngleTo(lig_normal)
                            if angle_between_limits(angle, *self.rules["Aromatic"][kind]["angle"], ring=True):
                                return 1
        return 0

    def get_metal_donor(self, ligand, residue):
        """Get the presence or absence of a metal complexation where the ligand is a metal"""
        return self._get_metallic(ligand, residue)

    def get_metal_acceptor(self, ligand, residue):
        """Get the presence or absence of a metal complexation where the residue is a metal"""
        return self._get_metallic(residue, ligand)

    def _get_metallic(self, metal_frame, ligand_frame):
        """Get the presence or absence of a metal complexation"""
        metal = self.SMARTS["Metal"]
        ligand = self.SMARTS["Ligand"]
        ligand_matches = ligand_frame.GetSubstructMatches(ligand)
        metal_matches = metal_frame.GetSubstructMatches(metal)
        if ligand_matches and metal_matches:
            for ligand_match in ligand_matches:
                ligand_atom = rdGeometry.Point3D(*ligand_frame.xyz[ligand_match[0]])
                for metal_match in metal_matches:
                    metal_atom = rdGeometry.Point3D(*metal_frame.xyz[metal_match[0]])
                    dist = ligand_atom.Distance(metal_atom)
                    if dist <= self.rules["Metallic"]["distance"]:
                        return 1
        return 0

    def generate_bitstring(self, ligand, residue):
        """Generate the complete bitstring for the interactions of a residue with a ligand"""
        bitstring = []
        for interaction in self.interactions:
            if   interaction == 'HBdonor':
                bitstring.append(self.get_hbond_donor(ligand, residue))
            elif interaction == 'HBacceptor':
                bitstring.append(self.get_hbond_acceptor(ligand, residue))
            elif interaction == 'XBdonor':
                bitstring.append(self.get_xbond_donor(ligand, residue))
            elif interaction == 'XBacceptor':
                bitstring.append(self.get_xbond_acceptor(ligand, residue))
            elif interaction == 'Cation':
                bitstring.append(self.get_cationic(ligand, residue))
            elif interaction == 'Anion':
                bitstring.append(self.get_anionic(ligand, residue))
            elif interaction == 'PiStacking':
                bitstring.append(self.get_pi_stacking(ligand, residue))
            elif interaction == 'FaceToFace':
                bitstring.append(self.get_face_to_face(ligand, residue))
            elif interaction == 'EdgeToFace':
                bitstring.append(self.get_edge_to_face(ligand, residue))
            elif interaction == 'Pi-Cation':
                bitstring.append(self.get_pi_cation(ligand, residue))
            elif interaction == 'Cation-Pi':
                bitstring.append(self.get_cation_pi(ligand, residue))
            elif interaction == 'Hydrophobic':
                bitstring.append(self.get_hydrophobic(ligand, residue))
            elif interaction == 'MBdonor':
                bitstring.append(self.get_metal_donor(ligand, residue))
            elif interaction == 'MBacceptor':
                bitstring.append(self.get_metal_acceptor(ligand, residue))
        return ''.join(str(bit) for bit in bitstring)


    def generate_ifp(self, ligand_frame, protein_frame):
        """Generates the complete IFP between two Frames"""
        IFPvector = DataStructs.ExplicitBitVect(len(self.interactions)*len(protein_frame.residues.keys()))
        i=0
        IFP = ''
        for resname, residue in protein_frame.residues.items():
            bitstring = self.generate_bitstring(ligand_frame, residue)
            for bit in bitstring:
                if bit == '1':
                    IFPvector.SetBit(i)
                i+=1
            IFP += bitstring
        return IFP, IFPvector
