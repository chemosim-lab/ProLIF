import json
from os import path
from rdkit import Chem, DataStructs
from rdkit import Geometry as rdGeometry
from math import degrees
from .logger import logger
from .utils import (get_resnumber,
                    getCentroid,
                    isinAngleLimits,
                    getNormalVector)

class Fingerprint:
    """Class that generates an interaction fingerprint between a protein and a ligand"""

    def __init__(self,
        json_file=path.join(path.dirname(__file__),'parameters.json'),
        interactions=['HBdonor','HBacceptor','cation','anion','FaceToFace','EdgeToFace','hydrophobic']):
        # read parameters from json file
        with open(json_file) as data_file:
            logger.debug('Reading JSON parameters file from {}'.format(json_file))
            self.prm = json.load(data_file)
        # create aromatic patterns from json file
        self.AROMATIC_PATTERNS = [ Chem.MolFromSmarts(smart) for smart in self.prm["aromatic"]["smarts"]]
        # read interactions to compute
        self.interactions = interactions
        logger.info('Built fingerprint generator using the following bitstring: {}'.format(' '.join(self.interactions)))


    def __repr__(self):
        return ' '.join(self.interactions)


    def hasHydrophobic(self, ligand, residue):
        """Get the presence or absence of an hydrophobic interaction between
        a residue and a ligand"""
        # define SMARTS query as hydrophobic
        hydrophobic = Chem.MolFromSmarts(self.prm["hydrophobic"]["smarts"])
        # get atom tuples matching query
        lig_matches = ligand.mol.GetSubstructMatches(hydrophobic)
        res_matches = residue.mol.GetSubstructMatches(hydrophobic)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                # define ligand atom matching query as 3d point
                lig_atom = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                for res_match in res_matches:
                    # define residue atom matching query as 3d point
                    res_atom = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    # compute distance between points
                    dist = lig_atom.Distance(res_atom)
                    if dist <= self.prm["hydrophobic"]["distance"]:
                        return 1
        return 0


    def hasHBdonor(self, ligand, residue):
        """Get the presence or absence of a H-bond interaction between
        a residue as an acceptor and a ligand as a donor"""
        # define hbond donor and acceptor smarts queries
        donor    = Chem.MolFromSmarts(self.prm["HBond"]["donor"])
        acceptor = Chem.MolFromSmarts(self.prm["HBond"]["acceptor"])
        # get atom tuples matching queries
        lig_matches = ligand.mol.GetSubstructMatches(donor)
        res_matches = residue.mol.GetSubstructMatches(acceptor)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                # D-H ... A
                d = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                h = rdGeometry.Point3D(*ligand.coordinates[lig_match[1]])
                for res_match in res_matches:
                    a = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    dist = d.Distance(a)
                    if dist <= self.prm["HBond"]["distance"]:
                        # define vector between H and D, and H and A
                        hd = h.DirectionVector(d)
                        ha = h.DirectionVector(a)
                        # get angle between hd and ha in degrees
                        angle = degrees(hd.AngleTo(ha))
                        if isinAngleLimits(angle, *self.prm["HBond"]["angle"]):
                            return 1
        return 0


    def hasHBacceptor(self, ligand, residue):
        """Get the presence or absence of a H-bond interaction between
        a residue as a donor and a ligand as an acceptor"""
        donor    = Chem.MolFromSmarts(self.prm["HBond"]["donor"])
        acceptor = Chem.MolFromSmarts(self.prm["HBond"]["acceptor"])
        lig_matches = ligand.mol.GetSubstructMatches(acceptor)
        res_matches = residue.mol.GetSubstructMatches(donor)
        if lig_matches and res_matches:
            for res_match in res_matches:
                # D-H ... A
                d = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                h = rdGeometry.Point3D(*residue.coordinates[res_match[1]])
                for lig_match in lig_matches:
                    a = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                    dist = d.Distance(a)
                    if dist <= self.prm["HBond"]["distance"]:
                        hd = h.DirectionVector(d)
                        ha = h.DirectionVector(a)
                        # get angle between hd and ha in degrees
                        angle = degrees(hd.AngleTo(ha))
                        if isinAngleLimits(angle, *self.prm["HBond"]["angle"]):
                            return 1
        return 0


    def hasXBdonor(self, ligand, residue):
        """Get the presence or absence of a Halogen Bond where the ligand acts as
        a donor"""
        donor    = Chem.MolFromSmarts(self.prm["XBond"]["donor"])
        acceptor = Chem.MolFromSmarts(self.prm["XBond"]["acceptor"])
        lig_matches = ligand.mol.GetSubstructMatches(donor)
        res_matches = residue.mol.GetSubstructMatches(acceptor)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                # D-X ... A
                d = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                x = rdGeometry.Point3D(*ligand.coordinates[lig_match[1]])
                for res_match in res_matches:
                    a = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    dist = d.Distance(a)
                    if dist <= self.prm["XBond"]["distance"]:
                        xd = x.DirectionVector(d)
                        xa = x.DirectionVector(a)
                        # get angle between hd and ha in degrees
                        angle = degrees(xd.AngleTo(xa))
                        if isinAngleLimits(angle, *self.prm["XBond"]["angle"]):
                            return 1
        return 0


    def hasXBacceptor(self, ligand, residue):
        """Get the presence or absence of a Halogen Bond where the residue acts as
        a donor"""
        donor    = Chem.MolFromSmarts(self.prm["XBond"]["donor"])
        acceptor = Chem.MolFromSmarts(self.prm["XBond"]["acceptor"])
        lig_matches = ligand.mol.GetSubstructMatches(acceptor)
        res_matches = residue.mol.GetSubstructMatches(donor)
        if lig_matches and res_matches:
            for res_match in res_matches:
                # D-X ... A
                d = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                x = rdGeometry.Point3D(*residue.coordinates[res_match[1]])
                for lig_match in lig_matches:
                    a = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                    dist = d.Distance(a)
                    if dist <= self.prm["XBond"]["distance"]:
                        xd = x.DirectionVector(d)
                        xa = x.DirectionVector(a)
                        # get angle between hd and ha in degrees
                        angle = degrees(xd.AngleTo(xa))
                        if isinAngleLimits(angle, *self.prm["XBond"]["angle"]):
                            return 1
        return 0


    def hasCationic(self, ligand, residue):
        """Get the presence or absence of an ionic interaction between a residue
        as an anion and a ligand as a cation"""
        cation = Chem.MolFromSmarts(self.prm["ionic"]["cation"])
        anion  = Chem.MolFromSmarts(self.prm["ionic"]["anion"])
        lig_matches = ligand.mol.GetSubstructMatches(cation)
        res_matches = residue.mol.GetSubstructMatches(anion)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                c = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                for res_match in res_matches:
                    a = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    dist = c.Distance(a)
                    if dist <= self.prm["ionic"]["distance"]:
                        return 1
        return 0


    def hasAnionic(self, ligand, residue):
        """Get the presence or absence of an ionic interaction between a residue
        as a cation and a ligand as an anion"""
        cation = Chem.MolFromSmarts(self.prm["ionic"]["cation"])
        anion  = Chem.MolFromSmarts(self.prm["ionic"]["anion"])
        lig_matches = ligand.mol.GetSubstructMatches(anion)
        res_matches = residue.mol.GetSubstructMatches(cation)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                a = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                for res_match in res_matches:
                    c = rdGeometry.Point3D(*residue.coordinates[lig_match[0]])
                    dist = a.Distance(c)
                    if dist <= self.prm["ionic"]["distance"]:
                        return 1
        return 0


    def hasPiCation(self, ligand, residue):
        """Get the presence or absence of an interaction between a residue as
        a cation and a ligand as a pi system"""
        # check for residues matching cation smarts query
        cation = Chem.MolFromSmarts(self.prm["ionic"]["cation"])
        res_matches = residue.mol.GetSubstructMatches(cation)
        if res_matches:
            # check for ligand matching pi query
            for pi in self.AROMATIC_PATTERNS:
                lig_matches = ligand.mol.GetSubstructMatches(pi)
                for res_match in res_matches:
                    # cation as 3d point
                    cat = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    for lig_match in lig_matches:
                        # get coordinates of atoms matching pi-system
                        pi_coords = ligand.coordinates[list(lig_match)]
                        # centroid of pi-system as 3d point
                        centroid  = rdGeometry.Point3D(*getCentroid(pi_coords))
                        # distance between cation and centroid
                        dist = cat.Distance(centroid)
                        if dist <= self.prm["pi-cation"]["distance"]:
                            # compute angle between centroid-normal and centroid-cation vectors
                            # get vector in the ring plane
                            atom_plane = rdGeometry.Point3D(*pi_coords[0])
                            centroid_plane = centroid.DirectionVector(atom_plane)
                            # vector normal to the ring plane
                            centroid_normal = rdGeometry.Point3D(*getNormalVector(centroid_plane))
                            # vector between the centroid and the charge
                            centroid_charge = centroid.DirectionVector(cat)
                            angle = degrees(centroid_normal.AngleTo(centroid_charge))
                            if isinAngleLimits(angle, *self.prm["pi-cation"]["angle"]):
                                return 1
        return 0


    def hasCationPi(self, ligand, residue):
        """Get the presence or absence of an interaction between a residue as a
        pi system and a ligand as a cation"""
        # check for ligand matching cation smarts query
        cation = Chem.MolFromSmarts(self.prm["ionic"]["cation"])
        lig_matches = ligand.mol.GetSubstructMatches(cation)
        if lig_matches:
            # check for residue matching pi query
            for pi in self.AROMATIC_PATTERNS:
                res_matches = residue.mol.GetSubstructMatches(pi)
                for lig_match in lig_matches:
                    cat = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                    for res_match in res_matches:
                        # get coordinates of atoms matching pi-system
                        pi_coords = residue.coordinates[list(res_match)]
                        # centroid of pi-system as 3d point
                        centroid  = rdGeometry.Point3D(*getCentroid(pi_coords))
                        # distance between cation and centroid
                        dist = cat.Distance(centroid)
                        if dist <= self.prm["pi-cation"]["distance"]:
                            # compute angle between centroid-normal and centroid-cation vectors
                            # get vector in the ring plane
                            atom_plane = rdGeometry.Point3D(*pi_coords[0])
                            centroid_plane = centroid.DirectionVector(atom_plane)
                            # vector normal to the ring plane
                            centroid_normal = rdGeometry.Point3D(*getNormalVector(centroid_plane))
                            # vector between the centroid and the charge
                            centroid_charge = centroid.DirectionVector(cat)
                            angle = degrees(centroid_normal.AngleTo(centroid_charge))
                            if isinAngleLimits(angle, *self.prm["pi-cation"]["angle"]):
                                return 1
        return 0


    def hasFaceToFace(self, ligand, residue):
        """Get the presence or absence of an aromatic face to face interaction
        between a residue and a ligand"""
        for pi_res in self.AROMATIC_PATTERNS:
            for pi_lig in self.AROMATIC_PATTERNS:
                res_matches = residue.mol.GetSubstructMatches(pi_res)
                lig_matches = ligand.mol.GetSubstructMatches(pi_lig)
                for lig_match in lig_matches:
                    lig_pi_coords = ligand.coordinates[list(lig_match)]
                    lig_centroid  = rdGeometry.Point3D(*getCentroid(lig_pi_coords))
                    for res_match in res_matches:
                        res_pi_coords = residue.coordinates[list(res_match)]
                        res_centroid  = rdGeometry.Point3D(*getCentroid(res_pi_coords))
                        dist = lig_centroid.Distance(res_centroid)
                        if dist <= self.prm["aromatic"]["distance"]:
                            # ligand
                            lig_plane = rdGeometry.Point3D(*lig_pi_coords[0])
                            lig_centroid_plane = lig_centroid.DirectionVector(lig_plane)
                            lig_normal = rdGeometry.Point3D(*getNormalVector(lig_centroid_plane))
                            # residue
                            res_plane = rdGeometry.Point3D(*res_pi_coords[0])
                            res_centroid_plane = res_centroid.DirectionVector(res_plane)
                            res_normal = rdGeometry.Point3D(*getNormalVector(res_centroid_plane))
                            # angle
                            angle = degrees(res_normal.AngleTo(lig_normal))
                            if isinAngleLimits(angle, *self.prm["aromatic"]["FaceToFace"]):
                                return 1
        return 0


    def hasEdgeToFace(self, ligand, residue):
        """Get the presence or absence of an aromatic face to edge interaction
        between a residue and a ligand"""
        for pi_res in self.AROMATIC_PATTERNS:
            for pi_lig in self.AROMATIC_PATTERNS:
                res_matches = residue.mol.GetSubstructMatches(pi_res)
                lig_matches = ligand.mol.GetSubstructMatches(pi_lig)
                for lig_match in lig_matches:
                    lig_pi_coords = ligand.coordinates[list(lig_match)]
                    lig_centroid  = rdGeometry.Point3D(*getCentroid(lig_pi_coords))
                    for res_match in res_matches:
                        res_pi_coords = residue.coordinates[list(res_match)]
                        res_centroid  = rdGeometry.Point3D(*getCentroid(res_pi_coords))
                        dist = lig_centroid.Distance(res_centroid)
                        if dist <= self.prm["aromatic"]["distance"]:
                            # ligand
                            lig_plane = rdGeometry.Point3D(*lig_pi_coords[0])
                            lig_centroid_plane = lig_centroid.DirectionVector(lig_plane)
                            lig_normal = rdGeometry.Point3D(*getNormalVector(lig_centroid_plane))
                            # residue
                            res_plane = rdGeometry.Point3D(*res_pi_coords[0])
                            res_centroid_plane = res_centroid.DirectionVector(res_plane)
                            res_normal = rdGeometry.Point3D(*getNormalVector(res_centroid_plane))
                            # angle
                            angle = degrees(res_normal.AngleTo(lig_normal))
                            if isinAngleLimits(angle, *self.prm["aromatic"]["EdgeToFace"]):
                                return 1
        return 0


    def hasMetalDonor(self, ligand, residue):
        """Get the presence or absence of a metal complexation where the ligand is a metal"""
        metal = Chem.MolFromSmarts(self.prm["metallic"]["metal"])
        lig   = Chem.MolFromSmarts(self.prm["metallic"]["ligand"])
        lig_matches = ligand.mol.GetSubstructMatches(metal)
        res_matches = residue.mol.GetSubstructMatches(lig)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                lig_atom = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                for res_match in res_matches:
                    res_atom = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    dist = lig_atom.Distance(res_atom)
                    if dist <= self.prm["metallic"]["distance"]:
                        return 1
        return 0


    def hasMetalAcceptor(self, ligand, residue):
        """Get the presence or absence of a metal complexation where the residue is a metal"""
        metal = Chem.MolFromSmarts(self.prm["metallic"]["metal"])
        lig   = Chem.MolFromSmarts(self.prm["metallic"]["ligand"])
        lig_matches = ligand.mol.GetSubstructMatches(lig)
        res_matches = residue.mol.GetSubstructMatches(metal)
        if lig_matches and res_matches:
            for lig_match in lig_matches:
                lig_atom = rdGeometry.Point3D(*ligand.coordinates[lig_match[0]])
                for res_match in res_matches:
                    res_atom = rdGeometry.Point3D(*residue.coordinates[res_match[0]])
                    dist = lig_atom.Distance(res_atom)
                    if dist <= self.prm["metallic"]["distance"]:
                        return 1
        return 0


    def generateBitstring(self, ligand, residue):
        """Generate the complete bitstring for the interactions of a residue with a ligand"""
        bitstring = []
        for interaction in self.interactions:
            if   interaction == 'HBdonor':
                bitstring.append(self.hasHBdonor(ligand, residue))
            elif interaction == 'HBacceptor':
                bitstring.append(self.hasHBacceptor(ligand, residue))
            elif interaction == 'XBdonor':
                bitstring.append(self.hasXBdonor(ligand, residue))
            elif interaction == 'XBacceptor':
                bitstring.append(self.hasXBdonor(ligand, residue))
            elif interaction == 'cation':
                bitstring.append(self.hasCationic(ligand, residue))
            elif interaction == 'anion':
                bitstring.append(self.hasAnionic(ligand, residue))
            elif interaction == 'FaceToFace':
                bitstring.append(self.hasFaceToFace(ligand, residue))
            elif interaction == 'EdgeToFace':
                bitstring.append(self.hasEdgeToFace(ligand, residue))
            elif interaction == 'pi-cation':
                bitstring.append(self.hasPiCation(ligand, residue))
            elif interaction == 'cation-pi':
                bitstring.append(self.hasCationPi(ligand, residue))
            elif interaction == 'hydrophobic':
                bitstring.append(self.hasHydrophobic(ligand, residue))
            elif interaction == 'MBdonor':
                bitstring.append(self.hasMetalDonor(ligand, residue))
            elif interaction == 'MBacceptor':
                bitstring.append(self.hasMetalAcceptor(ligand, residue))
        return ''.join(str(bit) for bit in bitstring)


    def generateIFP(self, ligand, protein):
        """Generates the complete IFP from each residue's bitstring"""
        IFPvector = DataStructs.ExplicitBitVect(len(self.interactions)*len(protein.residues))
        i=0
        IFP = ''
        for residue in sorted(protein.residues, key=get_resnumber):
            bitstring = self.generateBitstring(ligand, protein.residues[residue])
            for bit in bitstring:
                if bit == '1':
                    IFPvector.SetBit(i)
                i+=1
            IFP += bitstring
        ligand.setIFP(IFP, IFPvector)
