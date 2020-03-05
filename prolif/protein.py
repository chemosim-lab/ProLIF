import logging
from rdkit import Chem
from rdkit import Geometry as rdGeometry
from .topology import Topology
from .trajectory import Trajectory

logger = logging.getLogger("prolif")


class Protein(Trajectory):
    """
    The Protein class, which inherits from the `prolif.Trajectory` class.

    """

    def __init__(self, topology, coordinates, reference=None, cutoff=6.0, reference_frame=0, residues_list=[], name="protein"):
        super().__init__(topology, coordinates, name=name)
        if not residues_list:
            if reference:
                self.reference = reference
                self.reference_frame = reference_frame
                self.cutoff = cutoff
                logger.info('Detecting residues within {} Å of the reference molecule in frame {}'.format(cutoff, reference_frame))
                self.detect_pocket_residues(reference, cutoff, frame=reference_frame)
            else:
                logger.info('Considering all protein residues for calculations. This might take a while.')

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_frames} frame(s), {len(self.residues_list)} residues, {self.GetNumAtoms()} atoms, {self.GetNumBonds()} bonds, {Chem.GetSSSR(self)} rings"
        return f"<{name}: {params} at 0x{id(self):02x}>"

    def detect_pocket_residues(self, reference, cutoff=6.0, frame=0, prot_frame=None, ref_frame=None):
        """Detect residues close to a reference ligand, based on the given frame"""
        residues_list = []
        # set which frame to use
        lig_frame = ref_frame if ref_frame else frame
        prot_frame = prot_frame if prot_frame else frame
        logger.info("Detecting pocket residues using frame #{} for the protein and frame #{} for the reference ligand".format(prot_frame, lig_frame))
        prot_frame = self.get_frame(prot_frame)
        ref_frame = reference.get_frame(lig_frame)
        ref_points = reference.get_USRlike_atoms(lig_frame)
        for residue in prot_frame:
            # skip residues already inside the list
            if residue.resname in residues_list:
                continue
            # skip residues with centroid far from ligand centroid
            if residue.centroid.Distance(ref_frame.centroid) > 2*cutoff:
                continue
            # iterate over each reference point
            for ref_point in ref_points:
                for atom_crd in residue.xyz:
                    resid_point = rdGeometry.Point3D(*atom_crd)
                    dist = ref_point.Distance(resid_point)
                    if dist <= cutoff:
                        residues_list.append(residue.resname)
                        break
                if residue.resname in residues_list:
                    break
        logger.info('Detected {} residues inside the binding pocket'.format(len(residues_list)))
        self.top.residues_list = residues_list

    @property
    def residues_list(self):
        return self.top.residues_list
