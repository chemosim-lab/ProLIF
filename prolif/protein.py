from rdkit import Chem
from rdkit import Geometry as rdGeometry
from .trajectory import Trajectory
from .logger import logger


class Protein(Trajectory):
    """
    The Protein class, which inherits from the `prolif.Trajectory` class.

    """

    def __init__(self, topology, coordinates, reference=None, cutoff=6.0, residues_list=[]):
        super().__init__(topology, coordinates)
        if not residues_list:
            if reference:
                logger.info('Detecting residues within {} Å of the reference molecule'.format(cutoff))
                residues_list = self.detect_pocket_residues(reference, cutoff)
            else:
                logger.info('Considering all protein residues for calculations. This might take a while.')
                residues_list = list(self.top.residues)
        self.residues_list = residues_list

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_frames} frame(s), {self.top.residues} residues, {self.GetNumAtoms()} atoms, {self.GetNumBonds()} bonds, {Chem.GetSSSR(self)} rings"
        return f"<{name}: {params} at 0x{id(self):02x}>"

    def detect_pocket_residues(self, reference, cutoff=6.0):
        """Detect residues close to a reference ligand"""
        residues_list = []
        frame = next(iter(self))
        for ref_point in reference.get_USRlike_atoms():
            for residue in frame:
                if residue.centroid.Distance(ref_point) > 10:
                    # skip residues with centroid far from ligand reference point
                    continue
                if residue.resname in residues_list:
                    # skip residues already inside the list
                    continue
                for atom_crd in residue.xyz:
                    resid_point = rdGeometry.Point3D(*atom_crd)
                    dist = ref_point.Distance(resid_point)
                    if dist <= cutoff:
                        residues_list.append(residue.resname)
                        break
        logger.info('Detected {} residues inside the binding pocket'.format(len(residues_list)))
        return residues_list
