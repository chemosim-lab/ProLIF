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

    def __init__(self, topology, coordinates, reference=None, cutoff=6.0, reference_frame=0, residues_list=[]):
        super().__init__(topology, coordinates)
        if not residues_list:
            if reference:
                logger.info('Detecting residues within {} Å of the reference molecule in frame {}'.format(cutoff, reference_frame))
                residues_list = self.detect_pocket_residues(reference, cutoff, frame=reference_frame)
            else:
                logger.info('Considering all protein residues for calculations. This might take a while.')
                residues_list = list(self.top.residues)
        self.residues_list = residues_list

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_frames} frame(s), {len(self.residues_list)} residues, {self.GetNumAtoms()} atoms, {self.GetNumBonds()} bonds, {Chem.GetSSSR(self)} rings"
        return f"<{name}: {params} at 0x{id(self):02x}>"

    @classmethod
    def from_pytraj(cls, traj, **kwargs):
        """Create a protein from a pytraj `Trajectory`"""
        topology = Topology.from_pytraj(traj.topology)
        coordinates = traj.xyz
        return cls(topology, coordinates, **kwargs)

    @classmethod
    def from_mdtraj(cls, traj, **kwargs):
        """Create a trajectory from a mdtraj `Trajectory`"""
        topology = Topology.from_mdtraj(traj.topology)
        coordinates = 10*traj.xyz.astype(float)
        return cls(topology, coordinates, **kwargs)

    def detect_pocket_residues(self, reference, cutoff=6.0, frame=0):
        """Detect residues close to a reference ligand, based on the given frame"""
        residues_list = []
        prot_frame = self.get_frame(frame)
        ref_frame = reference.get_frame(frame)
        ref_points = reference.get_USRlike_atoms(frame)
        for residue in prot_frame:
            # skip residues with centroid far from ligand centroid
            if residue.centroid.Distance(ref_frame.centroid) > 2*cutoff:
                continue
            # skip residues already inside the list
            if residue.resname in residues_list:
                continue
            # iterate over each reference point
            for ref_point in ref_points:
                for atom_crd in residue.xyz:
                    resid_point = rdGeometry.Point3D(*atom_crd)
                    dist = ref_point.Distance(resid_point)
                    if dist <= cutoff:
                        residues_list.append(residue.resname)
                        break
        logger.info('Detected {} residues inside the binding pocket'.format(len(residues_list)))
        return residues_list
