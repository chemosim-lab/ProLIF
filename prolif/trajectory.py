import copy
from rdkit import Chem
from .topology import Topology
from .frame import Frame

class Trajectory(Chem.Mol):
    """
    The Trajectory class, which inherits from RDKit `Chem.Mol` class. The
    difference with the Topology class is that the resulting `Chem.Mol` produces
    an iterator of frames with 3D coordinates.
    """

    def __init__(self, topology, coordinates):
        super().__init__(topology)
        self.top = topology
        self.n_frames = len(coordinates)
        self.coordinates = coordinates

    def __iter__(self):
        self.n_frame = 0
        return self

    def __next__(self):
        """Returns a frame with 3D coordinates"""
        if self.n_frame >= self.n_frames:
            raise StopIteration
        xyz = self.coordinates[self.n_frame]
        frame = Frame(self.top, xyz, n_frame=self.n_frame)
        self.n_frame += 1
        return frame

    @classmethod
    def from_pytraj(cls, traj):
        """Create a trajectory from a pytraj `Trajectory`"""
        topology = Topology.from_pytraj(traj.topology)
        coordinates = traj.xyz
        return cls(topology, coordinates)

    @classmethod
    def from_mdtraj(cls, traj):
        """Create a trajectory from a mdtraj `Trajectory`"""
        topology = Topology.from_mdtraj(traj.topology)
        coordinates = 10*traj.xyz.astype(float)
        return cls(topology, coordinates)

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_frames} frame(s), {self.GetNumAtoms()} atoms, {self.GetNumBonds()} bonds, {Chem.GetSSSR(self)} rings"
        return f"<{name}: {params} at 0x{id(self):02x}>"
