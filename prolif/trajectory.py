import copy
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .topology import Topology

class Trajectory(Chem.Mol):
    """
    The Trajectory class, which inherits from RDKit `Chem.Mol` class.
    The difference with the Topology class is that the resulting `Chem.Mol` has
    a generator of conformers with 3D coordinates.
    """

    def __init__(self, topology, coordinates):
        super().__init__(topology)
        self._coordinates = coordinates

    def __iter__(self):
        conformer = Chem.Conformer(self.GetNumAtoms())
        for xyz in self._coordinates:
            for atom in self.GetAtoms():
                i = atom.GetIdx()
                position = Chem.rdGeometry.Point3D(*xyz[i])
                conformer.SetAtomPosition(i, position)
            id = self.AddConformer(conformer)
            self.conformer = self.GetConformer(id)
            self.centroid = rdMolTransforms.ComputeCentroid(self.conformer)
            self.xyz = xyz
            yield self
            self.RemoveConformer(id)

    @classmethod
    def from_pytraj(cls, traj):
        """Create a trajectory from a pytraj `Trajectory`"""
        topology = Topology.from_pytraj(traj.topology)
        coordinates = (frame.xyz for frame in traj)
        return cls(topology, coordinates)

    @property
    def coordinates(self):
        return self.xyz
