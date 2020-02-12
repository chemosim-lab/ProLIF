import copy
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .residue import ResidueFrame

class Frame(Chem.Mol):
    """The Frame class, which inherits from RDKit `Chem.Mol` class.

    Contains a conformer of the whole molecule. Looping over a frame returns a
    ResidueFrame."""

    def __init__(self, topology, xyz, n_frame=0):
        super().__init__(topology)
        conformer = Chem.Conformer(self.GetNumAtoms())
        for atom in self.GetAtoms():
            atom_index = atom.GetIdx()
            position = Chem.rdGeometry.Point3D(*xyz[atom_index])
            conformer.SetAtomPosition(atom_index, position)
        self.AddConformer(conformer)
        self.n_frame = n_frame
        self.xyz = xyz
        self.n_residues = len(topology.residues_list)
        self.residues = topology.residues

    def __iter__(self):
        self.n_residue = 0
        return self

    def __next__(self):
        if self.n_residue >= self.n_residues:
            raise StopIteration
        resname = list(self.residues.keys())[self.n_residue]
        atom_map = self.residues.atom_map[resname]
        atom_xyz_indices = [atom_map[atom.GetIdx()] for atom in self.residues[resname].GetAtoms()]
        residue = ResidueFrame(self.residues[resname], self.xyz[atom_xyz_indices])
        self.n_residue += 1
        return residue

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"frame #{self.n_frame} with centroid {list(self.centroid)}"
        return f"<{name}: {params} at 0x{id(self):02x}>"

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())
