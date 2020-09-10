from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .residue import Residue
from .utils import get_residue_components, get_resid, ResidueId

class Molecule(Chem.Mol):
    """Molecule class"""

    def __init__(self, mol):
        super().__init__(mol)
        residues = {}
        atom_map = {}
        conformers = {}
        xyz = self.GetConformer().GetPositions()

        for atom in self.GetAtoms():
            resid = get_resid(atom)
            try:
                residues[resid]
            except KeyError:
                residues[resid] = Chem.RWMol()
                atom_map[resid] = {}
                conformers[resid] = Chem.Conformer()
            finally:
                idx = residues[resid].AddAtom(atom)
                atom_map[resid][atom.GetIdx()] = idx
                conformers[resid].SetAtomPosition(idx, xyz[atom.GetIdx()])

        for bond in self.GetBonds():
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            resid1 = get_resid(a1)
            resid2 = get_resid(a2)
            if resid1 == resid2:
                residues[resid1].AddBond(atom_map[resid][a1.GetIdx()],
                                         atom_map[resid][a2.GetIdx()],
                                         bond.GetBondType())

        for resid, conf in conformers.items():
            residues[resid].AddConformer(conf)
        self.residues = {resid: Residue(mol) for resid, mol in residues.items()}
        self.n_residues = len(self.residues)
        self._residues_indices_map = {i: resid for i, resid in enumerate(
                                      self.residues.keys())}

    def __iter__(self):
        self._residue_index = 0
        return self

    def __next__(self):
        if self._residue_index >= self.n_residues:
            raise StopIteration
        resid = self._residues_indices_map[self._residue_index]
        self._residue_index += 1
        return self.residues[resid]

    def __getitem__(self, selection):
        if isinstance(selection, int):
            resid = self._residues_indices_map[selection]
        else:
            resid = get_residue_components(selection)
            resid = ResidueId(*resid)
        return self.residues[resid]

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"with {self.n_residues} residues"
        return f"<{name}: {params} at {id(self):#x}>"

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()
