import copy
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from .residue import Residue, ResidueId

class Molecule(Chem.Mol):
    """Molecule class"""
    _cache = {}

    def __init__(self, mol):
        super().__init__(mol)

        residues = set()
        for atom in self.GetAtoms():
            resid = ResidueId.from_atom(atom)
            residues.add(resid)
        key = hash(tuple(residues))
        try:
            residues, atom_map = self._cache[key]
        except KeyError:
            if self._cache:
                last_key = list(self._cache.keys())[-1]
                self._cache = {last_key: self._cache[last_key]}
            self._cache[key] = residues, atom_map = self._make_residues()
        residues = copy.deepcopy(residues)

        conformers = {}
        xyz = self.GetConformer().GetPositions()
        for resid, mol in residues.items():
            for atom in mol.GetAtoms():
                try:
                    conformers[resid]
                except KeyError:
                    conformers[resid] = Chem.Conformer()
                finally:
                    idx = atom.GetIntProp("__mapindex")
                    atom.ClearProp("__mapindex")
                    conformers[resid].SetAtomPosition(
                        atom_map[resid][idx], xyz[idx])
            residues[resid].AddConformer(conformers[resid])

        self.residues = {resid: Residue(mol) for resid, mol in sorted(
                         residues.items(), key=lambda x: (x[0].chain, x[0].number))}
        self.n_residues = len(self.residues)
        self._residues_indices_map = {i: resid for i, resid in enumerate(
                                      self.residues.keys())}
    
    def _make_residues(self):
        residues = {}
        atom_map = {}

        for atom in self.GetAtoms():
            resid = ResidueId.from_atom(atom)
            try:
                residues[resid]
            except KeyError:
                residues[resid] = Chem.RWMol()
                atom_map[resid] = {}
            finally:
                atom.SetIntProp("__mapindex", atom.GetIdx())
                idx = residues[resid].AddAtom(atom)
                atom_map[resid][atom.GetIdx()] = idx

        for bond in self.GetBonds():
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            resid1 = ResidueId.from_atom(a1)
            resid2 = ResidueId.from_atom(a2)
            if resid1 == resid2:
                residues[resid1].AddBond(atom_map[resid1][a1.GetIdx()],
                                         atom_map[resid1][a2.GetIdx()],
                                         bond.GetBondType())
        return residues, atom_map

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
        if isinstance(selection, ResidueId):
            resid = selection
        elif isinstance(selection, int):
            if selection < 0:
                selection = self.n_residues + selection
            resid = self._residues_indices_map[selection]
        else:
            resid = ResidueId.from_string(selection)
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
