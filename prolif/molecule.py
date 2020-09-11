import copy
from collections import defaultdict
from collections.abc import Iterable
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .residue import Residue, ResidueId

class Molecule(Chem.Mol):
    """Molecule class"""
    _cache = {}

    def __init__(self, mol, cache=True):
        super().__init__(mol)

        if cache:
            # build the list of residues in the molecule
            residues = set()
            for atom in self.GetAtoms():
                resid = ResidueId.from_atom(atom)
                residues.add(resid)
            key = hash(tuple(residues))
            # search for a molecule with the same residues in the cache
            try:
                residues, atom_map = self._cache[key]
            except KeyError:
                # if the cache is not empty
                if self._cache:
                    last_key = list(self._cache.keys())[-1]
                    self._cache = {last_key: self._cache[last_key]}
                self._cache[key] = residues, atom_map = self._make_residues()
            residues = copy.deepcopy(residues)
        else:
            self._cache.clear()
            residues, atom_map = self._make_residues()

        # add coordinates to atoms
        conformers = defaultdict(Chem.Conformer)
        xyz = self.GetConformer().GetPositions()
        for resid, mol in residues.items():
            for atom in mol.GetAtoms():
                idx = atom.GetUnsignedProp("__mapindex")
                conformers[resid].SetAtomPosition(
                    atom_map[resid][idx], xyz[idx])
            residues[resid].AddConformer(conformers[resid])

        # assign the list of residues to the molecule
        self.residues = {resid: Residue(mol) for resid, mol in sorted(
                         residues.items(), key=lambda x: (x[0].chain, x[0].number))}
        self.atom_map = atom_map
        self.n_residues = len(self.residues)
        self._residues_indices = list(self.residues.keys())

    def _make_residues(self):
        """Generate a dict of residues"""
        residues = defaultdict(Chem.RWMol)
        atom_map = defaultdict(dict)

        for atom in self.GetAtoms():
            resid = ResidueId.from_atom(atom)
            atom.SetUnsignedProp("__mapindex", atom.GetIdx())
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
        for i in range(self.n_residues):
            resid = self._residues_indices[i]
            yield self.residues[resid]

    def __getitem__(self, selection):
        if isinstance(selection, ResidueId):
            return self.residues[selection]
        elif isinstance(selection, int):
            resid = self._residues_indices[selection]
            return self.residues[resid]
        elif isinstance(selection, str):
            resid = ResidueId.from_string(selection)
            return self.residues[resid]
        elif isinstance(selection, slice):
            resids = self._residues_indices[selection]
            return {resid: self.residues[resid] for resid in resids}
        elif isinstance(selection, Iterable):
            if isinstance(selection[0], int):
                resids = [self._residues_indices[i] for i in selection]
            elif isinstance(selection[0], str):
                resids = [ResidueId.from_string(s) for s in selection]
            return {resid: self.residues[resid] for resid in resids}
        raise ValueError("Expected a ResidueId, int or str, got "
                         f"{type(selection)} instead")

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"with {self.n_residues} residues"
        return f"<{name} {params} at {id(self):#x}>"

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()
