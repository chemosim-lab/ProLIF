import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .residue import Residue, ResidueId, ResidueGroup

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
        self.residues = ResidueGroup(sorted([(resid, Residue(mol))
                                     for resid, mol in residues.items()],
                                     key=lambda x: (x[0].chain, x[0].number)))
        self.atom_map = atom_map

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
        for residue in self.residues.values():
            yield residue

    def __getitem__(self, key):
        return self.residues[key]

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_residues} residues and {self.GetNumAtoms()} atoms"
        return f"<{name} with {params} at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self.residues)

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()
