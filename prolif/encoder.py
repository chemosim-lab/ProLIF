import logging
from functools import wraps
import numpy as np
from .interactions import _INTERACTIONS


logger = logging.getLogger("prolif")

def _only_return_bits(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        results = f(*args, **kwargs)
        return results if isinstance(results, (bool, type(None))) else results[0]
    return wrapper


class Encoder:
    """Class that generates an interaction fingerprint between a protein and a ligand"""

    def __init__(self, interactions=["Hydrophobic", "HBDonor", "HBAcceptor",
                 "PiStacking", "Anionic", "Cationic", "CationPi", "PiCation"],
                 return_atoms=False):
        self.return_atoms = return_atoms
        # read interactions to compute
        self.interactions = {}
        if interactions == "all":
            interactions = [i for i in _INTERACTIONS.keys() 
                            if not (i.startswith("_") or i == "Interaction")]
        for interaction in interactions:
            self.add_interaction(interaction)
        logger.info('The fingerprint factory will generate the following bitstring: {}'.format(' '.join(interactions)))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_interactions} interactions: {list(self.interactions.keys())}"
        return f"<{name}: {params} at {id(self):#x}>"

    def add_interaction(self, interaction):
        """Add an interaction to the encoder"""
        func = _INTERACTIONS[interaction]().detect
        if not self.return_atoms:
            func = _only_return_bits(func)
        self.interactions[interaction] = func
        setattr(self, interaction.lower(), func)
    
    @property
    def n_interactions(self):
        return len(self.interactions)

    def run(self, res1, res2):
        """Generate the complete bitstring for the interactions between two
        residues

        Parameters
        ----------
        res1 : prolif.residue.Residue
            A residue (usually from a ligand in examples)
        res2 : prolif.residue.Residue
            A residue (usually from a protein in examples)

        Returns
        -------
        bitstring : np.ndarray
            An array storing the encoded interactions between res1 and res2
        atoms : list, optionnal
            A list containing tuples of (res1_atom_index, res2_atom_index) for
            each interaction. Available if the encoder was created with
            ``return_atoms=True``
        """
        bitstring = []
        if self.return_atoms:
            atoms_lst = []
            for interaction_function in self.interactions.values():
                bit, *atoms = interaction_function(res1, res2)
                bitstring.append(bit)
                atoms_lst.append(atoms)
            bitstring = np.array(bitstring, dtype=bool)
            return bitstring, atoms_lst
        for interaction_function in self.interactions.values():
            bit = interaction_function(res1, res2)
            bitstring.append(bit)
        return np.array(bitstring, dtype=bool)
