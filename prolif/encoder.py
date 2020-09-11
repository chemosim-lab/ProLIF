import logging
from array import array
from .interactions import _INTERACTIONS


logger = logging.getLogger("prolif")


class Encoder:
    """Class that generates an interaction fingerprint between a protein and a ligand"""

    def __init__(self, interactions=["Hydrophobic", "HBDonor", "HBAcceptor",
                 "PiStacking", "Anionic", "Cationic", "CationPi", "PiCation"]):
        # read interactions to compute
        self.interactions = {}
        if interactions == "all":
            interactions = [i for i in _INTERACTIONS.keys() 
                            if not (i.startswith("_") or i == "Interaction")]
        for interaction in interactions:
            self.add_interaction(interaction)
        self.n_interactions = len(self.interactions)
        logger.info('The fingerprint factory will generate the following bitstring: {}'.format(' '.join(interactions)))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_interactions} interactions: {list(self.interactions.keys())}"
        return f"<{name}: {params} at {id(self):#x}>"

    def add_interaction(self, interaction):
        """Add an interaction to the encoder"""
        if callable(interaction):
            func = interaction
            name = interaction.__name__
        else:
            func = _INTERACTIONS[interaction]().detect
            name = interaction
        self.interactions[interaction] = func
        setattr(self, name.lower(), func)

    def get_bitstring(self, res1, res2):
        """Generate the complete bitstring for the interactions between two residues"""
        bitstring = array("B")
        for interaction_function in self.interactions.values():
            bitstring.append(interaction_function(res1, res2))
        return bitstring

    def get_ifp(self, ligand, protein, residues):
        """Generates the complete IFP between two Molecules for a list of residues"""
        ifp = {}
        for lig in ligand:
            for resid in residues:
                res = protein[resid]
                key = (lig.resid, res.resid)
                ifp[key] = self.get_bitstring(lig, res)
        return ifp
