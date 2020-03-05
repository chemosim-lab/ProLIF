import logging, re
import pandas as pd
from .utils import get_resnumber
from .fingerprint import FingerprintFactory

logger = logging.getLogger("prolif")


class Dataframe(pd.DataFrame):
    """Class to store IFPs and compute similarity between fingerprints"""

    # properties for subclassing pandas
    _metadata = ['ff','interactions','n_interactions','id_columns']

    # subclassing pandas
    @property
    def _constructor(self):
        return Dataframe

    def configure(self, ff: FingerprintFactory):
        """Configure the dataframe with the FingerprintFactory"""
        # new properties
        self.ff = ff
        self.interactions = list(ff.interactions.keys())
        self.n_interactions = ff.n_interactions
        self.id_columns = ["Ligand name", "Ligand frame", "Ligand residue", "Protein name", "Protein frame"]
        return self

    def generate_ifp(self, ligand_frame, protein_frame):
        """Compute and append an IFP to the dataframe"""
        ifp = self.ff.generate_ifp(ligand_frame, protein_frame)
        self.append_ifp(ifp)

    def append_ifp(self, ifp):
        """Append an IFP to the dataframe"""
        df = pd.concat([self, pd.DataFrame(ifp)], sort=True)
        # update
        self.__dict__.update(df.__dict__)

    def curate(self):
        """Convert the IFP dataframe to a human-readable version.
        Fills NaN columns with the appropriate 0-filled list, sorts the columns,
        and adds a multi-index for readability"""
        residues = self.drop(columns=self.id_columns).columns.tolist()
        # replace NaN by list of 0
        self[residues] = self[residues].applymap(lambda x: x if isinstance(x, list) else [0]*self.n_interactions)
        # sort columns
        residues.sort(key=get_resnumber)
        # explode columns lists
        ifps = Dataframe()
        ids = self[self.id_columns].copy()
        for res in residues:
            columns = ["%s_%s" % (res,i) for i in self.interactions]
            ifps[columns] = self[res].apply(lambda x: pd.Series(x))
        ifps.columns = pd.MultiIndex.from_product([residues, self.interactions], names=["residue","interaction"])
        ids.columns = pd.MultiIndex.from_product([self.id_columns,[""]])
        df = pd.concat([ids, ifps], axis=1)
        # update
        self.__dict__.update(df.__dict__)

    def get_residue(self, residue):
        """Returns all interactions bits corresponding to a particular residue"""
        return pd.concat([
            self[self.id_columns].droplevel(1, axis=1), self[residue]
        ], axis=1)

    def get_interaction(self, interaction):
        """Returns all residues interaction bits corresponding to a particular interaction"""
        return pd.concat([
            self[self.id_columns].droplevel(1, axis=1), self.xs(interaction, axis=1, level=1)
        ], axis=1)
