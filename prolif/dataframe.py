import logging
import copy
from functools import wraps
import pandas as pd
import numpy as np
from rdkit import DataStructs
from .utils import get_resnumber, requires_sklearn, requires_seaborn
from .fingerprint import FingerprintFactory
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
except ImportError:
    pass
try:
    import seaborn as sns
except ImportError:
    pass

logger = logging.getLogger("prolif")



class Dataframe(pd.DataFrame):
    """Class to store IFPs and compute similarity between fingerprints"""

    # properties for subclassing pandas
    _metadata = ['ff','interactions','n_interactions','id_columns','_configured']
    # dataframe was configured with a FingerprintFactory ?
    _configured = False

    # subclassing pandas
    @property
    def _constructor(self):
        return Dataframe

    def configure(self, ff: FingerprintFactory):
        """Configure the dataframe with the FingerprintFactory"""
        # new properties
        self._configured = True
        self.ff = ff
        self.interactions = list(ff.interactions.keys())
        self.n_interactions = ff.n_interactions
        self.id_columns = ["Ligand name", "Ligand frame", "Ligand residue", "Protein name", "Protein frame"]

    def requires_config(func):
        """Check if the dataframe has been configured with a FingerprintFactory"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            assert self._configured, "The Dataframe needs to be configured with df.configure()"
            return func(self, *args, **kwargs)
        return wrapper

    @requires_config
    def generate_ifp(self, ligand_frame, protein_frame):
        """Compute and append an IFP to the dataframe"""
        ifp = self.ff.generate_ifp(ligand_frame, protein_frame)
        self.append_ifp(ifp)

    def append_ifp(self, ifp):
        """Append an IFP to the dataframe"""
        df = pd.concat([self, pd.DataFrame(ifp)], sort=True)
        # update
        self.__dict__.update(df.__dict__)

    @requires_config
    def postprocess(self) -> 'Dataframe':
        """Convert the IFP dataframe to a human-readable version.
        Fills NaN columns with 0s, sorts the columns,
        and adds a multi-index for readability"""
        residues = self.drop(columns=self.id_columns).columns.tolist()
        self.reset_index(inplace=True, drop=True)
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

    @requires_config
    def get_residue(self, residue) -> 'Dataframe':
        """Returns all interactions bits corresponding to a particular residue"""
        return pd.concat([
            self[self.id_columns].droplevel(1, axis=1), self[residue]
        ], axis=1)

    @requires_config
    def get_interaction(self, interaction) -> 'Dataframe':
        """Returns all residues interaction bits corresponding to a particular interaction"""
        return pd.concat([
            self[self.id_columns].droplevel(1, axis=1), self.xs(interaction, axis=1, level=1)
        ], axis=1)

    @requires_config
    def get_pocket_residues(self) -> list:
        """Returns a list of residues that interacted with the ligand during the simulation"""
        return self.droplevel(1, axis=1).drop(columns=self.id_columns).columns.unique().tolist()

    def as_rdkit_bitvector(self, index):
        """Returns the interactions of a frame as an rdkit `DataStructs.ExplicitBitVect`"""
        residues = self.get_pocket_residues()
        v = DataStructs.ExplicitBitVect(self.ff.n_interactions*len(residues))
        for i in np.flatnonzero(self[residues].iloc[index].values).tolist():
            v.SetBit(i)
        return v

    def get_tanimoto_score(self, index, reference_index=0, as_distance=False) -> float:
        """Returns Tanimoto similarity between two indices in the Dataframe"""
        return DataStructs.TanimotoSimilarity(
            self.as_rdkit_bitvector(reference_index),
            self.as_rdkit_bitvector(index),
        )

    def get_tanimoto(self, reference_index=0, as_distance=False) -> 'Dataframe':
        """"Returns all Tanimoto similarity values based on a reference index"""
        values = DataStructs.BulkTanimotoSimilarity(
            self.as_rdkit_bitvector(reference_index),
            [ self.as_rdkit_bitvector(i) for i in self.index ],
        returnDistance=as_distance)
        return pd.concat([
            self[self.id_columns].droplevel(1,axis=1), pd.Series(values, name="Tanimoto", index=self.index)
        ], axis=1)

    def get_tanimoto_matrix(self, as_distance=False) -> list:
        """Returns the upper diagonal matrix of Tanimoto similarity as a list of lists"""
        distances, indices = [], self.index.tolist()
        # build rdkit bitvectors
        bitvector = {i: self.as_rdkit_bitvector(i) for i in indices}
        # compute bulk tanimoto
        for i in range(1,len(indices)):
            values = DataStructs.BulkTanimotoSimilarity(
                bitvector[i], [ bitvector[j] for j in indices[:i] ],
                returnDistance=as_distance)
            distances.append(values)
        return distances

    @requires_sklearn
    def kmeans_clustering(self, n_clusters):
        """Performs Kmeans clustering on the dataframe, using sklearn. Returns
        the Dataframe with a `Cluster` column added"""
        pocket_residues = self.get_pocket_residues()
        kmeans = KMeans(n_clusters=n_clusters)
        X = self[pocket_residues].values
        kmeans.fit(X)
        cs = kmeans.predict(X)
        # show closest to cluster centroid
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        print("Frames closest to centroids:")
        print(", ".join(["cluster %d: %d"%(i,f) for i,f in enumerate(closest)]))
        # add cluster column
        data = self[self.id_columns].droplevel(1, axis=1).copy()
        for i, cluster in enumerate(cs):
            data.loc[i, "Cluster"] = cluster
        data["Cluster"] = data["Cluster"].astype(int)
        # merge
        data.columns = pd.MultiIndex.from_product([data.columns,[""]])
        data = pd.merge(data, self, on=self.id_columns, how="inner")
        return data

    @requires_seaborn
    def plot_ifp(self, drop_residues:bool=True, drop_threshold:float=0.01, **kwargs):
        """Plots the IFP for a MD simulation, frame by frame, using seaborn.

        Parameters
        ----------
        drop_residues: bool, True
            Drop residues that interact in less than X% of frames with the ligand
        drop_threshold: float, 0.01
            Threshold value to discard residues. Needs `drop_residues=True`
        font_scale: float, 1.5
            Font scale on the plot
        ylim: tuple, (0, nframes)
            Limits of frames displayed on the Y-axis

        All remaining parameters are passed to `sns.catplot`
        """
        data = pd.melt(self, id_vars=self.id_columns, var_name=["residue","interaction"])
        data = data[data["value"] != 0]
        data.reset_index(inplace=True, drop=True)
        data.drop(columns=["Ligand name","Ligand residue","Protein name","Protein frame"], inplace=True)
        data.rename(columns={"Ligand frame":"Frame"}, inplace=True)
        if drop_residues:
            # remove residues appearing less than 1% of the time
            t = data.groupby(["residue","interaction"], as_index=False)\
                .agg({"value":"count"}).groupby("residue", as_index=False)\
                .agg({"value":"max"})
            nframes = data["Frame"].max() + 1
            threshold = int(round(drop_threshold*nframes))
            todrop = t.loc[t["value"] < threshold].residue.tolist()
            data = data[~data["residue"].isin(todrop)]
            print("Removed", ", ".join(todrop), f"""- interacting with the ligand in less than {100*drop_threshold}% of frames ({threshold}/{nframes} frames)""")
        # plot
        font_scale = kwargs.pop("font_scale", 1.5)
        ylim = kwargs.pop("ylim", (0, data.Frame.max()+1))
        user_kwargs = copy.deepcopy(kwargs)
        kwargs = dict(
            height=10, aspect=0.08, jitter=0, sharex=False,
            marker="_", s=7, linewidth=2)
        kwargs.update(user_kwargs)
        sns.set_context(font_scale=font_scale)
        g = sns.catplot(
            data=data, x="interaction", y="Frame",
            hue="interaction", col="residue", **kwargs)
        g.set_titles("{col_name}", rotation=90)
        g.set(xticks=[], ylim=ylim)
        g.set_xticklabels([])
        g.set_xlabels("")
        g.add_legend()
        for ax in g.axes.flat:
            ax.invert_yaxis()
        return g
