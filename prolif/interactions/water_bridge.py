"""Module for the WaterBridge interaction implementation."""

import itertools as it
from collections import defaultdict
from typing import Any, Iterator, Optional

import networkx as nx

from prolif.fingerprint import Fingerprint
from prolif.ifp import IFP, InteractionData


class WaterBridge:
    """Implementation of the WaterBridge analysis for trajectories.

    Parameters
    ----------
    parameters : dict
        Parameters for the HBDonor and HBAcceptor interactions passed to the underlying
        fingerprint generator. See
        :class:`prolif.fingerprint.Fingerprint` for more details.
    count : bool
        Whether to generate a count fingerprint or just a binary one.
    ifp_store : dict
        Container for the results.
    kwargs : Any
        Additional arguments passed at runtime to the fingerprint generator's ``run``
        method.

    Notes
    -----
    This analysis currently only runs in serial.
    """

    def __init__(
        self,
        parameters: Optional[dict] = None,
        count: bool = False,
        ifp_store: Optional[dict[int, IFP]] = None,
        **kwargs: Any,
    ):
        kwargs.pop("n_jobs", None)
        self.residues = kwargs.pop("residues", None)
        self.water_fp = Fingerprint(
            interactions=["HBDonor", "HBAcceptor"],
            parameters=parameters,
            count=count,
        )
        self.kwargs = kwargs
        self.ifp = {} if ifp_store is None else ifp_store

    def run(self, traj, lig, prot, water, order=1) -> dict[int, IFP]:
        """Run the water bridge analysis.

        Parameters
        ----------
        traj : MDAnalysis.coordinates.base.ProtoReader or MDAnalysis.coordinates.base.FrameIteratorSliced
            Iterate over this Universe trajectory or sliced trajectory object
            to extract the frames used for the fingerprint extraction
        lig : MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the ligand
        prot : MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the protein (with multiple residues)
        water: MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the water molecules
        order: int
            Treshold for water bridge order
        """  # noqa: E501
        # Run analysis for ligand-water and water-protein interactions
        lig_water_ifp: dict[int, IFP] = self.water_fp._run_serial(
            traj, lig, water, residues=None, **self.kwargs
        )
        water_prot_ifp: dict[int, IFP] = self.water_fp._run_serial(
            traj, water, prot, residues=self.residues, **self.kwargs
        )
        if order >= 2:
            # Run water-water interaction analysis
            water_ifp: dict[int, IFP] = self.water_fp._run_serial(
                traj, water, water, residues=None, **self.kwargs
            )

        for frame in lig_water_ifp:
            ifp_lw = lig_water_ifp[frame]  # Ligand → Water
            ifp_wp = water_prot_ifp[frame]  # Water → Protein
            self.ifp.setdefault(frame, IFP())

            if order >= 2:
                ifp_ww = water_ifp[frame]  # WaterX -> WaterY
                self._any_order(frame, ifp_lw, ifp_ww, ifp_wp, order=order)

            else:
                self._first_order_only(frame, ifp_lw, ifp_wp)

        return self.ifp

    def _first_order_only(
        self, frame: int, ifp_lw: IFP, ifp_wp: IFP
    ) -> Iterator[tuple[InteractionData, InteractionData]]:
        """Iterates over all relevant combinations of ligand-water-protein"""
        # for each ligand-water interaction
        for data_lw in ifp_lw.interactions():
            # for each water-protein interaction
            for data_wp in ifp_wp.interactions():
                if data_lw.protein == data_wp.ligand:
                    self._merge_metadata(frame, data_lw, data_wp)

    def _any_order(
        self, frame: int, ifp_lw: IFP, ifp_ww: IFP, ifp_wp: IFP, order: int
    ) -> None:
        """Generate results for any order of water-bridge interactions.

        Constructs a graph to represent the water network and iterates over all paths
        up to a given length (corresponding to the ``order``).
        """
        # MultiGraph to allow the same pair of nodes to interact as both HBA and HBD
        # and record it as different paths
        graph = nx.MultiGraph()
        nodes = defaultdict(set)

        # construct graph of water interactions
        for ifp, role in [(ifp_lw, "ligand"), (ifp_wp, "protein")]:
            for data in ifp.interactions():
                graph.add_edge(data.ligand, data.protein, int_data=data)
                # assign ligand and protein residue nodes to corresponding role
                nodes[role].add(getattr(data, role))

        # remove mirror interactions before adding water-water to the graph
        # TODO: sort these so that it matches the order going from ligand to protein
        deduplicated = {
            frozenset((ligand_resid, protein_resid))
            for ligand_resid, protein_resid in ifp_ww
            if ligand_resid != protein_resid
        }
        for ligand_resid, protein_resid in deduplicated:
            ww_dict = ifp_ww.data[ligand_resid, protein_resid]
            for int_name, metadata_tuple in ww_dict.items():
                for metadata in metadata_tuple:
                    data = InteractionData(
                        ligand=ligand_resid,
                        protein=protein_resid,
                        interaction=int_name,
                        metadata=metadata,
                    )
                    graph.add_edge(
                        data.ligand, data.protein, int_data=data, water_only=True
                    )

        # find all edge paths of length up to `order + 1`
        for source in nodes["ligand"]:
            targets = (t for t in nodes["protein"] if nx.has_path(graph, source, t))
            paths = nx.all_simple_edge_paths(graph, source, targets, cutoff=order + 1)
            for path in paths:
                # path is a list[tuple[node_id1, node_id2, interaction]]
                # first element in path is LIG-WAT1, last is WATn-PROT
                data_lw = graph.edges[path[0]]["int_data"]
                data_wp = graph.edges[path[-1]]["int_data"]
                ww_edges = [graph.edges[p] for p in path[1:-1]]
                # only include if strictly passing through water
                if all(e.get("water_only") for e in ww_edges):
                    data_ww_list = [e["int_data"] for e in ww_edges]
                    self._merge_metadata(frame, data_lw, data_wp, *data_ww_list)

    def _merge_metadata(
        self,
        frame: int,
        data_lw: InteractionData,
        data_wp: InteractionData,
        *data_ww_args: InteractionData,
    ) -> None:
        """Merge results from all fingerprints on matching water residues"""
        # get indices for individual water molecules
        water_indices = defaultdict(set)
        for data, role in [
            (data_lw, "protein"),
            (data_wp, "ligand"),
            *it.chain.from_iterable([
                [(data_ww, "ligand"), (data_ww, "protein")] for data_ww in data_ww_args
            ]),
        ]:
            resid = getattr(data, role)
            water_indices[str(resid)].update(data.metadata["indices"][role])
        # construct merged metadata
        metadata = {
            "indices": {
                "ligand": data_lw.metadata["indices"]["ligand"],
                "protein": data_wp.metadata["indices"]["protein"],
                **{key: tuple(indices) for key, indices in water_indices.items()},
            },
            "parent_indices": {
                "ligand": data_lw.metadata["parent_indices"]["ligand"],
                "protein": data_wp.metadata["parent_indices"]["protein"],
                "water": tuple(
                    set().union(
                        data_lw.metadata["parent_indices"]["protein"],
                        data_wp.metadata["parent_indices"]["ligand"],
                        *it.chain.from_iterable([
                            [
                                data_ww.metadata["parent_indices"]["ligand"],
                                data_ww.metadata["parent_indices"]["protein"],
                            ]
                            for data_ww in data_ww_args
                        ]),
                    )
                ),
            },
            "water_residues": tuple(
                dict.fromkeys(  # uniquify but keep order
                    [
                        data_lw.protein,
                        *it.chain.from_iterable([
                            [data_ww.ligand, data_ww.protein]
                            for data_ww in data_ww_args
                        ]),
                        data_wp.ligand,
                    ]
                )
            ),
            "order": len(data_ww_args) + 1,
            "ligand_role": data_lw.interaction,
            "protein_role": (  # invert role
                "HBDonor" if data_wp.interaction == "HBAcceptor" else "HBAcceptor"
            ),
            **{
                f"{key}{suffix}": data.metadata[key]
                for suffix, data in [
                    (f"_ligand_{data_lw.protein}", data_lw),
                    *it.chain.from_iterable([
                        [(f"_{data_ww.ligand}_{data_ww.protein}", data_ww)]
                        for data_ww in data_ww_args
                    ]),
                    (f"_{data_wp.ligand}_protein", data_wp),
                ]
                for key in ["distance", "DHA_angle"]
            },
        }

        # store metadata
        self.ifp[frame].setdefault((data_lw.ligand, data_wp.protein), {}).setdefault(
            "WaterBridge", []
        ).append(metadata)
