"""Module for the WaterBridge interaction implementation."""

import itertools as it
from collections import defaultdict
from typing import Iterator, Optional

import networkx as nx

from prolif.ifp import IFP, InteractionData
from prolif.interactions.base import BridgedInteraction


class WaterBridge(BridgedInteraction):
    """Implementation of the WaterBridge analysis.

    Parameters
    ----------
    water : MDAnalysis.core.groups.AtomGroup
        An MDAnalysis AtomGroup containing the water molecules
    order : int
        Maximum number of water molecules that can be involved in a water-bridged
        interaction.
    min_order : int
        Minimum number of water molecules that can be involved in a water-bridged
        interaction.
    hbdonor : Optional[dict]
        Parameters for the HBDonor interaction passed to the underlying fingerprint
        generator. See :class:`~prolif.interactions.interactions.HBDonor` for more
        details.
    hbacceptor : Optional[dict]
        Same as above for :class:`~prolif.interactions.interactions.HBAcceptor`.
    count : bool
        Whether to generate a count fingerprint or just a binary one.

    Notes
    -----
    This analysis currently only runs in serial.
    """

    def __init__(
        self,
        water,
        order: int = 1,
        min_order: int = 1,
        hbdonor: Optional[dict] = None,
        hbacceptor: Optional[dict] = None,
        count: bool = False,
    ):
        # circular import
        from prolif.fingerprint import Fingerprint

        super().__init__()
        if order < 1:
            raise ValueError("order must be greater than 0")
        if min_order > order:
            raise ValueError("min_order cannot be greater than order")
        self.water = water
        self.order = order
        self.min_order = min_order
        self.water_fp = Fingerprint(
            interactions=["HBDonor", "HBAcceptor"],
            parameters={"HBDonor": hbdonor or {}, "HBAcceptor": hbacceptor or {}},
            count=count,
        )

    def setup(self, ifp_store=None, **kwargs) -> None:
        super().setup(ifp_store=ifp_store, **kwargs)
        kwargs.pop("n_jobs", None)
        self.residues = kwargs.pop("residues", None)
        self.kwargs = kwargs

    def run(self, traj, lig, prot) -> dict[int, IFP]:
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
        """  # noqa: E501
        # Run analysis for ligand-water and water-protein interactions
        lig_water_ifp: dict[int, IFP] = self.water_fp._run_serial(
            traj, lig, self.water, residues=None, **self.kwargs
        )
        water_prot_ifp: dict[int, IFP] = self.water_fp._run_serial(
            traj, self.water, prot, residues=self.residues, **self.kwargs
        )
        if self.order >= 2:
            # Run water-water interaction analysis
            water_ifp: dict[int, IFP] = self.water_fp._run_serial(
                traj, self.water, self.water, residues=None, **self.kwargs
            )

        for frame in lig_water_ifp:
            ifp_lw = lig_water_ifp[frame]  # Ligand → Water
            ifp_wp = water_prot_ifp[frame]  # Water → Protein
            self.ifp.setdefault(frame, IFP())

            if self.order >= 2:
                ifp_ww = water_ifp[frame]  # WaterX -> WaterY
                self._any_order(frame, ifp_lw, ifp_ww, ifp_wp)

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

    def _any_order(self, frame: int, ifp_lw: IFP, ifp_ww: IFP, ifp_wp: IFP) -> None:
        """Generate results for any order of water-bridge interactions.

        Constructs a graph to represent the water network and iterates over all paths
        up to a given length (corresponding to ``order + 1``).
        """
        # MultiGraph to allow the same pair of nodes to interact as both HBA and HBD
        # and potentially multiple groups of atoms satisfying the constraints.
        # Each of these interaction will have its own edge in the graph.
        graph = nx.MultiGraph()
        nodes = defaultdict(set)

        # construct graph of water interactions
        for ifp, role in [(ifp_lw, "ligand"), (ifp_wp, "protein")]:
            for data in ifp.interactions():
                graph.add_edge(data.ligand, data.protein, int_data=data)
                # assign ligand and protein residue nodes to corresponding role
                nodes[role].add(getattr(data, role))

        # remove mirror interactions before adding water-water to the graph
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
            paths = nx.all_simple_edge_paths(
                graph, source, targets, cutoff=self.order + 1
            )
            for path in paths:
                if len(path) <= self.min_order:
                    continue
                # path is a list[tuple[node_id1, node_id2, deduplication_key]]
                # first element in path is lig-water1, last is waterN-prot
                data_lw = graph.edges[path[0]]["int_data"]
                data_wp = graph.edges[path[-1]]["int_data"]
                ww_edges = [graph.edges[p] for p in path[1:-1]]
                # only include if strictly passing through water (going back through
                # ligand or protein is not a valid higher-order interaction)
                if all(e.get("water_only") for e in ww_edges):
                    # reorder ligand and protein in InteractionData to be contiguous
                    # i.e. lig-w1, w1-w2, w2-prot instead of lig-w1, w2-w1, w2-prot
                    data_ww_list = []
                    left = data_lw.protein
                    for e in ww_edges:
                        d = e["int_data"]
                        is_sorted = d.ligand == left
                        data_ww = InteractionData(
                            ligand=d.ligand if is_sorted else d.protein,
                            protein=d.protein if is_sorted else d.ligand,
                            # interaction name is not kept in final metadata
                            # so no need to invert it
                            interaction=d.interaction,
                            # the indices of "ligand" water and "protein" water are
                            # merged in the final metadata for water mols so no need to
                            # invert the roles in indices
                            metadata=d.metadata,
                        )
                        data_ww_list.append(data_ww)
                        left = data_ww.protein
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
