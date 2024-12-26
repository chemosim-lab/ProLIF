"""Module for the WaterBridge interaction implementation."""

from collections import defaultdict
from typing import Any, Callable, Iterator

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
    """

    def __init__(
        self,
        parameters: dict | None = None,
        count: bool = False,
        ifp_store: dict[int, IFP] | None = None,
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

        # Run water-water interaction analysis if order is 2 or above
        if order >= 2:
            water_ifp: dict[int, IFP] | None = self.water_fp._run_serial(
                traj, water, water, residues=None, **self.kwargs
            )
        else:
            water_ifp = None

        # first order
        self._merge_metadata(lig_water_ifp, None, water_prot_ifp)

        # add metadata for higher order
        if water_ifp:
            self._merge_metadata(lig_water_ifp, water_ifp, water_prot_ifp)

        return self.ifp

    def _merge_metadata(
        self,
        lig_water_ifp: dict[int, IFP],
        water_ifp: dict[int, IFP] | None,
        water_prot_ifp: dict[int, IFP],
    ) -> None:
        """Merge results from all fingerprints on matching water residues"""
        for frame in lig_water_ifp:
            ifp_lw = lig_water_ifp[frame]  # Ligand → Water
            ifp_wp = water_prot_ifp[frame]  # Water → Protein
            ifp_ww = water_ifp[frame] if water_ifp else None  # Water → Water

            # for each ligand-water1 interaction
            for data_lw in ifp_lw.interactions():
                # for each water1-water2 interaction (optional)
                for data_ww in self._filter_interactions(
                    ifp_ww,
                    # match with ligand-water1 and exclude interaction with itself
                    lambda x: x.ligand == data_lw.protein and x.ligand != x.protein,
                    # if water_ifp is None, simply yields data_lw
                    default=data_lw,
                ):
                    # for each water2-protein interaction where water1 == water2
                    for data_wp in self._filter_interactions(
                        ifp_wp, lambda x: x.ligand == data_ww.protein
                    ):
                        # get indices for individual water molecules
                        water_indices = defaultdict(set)
                        for data, role in [
                            (data_lw, "protein"),
                            (data_wp, "ligand"),
                            *(
                                ((data_ww, "ligand"), (data_ww, "protein"))
                                if ifp_ww
                                else ()
                            ),
                        ]:
                            resid = getattr(data, role)
                            water_indices[f"water_{resid}"].update(
                                data.metadata["indices"][role]
                            )
                        # construct merged metadata
                        metadata = {
                            "indices": {
                                "ligand": data_lw.metadata["indices"]["ligand"],
                                "protein": data_wp.metadata["indices"]["protein"],
                                **{
                                    key: tuple(indices)
                                    for key, indices in water_indices.items()
                                },
                            },
                            "parent_indices": {
                                "ligand": data_lw.metadata["parent_indices"]["ligand"],
                                "protein": data_wp.metadata["parent_indices"][
                                    "protein"
                                ],
                                "water": tuple(
                                    set().union(
                                        data_lw.metadata["parent_indices"]["protein"],
                                        data_wp.metadata["parent_indices"]["ligand"],
                                        *(
                                            (
                                                data_ww.metadata["parent_indices"][
                                                    "ligand"
                                                ],
                                                data_ww.metadata["parent_indices"][
                                                    "protein"
                                                ],
                                            )
                                            if ifp_ww
                                            else ()
                                        ),
                                    )
                                ),
                            },
                            "water_residues": tuple(
                                {
                                    data_lw.protein,
                                    data_ww.ligand,
                                    data_ww.protein,
                                    data_wp.ligand,
                                }
                                if ifp_ww
                                else (data_lw.protein,)
                            ),
                            "order": 2 if ifp_ww else 1,
                            "ligand_role": data_lw.interaction,
                            "protein_role": (  # invert role
                                "HBDonor"
                                if data_wp.interaction == "HBAcceptor"
                                else "HBAcceptor"
                            ),
                            **{
                                f"{key}{suffix}": data.metadata[key]
                                for suffix, data in ([
                                    ("_ligand_water", data_lw),
                                    ("_water_protein", data_wp),
                                    *(("_water_water", data_ww) if ifp_ww else ()),
                                ])
                                for key in ["distance", "DHA_angle"]
                            },
                        }

                        # store metadata
                        ifp = self.ifp.setdefault(frame, IFP())
                        ifp.setdefault(
                            (data_lw.ligand, data_wp.protein), {}
                        ).setdefault("WaterBridge", []).append(metadata)

    @staticmethod
    def _filter_interactions(
        ifp: IFP | None,
        predicate: Callable[[InteractionData], bool],
        default: InteractionData | None = None,
    ) -> Iterator[InteractionData]:
        """Filters interactions to those that satisfy the predicate. If ``ifp==None``,
        simply yields the ``default`` value.
        """
        if ifp is None:
            yield default
        else:
            for data in ifp.interactions():
                if predicate(data):
                    yield data
