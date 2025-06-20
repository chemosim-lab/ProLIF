"""
Plot interactions in 3D --- :mod:`prolif.plotting.complex3d`
============================================================

.. versionadded:: 2.0.0

.. autoclass:: Complex3D
   :members:

"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from rdkit.Geometry import Point3D

from prolif.exceptions import RunRequiredError
from prolif.plotting.complex3d.py3dmol_backend import Py3DmolBackend, Py3DMolSettings
from prolif.plotting.complex3d.pymol_backend import PyMOLBackend, PyMOLSettings
from prolif.plotting.utils import metadata_iterator
from prolif.utils import get_residues_near_ligand

if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP
    from prolif.molecule import Molecule
    from prolif.plotting.complex3d.base import Backend, Settings


class Complex3D:
    """Creates a py3Dmol plot of interactions.

    Parameters
    ----------
    ifp : IFP
        The interaction dictionary for a single frame.
    lig_mol : Molecule
        The ligand molecule to display.
    prot_mol : Molecule
        The protein molecule to display.
    water_mol : Optional[Molecule]
        Additional molecule (e.g. waters) to display.
    backend_settings: Literal["py3Dmol", "pymol"] | Py3DMolSettings | PyMOLSettings = "py3Dmol"
        The backend or backend settings to use. If a string is provided, the
        relevant backend is used with default settings. If a Settings object is
        provided, this will be used instead.
    backend_kwargs: Any
        Additional parameters passed to the backend's setup.

    .. versionchanged:: 2.1.0
        Added ``water_mol`` parameter to the constructor to display waters
        involved in WaterBridge interactions. Added ``save_png`` method to save the
        current state of the 3D viewer to a PNG. Added ``remove_hydrogens`` parameter
        to the ``display`` and ``compare`` methods to remove non-polar hydrogens that
        aren't involved in an interaction. Added ``only_interacting`` parameter to the
        ``display`` and ``compare`` methods to show all protein residues in the
        vicinity of the ligand, or only the ones participating in an interaction. Moved
        most options to the `backend_settings` object (accessible through
        :attr:`backend`'s :attr:`settings`).
    """  # noqa: E501

    def __init__(
        self,
        ifp: IFP,
        lig_mol: Molecule,
        prot_mol: Molecule,
        water_mol: Molecule | None = None,
        backend_settings: Literal["py3Dmol", "pymol"]
        | Py3DMolSettings
        | PyMOLSettings = "py3Dmol",
        **backend_kwargs: Any,
    ) -> None:
        object.__setattr__(self, "backend", get_3d_plot_backend(backend_settings))
        self.backend: "Backend"
        self.backend_kwargs = backend_kwargs
        self.ifp = ifp
        self.lig_mol = lig_mol
        self.prot_mol = prot_mol
        self.water_mol = water_mol
        self.interface: Any = None

    @classmethod
    def from_fingerprint(
        cls,
        fp: Fingerprint,
        lig_mol: Molecule,
        prot_mol: Molecule,
        water_mol: Molecule | None = None,
        *,
        frame: int,
        backend_settings: Literal["py3Dmol", "pymol"]
        | Py3DMolSettings
        | PyMOLSettings = "py3Dmol",
        **backend_kwargs: Any,
    ) -> Complex3D:
        """Creates a py3Dmol plot of interactions.

        Parameters
        ----------
        fp : prolif.fingerprint.Fingerprint
            The fingerprint object already executed using one of the ``run`` or
            ``run_from_iterable`` methods.
        frame : int
            The frame number chosen to select which interactions are going to be
            displayed.
        lig_mol : Molecule
            The ligand molecule to display.
        prot_mol : Molecule
            The protein molecule to display.
        water_mol : Optional[Molecule]
            Additional molecule (e.g. waters) to display.
        backend_settings: Literal["py3Dmol", "pymol"] | Py3DMolSettings | PyMOLSettings = "py3Dmol"
            The backend or backend settings to use. If a string is provided, the
            relevant backend is used with default settings. If a Settings object is
            provided, this will be used instead.
        backend_kwargs: Any
            Additional parameters passed to the backend's setup.
        """  # noqa: E501
        if not hasattr(fp, "ifp"):
            raise RunRequiredError(
                "Please run the fingerprint analysis before attempting to display"
                " results.",
            )
        ifp = fp.ifp[frame]
        return cls(
            ifp,
            lig_mol,
            prot_mol,
            water_mol,
            backend_settings=backend_settings,
            **backend_kwargs,
        )

    @staticmethod
    def get_ring_centroid(mol: Molecule, indices: tuple[int, ...]) -> Point3D:
        """Get the centroid of a ring system."""
        centroid = mol.xyz[list(indices)].mean(axis=0)
        return Point3D(*centroid)

    def display(
        self,
        size: tuple[int, int] = (650, 600),
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True,
    ) -> Complex3D:
        """Display the complex in 3D.

        Parameters
        ----------
        size: tuple[int, int] = (650, 600)
            Size of the interface.
        display_all : bool = False
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Not relevant if ``count=False`` in the ``Fingerprint``
            object.
        only_interacting : bool = True
            Whether to show all protein residues in the vicinity of the ligand, or
            only the ones participating in an interaction.
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True
            Whether to remove non-polar hydrogens (unless they are involved in an
            interaction).

        .. versionchanged:: 2.1.0
            Added ``only_interacting=True`` and ``remove_hydrogens=True`` parameters.
            Non-polar hydrogen atoms that aren't involved in interactions are now
            hidden. Added support for waters involved in WaterBridge interactions.
            Added support for multiple backends.

        """
        # setup view parameters
        if isinstance(self.backend, Py3DmolBackend):
            kwargs = {"viewergrid": (1, 1), "width": size[0], "height": size[1]}
        elif isinstance(self.backend, PyMOLBackend):
            kwargs = {"width": size[0], "height": size[1]}
        else:
            kwargs = {}
        self.backend.setup(**{**self.backend_kwargs, **kwargs})
        self.backend.prepare()

        # plot
        self._populate_view(
            display_all=display_all,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )
        self.interface = self.backend.interface
        return self

    def compare(
        self,
        other: Complex3D,
        *,
        size: tuple[int, int] = (900, 600),
        display_all: bool = False,
        linked: bool = True,
        color_unique: str | None = "magentaCarbon",
        only_interacting: bool = True,
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True,
    ) -> Complex3D:
        """Displays the initial complex side-by-side with a second one for easier
        comparison.

        Parameters
        ----------
        other: Complex3D
            Other ``Complex3D`` object to compare to.
        size: tuple[int, int] = (900, 600)
            The size of the interface.
        display_all : bool = False
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Not relevant if ``count=False`` in the ``Fingerprint``
            object.
        linked: bool = True
            Link mouse interactions (pan, zoom, translate) on both views.
        color_unique: str | None = "magentaCarbon",
            Which color to use for residues that have interactions that are found in one
            complex but not the other. Use ``None`` to disable the color override.
        only_interacting : bool = True
            Whether to show all protein residues in the vicinity of the ligand, or
            only the ones participating in an interaction.
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True
            Whether to remove non-polar hydrogens (unless they are involved in an
            interaction).

        .. versionadded:: 2.0.1

        .. versionchanged:: 2.1.0
            Added ``only_interacting=True`` and ``remove_hydrogens=True`` parameters.
            Non-polar hydrogen atoms that aren't involved in interactions are now
            hidden. Added support for waters involved in WaterBridge interactions.

        """
        # get set of interactions for both poses
        interactions1 = {
            (resid[1], i)
            for resid, interactions in self.ifp.items()
            for i in interactions
        }
        interactions2 = {
            (resid[1], i)
            for resid, interactions in other.ifp.items()
            for i in interactions
        }

        # get residues with interactions specific to pose 1
        highlights = (
            {r[0]: color_unique for r in interactions1 - interactions2}
            if color_unique
            else {}
        )

        # configure view parameters
        if isinstance(self.backend, Py3DmolBackend):
            kwargs = {
                "viewergrid": (1, 2),
                "width": size[0],
                "height": size[1],
                "linked": linked,
            }
        else:
            kwargs = {}

        self.backend.setup(**{**self.backend_kwargs, **kwargs})

        # prepare first plot
        if isinstance(self.backend, Py3DmolBackend):
            kwargs = {
                "position": (0, 0),
                "colormap": highlights,
            }
        else:
            kwargs = {}
        self.backend.prepare(**kwargs)

        # first plot
        self._populate_view(
            display_all=display_all,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )

        # get residues with interactions specific to pose 2
        highlights = (
            {r[0]: color_unique for r in interactions2 - interactions1}
            if color_unique
            else {}
        )

        # prepare second plot
        if isinstance(self.backend, Py3DmolBackend):
            kwargs = {
                "position": (0, 1),
                "colormap": highlights,
            }
        else:
            kwargs = {}
        other.backend = self.backend  # copy current plot state to other
        self.backend.prepare(**kwargs)

        # second plot
        other._populate_view(
            display_all=display_all,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )
        self.interface = self.backend.interface
        return self

    def add(
        self,
        other: Complex3D,
        *,
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True,
    ) -> Complex3D:
        """Add another ``Complex3D`` object to the current plot.

        Parameters
        ----------
        other: Complex3D
            Other ``Complex3D`` object to add to the plot.
        display_all : bool = False
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Not relevant if ``count=False`` in the ``Fingerprint``
            object.
        only_interacting : bool = True
            Whether to show all protein residues in the vicinity of the ligand, or
            only the ones participating in an interaction.
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True
            Whether to remove non-polar hydrogens (unless they are involved in an
            interaction).

        .. versionadded:: 2.1.0

        """
        saved_settings = self.backend.settings
        # copy current plot state to other
        other_settings = other.backend.settings
        backend = other.backend = self.backend

        if not self.interface:
            self.display(
                display_all=display_all,
                only_interacting=only_interacting,
                remove_hydrogens=remove_hydrogens,
            )

        # keep user-defined settings for other temporarily
        backend.settings = other_settings

        # prepare and plot
        model_id_start = backend._model_count
        backend.prepare()
        backend._model_count = model_id_start
        other._populate_view(
            display_all=display_all,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )

        self.interface = other.interface = backend.interface
        # restore
        backend.settings = saved_settings
        other.backend = get_3d_plot_backend(other_settings)
        return self

    def _populate_view(  # noqa: PLR0912
        self,
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True,
    ) -> None:
        backend = self.backend
        settings = cast("Settings", backend.settings)

        # load molecules
        backend.load_molecule(
            self.lig_mol,
            "ligand",
            settings.peptide_style
            if self.lig_mol.n_residues >= settings.peptide_threshold
            else settings.ligand_style,
        )
        backend.load_molecule(self.prot_mol, "protein", settings.protein_style)
        if self.water_mol:
            backend.load_molecule(self.water_mol, "water", settings.residues_style)

        self._interacting_atoms: defaultdict[str, set[int]] = defaultdict(set)
        # show all interacting residues
        for (lresid, presid), interactions in self.ifp.items():
            lres = self.lig_mol[lresid]
            pres = self.prot_mol[presid]
            # set model ids for reusing later
            for res, component, style in [
                (lres, "ligand", settings.ligand_style),
                (pres, "protein", settings.residues_style),
            ]:
                if res.resid not in backend.residues:
                    backend.show_residue(res, component, style)
            for interaction, metadata_tuple in interactions.items():
                # whether to display all interactions or only the one with the shortest
                # distance
                for metadata in metadata_iterator(metadata_tuple, display_all):
                    # record indices of atoms interacting
                    self._interacting_atoms["ligand"].update(
                        metadata["parent_indices"]["ligand"]
                    )
                    self._interacting_atoms["protein"].update(
                        metadata["parent_indices"]["protein"]
                    )
                    if interaction in settings.bridged_interactions and self.water_mol:
                        for wresid in metadata["water_residues"]:
                            self._interacting_atoms["water"].update(
                                metadata["parent_indices"][
                                    settings.bridged_interactions[interaction]
                                ]
                            )
                            wres = self.water_mol[wresid]
                            if wresid not in backend.residues:
                                backend.show_residue(
                                    wres, "water", settings.residues_style
                                )
                        # show cylinders for WaterBridge
                        distances = [d for d in metadata if d.startswith("distance_")]
                        for distlabel in distances:
                            _, src, dest = distlabel.split("_")
                            if src == "ligand":
                                atoms1 = metadata["parent_indices"]["ligand"][
                                    settings.ligand_displayed_atoms.get(interaction, 0)
                                ]
                                p1 = self.lig_mol.GetConformer().GetAtomPosition(atoms1)
                            else:
                                atoms1 = metadata["parent_indices"][src][0]
                                p1 = self.water_mol.GetConformer().GetAtomPosition(
                                    atoms1
                                )
                            if dest == "protein":
                                atoms2 = metadata["parent_indices"]["protein"][
                                    settings.protein_displayed_atoms.get(interaction, 0)
                                ]
                                p2 = self.prot_mol.GetConformer().GetAtomPosition(
                                    atoms2
                                )
                            else:
                                atoms2 = metadata["parent_indices"][dest][0]
                                p2 = self.water_mol.GetConformer().GetAtomPosition(
                                    atoms2
                                )

                            backend.add_interaction(
                                interaction,
                                distance=metadata[distlabel],
                                points=(p1, p2),
                                residues=(lresid, presid),
                                atoms=(atoms1, atoms2),
                            )

                    else:
                        # get coordinates for both points of the interaction
                        if interaction in settings.ligand_ring_interactions:
                            atoms1 = metadata["parent_indices"]["ligand"]
                            p1 = self.get_ring_centroid(self.lig_mol, atoms1)
                        else:
                            atoms1 = metadata["parent_indices"]["ligand"][
                                settings.ligand_displayed_atoms.get(interaction, 0)
                            ]
                            p1 = self.lig_mol.GetConformer().GetAtomPosition(atoms1)
                        if interaction in settings.protein_ring_interactions:
                            atoms2 = metadata["parent_indices"]["protein"]
                            p2 = self.get_ring_centroid(self.prot_mol, atoms2)
                        else:
                            atoms2 = metadata["parent_indices"]["protein"][
                                settings.protein_displayed_atoms.get(interaction, 0)
                            ]
                            p2 = self.prot_mol.GetConformer().GetAtomPosition(atoms2)
                        # add interaction line
                        dist = metadata.get("distance", float("nan"))
                        backend.add_interaction(
                            interaction,
                            distance=dist,
                            points=(p1, p2),
                            residues=(lresid, presid),
                            atoms=(atoms1, atoms2),
                        )

        # show "protein" residues that are close to the "ligand"
        if not only_interacting:
            self._show_unpaired_residues(self.prot_mol, "protein")
            if self.water_mol:
                self._show_unpaired_residues(self.water_mol, "water")

        # hide non-polar hydrogens (except if they are involved in an interaction)
        if remove_hydrogens:
            self._hide_hydrogens(remove_hydrogens)

        backend.finalize()

    def _show_unpaired_residues(self, mol: Molecule, component: str) -> None:
        pocket_residues = get_residues_near_ligand(self.lig_mol, mol)
        unpaired_residues = set(pocket_residues).difference(self.backend.residues)
        for resid in unpaired_residues:
            res = mol[resid]
            self.backend.show_residue(
                res, component, self.backend.settings.RESIDUES_STYLE
            )

    def _hide_hydrogens(
        self, remove_hydrogens: bool | Literal["ligand", "protein", "water"]
    ) -> None:
        to_hide: list[tuple[str, Molecule]] = []
        if remove_hydrogens in {"ligand", True}:
            to_hide.append(("ligand", self.lig_mol))
        if remove_hydrogens in {"protein", True}:
            to_hide.append(("protein", self.prot_mol))
        if remove_hydrogens in {"water", True} and self.water_mol:
            to_hide.append(("water", self.water_mol))
        for component, mol in to_hide:
            keep_indices: list[int] = [
                index
                for index in self._interacting_atoms[component]
                if mol.GetAtomWithIdx(index).GetAtomicNum() == 1
            ]
            self.backend.hide_hydrogens(component, keep_indices)

    def save_png(self, name: str = "prolif-3d.png") -> None:
        """Saves the current state of the 3D viewer to a PNG.

        .. versionadded:: 2.1.0
        """
        if self.interface is None:
            raise ValueError(
                "View not initialized, did you call `display`/`compare` first?",
            )
        self.backend.save_png(name)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the interface."""
        if isinstance(self.backend, Py3DmolBackend):
            if self.interface is None:
                raise ValueError(
                    "View not initialized, did you call `display`/`compare` first?",
                )
            warnings.warn(
                f"Accessing `.{name}` from a Complex3D object is deprecated, "
                "go through the `interface` attribute first, "
                f"i.e. `.interface.{name}` instead of `.{name}`",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self.interface, name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(self.backend, Py3DmolBackend) and hasattr(
            self.backend.settings, name.lower()
        ):
            warnings.warn(
                f"Configuring Py3Dmol styles/options `.{name}` on a Complex3D object "
                "is deprecated, use "
                f"`backend_settings=Py3DmolSettings({name.lower()}={value})` "
                "when creating the plot instead",
                DeprecationWarning,
                stacklevel=2,
            )
            setattr(self.backend.settings, name.lower(), value)
        else:
            super().__setattr__(name, value)

    def _repr_html_(self) -> str | None:
        if self.interface and hasattr(self.interface, "_repr_html_"):
            return self.interface._repr_html_()  # type: ignore[no-any-return]
        return None


@overload
def get_3d_plot_backend(
    settings: Literal["py3Dmol"] | Py3DMolSettings,
) -> Py3DmolBackend: ...
@overload
def get_3d_plot_backend(settings: Literal["pymol"] | PyMOLSettings) -> PyMOLBackend: ...
def get_3d_plot_backend(
    settings: Literal["py3Dmol", "pymol"] | Py3DMolSettings | PyMOLSettings,
) -> Backend:
    """Get the backend for making 3D plots."""
    if isinstance(settings, str):
        if settings == "py3Dmol":
            settings = Py3DMolSettings()
        elif settings == "pymol":
            settings = PyMOLSettings()
        else:
            raise ValueError(f"Unknown backend: {settings}")
    if isinstance(settings, Py3DMolSettings):
        return Py3DmolBackend(settings)
    if isinstance(settings, PyMOLSettings):
        return PyMOLBackend(settings)
    raise TypeError(
        f"Expected Py3DMolSettings or PyMOLSettings, got {type(settings)}",
    )
