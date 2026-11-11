"""
Plot interactions in 3D --- :mod:`prolif.plotting.complex3d`
============================================================

.. versionadded:: 2.0.0

.. autoclass:: Complex3D
   :members:

.. autofunction:: get_3d_plot_backend

"""

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from prolif.exceptions import RunRequiredError
from prolif.plotting.complex3d.base import (
    COMPLEX3D_BACKENDS,
    Complex3DBackend,
    Complex3DBaseBackendSettings,
    Complex3DGridBackend,
    get_backend,
)
from prolif.plotting.complex3d.py3dmol_backend import Py3DmolBackend, Py3DmolSettings
from prolif.plotting.complex3d.pymol_backend import PyMOLBackend, PyMOLSettings
from prolif.plotting.utils import metadata_iterator
from prolif.residue import ResidueId
from prolif.utils import get_residues_near_ligand

if TYPE_CHECKING:
    import py3Dmol

    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP
    from prolif.molecule import Molecule
    from prolif.plotting.complex3d.pymol_backend import PyMOLScreenshot
    from prolif.typeshed import Component

BackendSettingsT = TypeVar("BackendSettingsT", bound=Complex3DBaseBackendSettings)
InterfaceT = TypeVar("InterfaceT")

Complex3DBackendSettings: TypeAlias = (
    Literal["py3dmol", "pymol"]
    | Py3DmolSettings
    | PyMOLSettings
    | str
    | Complex3DBaseBackendSettings
)


class Complex3D(Generic[BackendSettingsT, InterfaceT]):
    """Creates a py3Dmol plot of interactions.

    Parameters
    ----------
    ifp : IFP
        The interaction dictionary for a single frame.
    lig_mol : Molecule
        The ligand molecule to display.
    prot_mol : Molecule
        The protein molecule to display.
    water_mol : Molecule | None
        Additional molecule (e.g. waters) to display.
    backend_settings: Complex3DBackendSettings
        The backend or backend settings to use. If a string is provided, the
        relevant backend is used with default settings. If a
        :class:`prolif.plotting.complex3d.base.Complex3DBaseBackendSettings` object
        is provided, this will be used instead.

    .. versionchanged:: 2.1.0
        Added ``water_mol`` parameter to the constructor to display waters
        involved in WaterBridge interactions. Added ``save_png`` method to save the
        current state of the 3D viewer to a PNG. Added ``remove_hydrogens`` parameter
        to the ``display`` and ``compare`` methods to remove non-polar hydrogens that
        aren't involved in an interaction. Added ``only_interacting`` parameter to the
        ``display`` and ``compare`` methods to show all protein residues in the
        vicinity of the ligand, or only the ones participating in an interaction.

    .. versionchanged:: 3.0.0
        Added support for multiple backends (``py3dmol`` and ``pymol``) and moved
        most options to the ``backend_settings`` object (accessible through
        :attr:`backend`'s :attr:`settings`).
    """

    @overload
    def __init__(  # type: ignore[no-any-unimported]
        self: "Complex3D[Py3DmolSettings, py3Dmol.view]",
        ifp: "IFP",
        lig_mol: "Molecule",
        prot_mol: "Molecule",
        water_mol: "Molecule | None",
        backend_settings: Literal["py3dmol"] | Py3DmolSettings,
    ) -> None: ...
    @overload
    def __init__(
        self: "Complex3D[PyMOLSettings, PyMOLScreenshot]",
        ifp: "IFP",
        lig_mol: "Molecule",
        prot_mol: "Molecule",
        water_mol: "Molecule | None",
        backend_settings: Literal["pymol"] | PyMOLSettings,
    ) -> None: ...
    @overload
    def __init__(
        self: "Complex3D[Complex3DBaseBackendSettings, Any]",
        ifp: "IFP",
        lig_mol: "Molecule",
        prot_mol: "Molecule",
        water_mol: "Molecule | None",
        backend_settings: str | Complex3DBaseBackendSettings,
    ) -> None: ...
    def __init__(
        self,
        ifp: "IFP",
        lig_mol: "Molecule",
        prot_mol: "Molecule",
        water_mol: "Molecule | None" = None,
        backend_settings: "Complex3DBackendSettings" = "py3dmol",
    ) -> None:
        self.backend = cast(
            Complex3DBackend[BackendSettingsT, Any, Any, InterfaceT],
            get_3d_plot_backend(backend_settings),
        )
        self.ifp = ifp
        self.lig_mol = lig_mol
        self.prot_mol = prot_mol
        self.water_mol = water_mol
        self.interface: InterfaceT | None = None

    @classmethod
    def from_fingerprint(
        cls,
        fp: "Fingerprint",
        lig_mol: "Molecule",
        prot_mol: "Molecule",
        water_mol: "Molecule | None" = None,
        *,
        frame: int,
        backend_settings: "Complex3DBackendSettings" = "py3dmol",
    ) -> Self:
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
        water_mol : Molecule | None
            Additional molecule (e.g. waters) to display.
        backend_settings: Complex3DBackendSettings
            The backend or backend settings to use. If a string is provided, the
            relevant backend is used with default settings. If a
            :class:`prolif.plotting.complex3d.base.Complex3DBaseBackendSettings` object
            is provided, this will be used instead.
        """
        if not hasattr(fp, "ifp"):
            raise RunRequiredError(
                "Please run the fingerprint analysis before attempting to display"
                " results.",
            )
        ifp = fp.ifp[frame]
        return cls(ifp, lig_mol, prot_mol, water_mol, backend_settings=backend_settings)

    def display(
        self,
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: "bool | Component" = True,
    ) -> Self:
        """Display the complex in 3D.

        Parameters
        ----------
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

        .. versionchanged:: 2.2.0
            Added ``sanitize`` parameter to allow sanitization of the RDKit
            molecules used for visualization, which can help avoid unkekulization
            issues when using the coordinates of the molecule directly.

        .. versionchanged:: X.Y.Z
            Removed ``sanitize`` parameter as unrequired following the backend changes.

        """
        # setup view parameters
        self.backend.setup()
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
        *others: "Complex3D",
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: "bool | Component" = True,
    ) -> Self:
        """Displays the initial complex side-by-side with a second one for easier
        comparison.

        Parameters
        ----------
        *other: Complex3D
            Other ``Complex3D`` object(s) to compare to.
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

        .. versionadded:: 2.0.1

        .. versionchanged:: 2.1.0
            Added ``only_interacting=True`` and ``remove_hydrogens=True`` parameters.
            Non-polar hydrogen atoms that aren't involved in interactions are now
            hidden. Added support for waters involved in WaterBridge interactions.

        """
        supported = ", ".join(
            sorted(
                {
                    spec.name
                    for spec in COMPLEX3D_BACKENDS.values()
                    if issubclass(spec.backend, Complex3DGridBackend)
                }
            )
        )
        if not isinstance(self.backend, Complex3DGridBackend):
            raise TypeError(
                "The backend for this complex does not support grid plots, please use "
                f"a compatible backend instead: {supported}.",
            )
        if not all(isinstance(other.backend, type(self.backend)) for other in others):
            raise TypeError("The backend for all complexes must be the same")

        # configure view parameters
        self.backend.grid_setup()
        interactions = [
            {
                (resid[1], i)
                for resid, interactions in c.ifp.items()
                for i in interactions
            }
            for c in cast(tuple[Complex3D, ...], (self, *others))
        ]
        # get residues with interactions specific to each complex
        unique = [
            i1.difference(set().union(*(i2 for i2 in interactions if i2 is not i1)))
            for i1 in interactions
        ]
        # make corresponding color overrides for each complex
        highlights = [
            {i[0]: color for i in u} if (color := settings.grid_color_unique) else {}
            for u, settings in zip(
                unique,
                (self.backend.settings, *[o.backend.settings for o in others]),
                strict=True,
            )
        ]

        # first plot
        self.backend.grid_prepare(0)
        settings1 = self.backend.settings
        with settings1.overrides(
            color_overrides=settings1.color_overrides | highlights[0]
        ):
            self._populate_view(
                display_all=display_all,
                only_interacting=only_interacting,
                remove_hydrogens=remove_hydrogens,
            )

        for i, (other, highlight) in enumerate(
            zip(others, highlights[1:], strict=True), start=1
        ):
            if not isinstance(other.backend, Complex3DGridBackend):
                raise TypeError(
                    "The backend for these complexes does not support grid plots, "
                    f"please use one of these compatible backends instead: {supported}",
                )

            # copy current plot state to other
            settings2 = other.backend.settings
            backend = other.backend = self.backend
            backend.settings = settings2

            with settings2.overrides(
                color_overrides=settings2.color_overrides | highlight,
                n_rows=settings1.n_rows,
                n_columns=settings1.n_columns,
            ):
                backend.grid_prepare(i)
                other._populate_view(
                    display_all=display_all,
                    only_interacting=only_interacting,
                    remove_hydrogens=remove_hydrogens,
                )

            # restore
            other.backend = get_3d_plot_backend(settings2)
        backend.settings = settings1
        self.interface = backend.interface
        return self

    def add(
        self,
        other: "Complex3D",
        *,
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: "bool | Component" = True,
    ) -> Self:
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

    def _populate_view(
        self,
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: "bool | Component" = True,
    ) -> None:
        backend = self.backend
        settings = backend.settings
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

        self._interacting_atoms: defaultdict["Component", set[int]] = defaultdict(set)
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
                    backend.show_residue(res, component, style)  # type:ignore[arg-type]
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
                                res1 = lresid
                            else:
                                # water-water or water-protein
                                atoms1 = metadata["parent_indices"][
                                    settings.bridged_interactions[interaction]
                                ][settings.protein_displayed_atoms.get(interaction, 0)]
                                res1 = ResidueId.from_string(src)
                            if dest == "protein":
                                atoms2 = metadata["parent_indices"]["protein"][
                                    settings.protein_displayed_atoms.get(interaction, 0)
                                ]
                                res2 = presid
                            else:
                                atoms2 = metadata["parent_indices"][
                                    settings.bridged_interactions[interaction]
                                ][settings.protein_displayed_atoms.get(interaction, 0)]
                                res2 = ResidueId.from_string(dest)

                            backend.add_interaction(
                                interaction,
                                distance=metadata[distlabel],
                                components=(
                                    "water" if src != "ligand" else "ligand",
                                    "water" if dest != "protein" else "protein",
                                ),
                                residues=(res1, res2),
                                atoms=(atoms1, atoms2),
                            )

                    else:
                        # get coordinates for both points of the interaction
                        atoms1 = (
                            metadata["parent_indices"]["ligand"]
                            if interaction in settings.ligand_ring_interactions
                            else metadata["parent_indices"]["ligand"][
                                settings.ligand_displayed_atoms.get(interaction, 0)
                            ]
                        )
                        atoms2 = (
                            metadata["parent_indices"]["protein"]
                            if interaction in settings.protein_ring_interactions
                            else metadata["parent_indices"]["protein"][
                                settings.protein_displayed_atoms.get(interaction, 0)
                            ]
                        )
                        # add interaction line
                        dist = metadata.get("distance", float("nan"))
                        backend.add_interaction(
                            interaction,
                            distance=dist,
                            components=("ligand", "protein"),
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

    def _show_unpaired_residues(self, mol: "Molecule", component: "Component") -> None:
        pocket_residues = get_residues_near_ligand(self.lig_mol, mol)
        unpaired_residues = set(pocket_residues).difference(self.backend.residues)
        for resid in unpaired_residues:
            res = mol[resid]
            self.backend.show_residue(
                res, component, self.backend.settings.residues_style
            )

    def _hide_hydrogens(
        self, remove_hydrogens: bool | Literal["ligand", "protein", "water"]
    ) -> None:
        to_hide: list[tuple["Component", "Molecule"]] = []
        if remove_hydrogens in {"ligand", True}:
            to_hide.append(("ligand", self.lig_mol))
        if remove_hydrogens in {"protein", True}:
            to_hide.append(("protein", self.prot_mol))
        if remove_hydrogens in {"water", True} and self.water_mol:
            # keep all hydrogens in interacting water molecules
            interacting_waters = {
                ResidueId.from_atom(self.water_mol.GetAtomWithIdx(i))
                for i in self._interacting_atoms["water"]
            }
            keep_indices: list[int] = [
                atom.GetIdx()
                for atom in self.water_mol.GetAtoms()
                if atom.GetAtomicNum() == 1
                and ResidueId.from_atom(atom) in interacting_waters
            ]
            self.backend.hide_hydrogens("water", keep_indices)

        for component, mol in to_hide:
            keep_indices = [
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

    def _repr_html_(self) -> str | None:
        if self.interface and hasattr(self.interface, "_repr_html_"):
            return self.interface._repr_html_()  # type: ignore[no-any-return]
        return None


@overload
def get_3d_plot_backend(
    backend: Literal["py3dmol"] | Py3DmolSettings,
) -> Py3DmolBackend: ...
@overload
def get_3d_plot_backend(backend: Literal["pymol"] | PyMOLSettings) -> PyMOLBackend: ...
@overload
def get_3d_plot_backend(
    backend: "str |Complex3DBaseBackendSettings",
) -> Complex3DBackend: ...
def get_3d_plot_backend(backend: "Complex3DBackendSettings") -> Complex3DBackend:
    """Get the backend for making 3D plots."""
    spec = get_backend(backend)
    settings = (
        backend
        if isinstance(backend, Complex3DBaseBackendSettings)
        else spec.settings()  # type: ignore[call-arg]
    )
    return spec.backend(settings)
