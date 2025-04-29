"""
Plot interactions in 3D --- :mod:`prolif.plotting.complex3d`
============================================================

.. versionadded:: 2.0.0

.. autoclass:: Complex3D
   :members:

"""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, ClassVar, Literal

import py3Dmol
from rdkit import Chem
from rdkit.Geometry import Point3D

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import separated_interaction_colors
from prolif.utils import get_centroid, get_residues_near_ligand, requires

with suppress(ModuleNotFoundError):
    from IPython.display import Javascript, display


if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP
    from prolif.molecule import Molecule
    from prolif.residue import Residue, ResidueId


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

    Attributes
    ----------
    COLORS : dict
        Dictionnary of colors used in the plot for interactions.
    LIGAND_STYLE : dict[str, dict] = {"stick": {"colorscheme": "cyanCarbon"}}
        Style object passed to ``3Dmol.js`` for the ligand.
    RESIDUES_STYLE : dict[str, dict] = {"stick": {}}
        Style object passed to ``3Dmol.js`` for the protein residues involved in
        interactions.
    PROTEIN_STYLE : dict[str, dict] = {"cartoon": {"style": "edged"}}
        Style object passed to ``3Dmol.js`` for the entire protein.
    PEPTIDE_STYLE : dict[str, dict] = "cartoon": {"style": "edged", "colorscheme": "cyanCarbon"}
        Style object passed to ``3Dmol.js`` for the ligand as a peptide if appropriate.
    PEPTIDE_THRESHOLD : int = 2
        Ligands with this number of residues or more will be displayed using
        ``PEPTIDE_STYLE`` in addition to the ``LIGAND_STYLE``.
    LIGAND_DISPLAYED_ATOM : dict[str, int]
        Which atom should be used to display an atom-to-atom interaction for the ligand.
        Refers to the order defined in the SMARTS pattern used in interaction
        definition. Interactions not specified here use ``0`` by default.
    PROTEIN_DISPLAYED_ATOM : dict[str, int]
        Same as :attr:`LIGAND_DISPLAYED_ATOM` for the protein.
    LIGAND_RING_INTERACTIONS : set[str]
        Which interactions should be displayed using the centroid instead of using
        :attr:`LIGAND_DISPLAYED_ATOM` for the ligand.
    PROTEIN_RING_INTERACTIONS : set[str]
        Which interactions should be displayed using the centroid instead of using
        :attr:`PROTEIN_DISPLAYED_ATOM` for the protein.
    RESIDUE_HOVER_CALLBACK : str
        JavaScript callback executed when hovering a residue involved in an interaction.
    INTERACTION_HOVER_CALLBACK : str
        JavaScript callback executed when hovering an interaction line.
    DISABLE_HOVER_CALLBACK : str
        JavaScript callback executed when the hovering event is finished.
    """  # noqa: E501

    COLORS: ClassVar[dict[str, str]] = {**separated_interaction_colors}
    LIGAND_STYLE: ClassVar[dict] = {"stick": {"colorscheme": "cyanCarbon"}}
    RESIDUES_STYLE: ClassVar[dict] = {"stick": {}}
    PROTEIN_STYLE: ClassVar[dict] = {"cartoon": {"style": "edged"}}
    PEPTIDE_STYLE: ClassVar[dict] = {
        "cartoon": {"style": "edged", "colorscheme": "cyanCarbon"},
    }
    PEPTIDE_THRESHOLD: ClassVar[int] = 5
    LIGAND_DISPLAYED_ATOM: ClassVar[dict] = {
        "HBDonor": 1,
        "XBDonor": 1,
    }
    PROTEIN_DISPLAYED_ATOM: ClassVar[dict] = {
        "HBAcceptor": 1,
        "XBAcceptor": 1,
    }
    RING_SYSTEMS: ClassVar[set[str]] = {
        "PiStacking",
        "EdgeToFace",
        "FaceToFace",
    }
    LIGAND_RING_INTERACTIONS: ClassVar[set[str]] = {*RING_SYSTEMS, "PiCation"}
    PROTEIN_RING_INTERACTIONS: ClassVar[set[str]] = {*RING_SYSTEMS, "CationPi"}
    RESIDUE_HOVER_CALLBACK: ClassVar[str] = """
    function(atom,viewer) {
        if(!atom.label) {
            atom.label = viewer.addLabel('%s:'+atom.atom+atom.serial,
                {position: atom, backgroundColor: 'mintcream', fontColor:'black'});
        }
    }"""
    INTERACTION_HOVER_CALLBACK: ClassVar[str] = """
    function(shape,viewer) {
        if(!shape.label) {
            shape.label = viewer.addLabel(shape.interaction,
                {position: shape, backgroundColor: 'black', fontColor:'white'});
        }
    }"""
    DISABLE_HOVER_CALLBACK: ClassVar[str] = """
    function(obj,viewer) {
        if(obj.label) {
            viewer.removeLabel(obj.label);
            delete obj.label;
        }
    }"""

    def __init__(self, ifp: IFP, lig_mol: Molecule, prot_mol: Molecule) -> None:
        self.ifp = ifp
        self.lig_mol = lig_mol
        self.prot_mol = prot_mol
        self._view: py3Dmol.view | None = None  # type: ignore[no-any-unimported]

    @classmethod
    def from_fingerprint(
        cls,
        fp: Fingerprint,
        lig_mol: Molecule,
        prot_mol: Molecule,
        *,
        frame: int,
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
        """
        if not hasattr(fp, "ifp"):
            raise RunRequiredError(
                "Please run the fingerprint analysis before attempting to display"
                " results.",
            )
        ifp = fp.ifp[frame]
        return cls(ifp, lig_mol, prot_mol)

    @staticmethod
    def get_ring_centroid(mol: Molecule, indices: tuple[int, ...]) -> Point3D:
        centroid = mol.xyz[list(indices)].mean(axis=0)
        return Point3D(*centroid)

    def display(
        self,
        size: tuple[int, int] = (650, 600),
        display_all: bool = False,
        only_interacting: bool = True,
        remove_hydrogens: bool | Literal["ligand", "protein"] = True,
    ) -> Complex3D:
        """Display as a py3Dmol widget view.

        Parameters
        ----------
        size: tuple[int, int] = (650, 600)
            The size of the py3Dmol widget view.
        display_all : bool = False
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Not relevant if ``count=False`` in the ``Fingerprint``
            object.
        only_interacting : bool = True
            Whether to show all protein residues in the vicinity of the ligand, or
            only the ones participating in an interaction.
        remove_hydrogens: bool | Literal["ligand", "protein"] = True
            Whether to remove non-polar hydrogens (unless they are involved in an
            interaction).

        .. versionchanged:: 2.1.0
            Added ``only_interacting=True`` and ``remove_hydrogens=True`` parameters.
            Non-polar hydrogen atoms that aren't involved in interactions are now
            hidden.

        """
        v = py3Dmol.view(width=size[0], height=size[1], viewergrid=(1, 1), linked=False)
        v.removeAllModels()
        self._populate_view(
            v,
            position=(0, 0),
            display_all=display_all,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )
        self._view = v
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
        remove_hydrogens: bool | Literal["ligand", "protein"] = True,
    ) -> Complex3D:
        """Displays the initial complex side-by-side with a second one for easier
        comparison.

        Parameters
        ----------
        other: Complex3D
            Other ``Complex3D`` object to compare to.
        size: tuple[int, int] = (900, 600)
            The size of the py3Dmol widget view.
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
        remove_hydrogens: bool | Literal["ligand", "protein"] = True
            Whether to remove non-polar hydrogens (unless they are involved in an
            interaction).

        .. versionadded:: 2.0.1

        .. versionchanged:: 2.1.0
            Added ``only_interacting=True`` and ``remove_hydrogens=True`` parameters.
            Non-polar hydrogen atoms that aren't involved in interactions are now
            hidden.

        """
        v = py3Dmol.view(
            width=size[0],
            height=size[1],
            linked=linked,
            viewergrid=(1, 2),
        )
        v.removeAllModels()

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
        self._populate_view(
            v,
            position=(0, 0),
            display_all=display_all,
            colormap=highlights,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )

        # get residues with interactions specific to pose 2
        highlights = (
            {r[0]: color_unique for r in interactions2 - interactions1}
            if color_unique
            else {}
        )
        other._populate_view(
            v,
            position=(0, 1),
            display_all=display_all,
            colormap=highlights,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )
        self._view = v
        return self

    def _populate_view(  # type: ignore[no-any-unimported]  # noqa: PLR0912
        self,
        view: py3Dmol.view | Complex3D,
        position: tuple[int, int] = (0, 0),
        display_all: bool = False,
        colormap: dict[ResidueId, str] | None = None,
        only_interacting: bool = True,
        remove_hydrogens: bool | Literal["ligand", "protein"] = True,
    ) -> None:
        if isinstance(view, Complex3D):
            if view._view is None:
                raise ValueError(
                    "View not initialized, did you call `display`/`compare` first?",
                )
            v = view._view
        else:
            v = view
        self._colormap = {} if colormap is None else colormap
        self._models: dict[ResidueId, int] = {}
        self._mid = -1
        self._interacting_atoms: dict[str, set[int]] = {
            "ligand": set(),
            "protein": set(),
        }

        # show all interacting residues
        for (lresid, presid), interactions in self.ifp.items():
            lres = self.lig_mol[lresid]
            pres = self.prot_mol[presid]
            # set model ids for reusing later
            for resid, res, style in [
                (lresid, lres, self.LIGAND_STYLE),
                (presid, pres, self.RESIDUES_STYLE),
            ]:
                if resid not in self._models:
                    self._add_residue_to_view(v, position, res, style)
            for interaction, metadata_tuple in interactions.items():
                # whether to display all interactions or only the one with the shortest
                # distance
                metadata_iterator = (
                    metadata_tuple
                    if display_all
                    else (
                        min(
                            metadata_tuple,
                            key=lambda m: m.get("distance", float("nan")),
                        ),
                    )
                )
                for metadata in metadata_iterator:
                    # record indices of atoms interacting
                    self._interacting_atoms["ligand"].update(
                        metadata["parent_indices"]["ligand"]
                    )
                    self._interacting_atoms["protein"].update(
                        metadata["parent_indices"]["protein"]
                    )

                    # get coordinates for both points of the interaction
                    if interaction in self.LIGAND_RING_INTERACTIONS:
                        p1 = self.get_ring_centroid(lres, metadata["indices"]["ligand"])
                    else:
                        p1 = lres.GetConformer().GetAtomPosition(
                            metadata["indices"]["ligand"][
                                self.LIGAND_DISPLAYED_ATOM.get(interaction, 0)
                            ],
                        )
                    if interaction in self.PROTEIN_RING_INTERACTIONS:
                        p2 = self.get_ring_centroid(
                            pres,
                            metadata["indices"]["protein"],
                        )
                    else:
                        p2 = pres.GetConformer().GetAtomPosition(
                            metadata["indices"]["protein"][
                                self.PROTEIN_DISPLAYED_ATOM.get(interaction, 0)
                            ],
                        )
                    # add interaction line
                    v.addCylinder(
                        {
                            "start": {"x": p1.x, "y": p1.y, "z": p1.z},
                            "end": {"x": p2.x, "y": p2.y, "z": p2.z},
                            "color": self.COLORS.get(interaction, "grey"),
                            "radius": 0.15,
                            "dashed": True,
                            "fromCap": 1,
                            "toCap": 1,
                        },
                        viewer=position,
                    )
                    # add label when hovering the middle of the dashed line by adding a
                    # dummy atom
                    c = Point3D(*get_centroid([p1, p2]))
                    modelID = self._models[lresid]
                    model = v.getModel(modelID, viewer=position)
                    interaction_label = f"{interaction}: {metadata['distance']:.2f}Å"
                    model.addAtoms(
                        [
                            {
                                "elem": "Z",
                                "x": c.x,
                                "y": c.y,
                                "z": c.z,
                                "interaction": interaction_label,
                            },
                        ],
                    )
                    model.setStyle(
                        {"interaction": interaction_label},
                        {"clicksphere": {"radius": 0.5}},
                    )
                    model.setHoverable(
                        {"interaction": interaction_label},
                        True,
                        self.INTERACTION_HOVER_CALLBACK,
                        self.DISABLE_HOVER_CALLBACK,
                    )

        # show "protein" residues that are close to the "ligand"
        if not only_interacting:
            pocket_residues = get_residues_near_ligand(self.lig_mol, self.prot_mol)
            pocket_residues = set(pocket_residues).difference(self._models)
            for resid in pocket_residues:
                res = self.prot_mol[resid]
                self._add_residue_to_view(v, position, res, self.RESIDUES_STYLE)

        # hide non-polar hydrogens (except if they are involved in an interaction)
        if remove_hydrogens:
            to_remove = []
            if remove_hydrogens in {"ligand", True}:
                to_remove.append(("ligand", self.lig_mol))
            if remove_hydrogens in {"protein", True}:
                to_remove.append(("protein", self.prot_mol))

            for resid in self._models:
                for moltype, mol in to_remove:
                    try:
                        modelID = self._models[resid]
                        res = mol[resid]
                    except KeyError:
                        continue
                    model = v.getModel(modelID, viewer=position)
                    int_atoms = self._interacting_atoms[moltype]
                    hide = [
                        a.GetIdx()
                        for a in res.GetAtoms()
                        if a.GetAtomicNum() == 1
                        and a.GetUnsignedProp("mapindex") not in int_atoms
                        and all(n.GetAtomicNum() in {1, 6} for n in a.GetNeighbors())
                    ]
                    model.setStyle({"index": hide}, {"stick": {"hidden": True}})

        # show protein
        mol = Chem.RemoveAllHs(self.prot_mol, sanitize=False)
        pdb = Chem.MolToPDBBlock(mol, flavor=0x20 | 0x10)
        v.addModel(pdb, "pdb", viewer=position)
        model = v.getModel(viewer=position)
        model.setStyle({}, self.PROTEIN_STYLE)

        # do the same for ligand if multiple residues
        if self.lig_mol.n_residues >= self.PEPTIDE_THRESHOLD:
            mol = Chem.RemoveAllHs(self.lig_mol, sanitize=False)
            pdb = Chem.MolToPDBBlock(mol, flavor=0x20 | 0x10)
            v.addModel(pdb, "pdb", viewer=position)
            model = v.getModel(viewer=position)
            model.setStyle({}, self.PEPTIDE_STYLE)

        v.zoomTo({"model": list(self._models.values())}, viewer=position)

    def _add_residue_to_view(  # type: ignore[no-any-unimported]
        self,
        v: py3Dmol.view,
        position: tuple[int, int],
        res: Residue,
        style: dict,
    ) -> None:
        self._mid += 1
        resid = res.resid
        v.addModel(Chem.MolToMolBlock(res), "sdf", viewer=position)
        model = v.getModel(viewer=position)
        if resid in self._colormap:
            resid_style = deepcopy(style)
            for key in resid_style:
                resid_style[key]["colorscheme"] = self._colormap[resid]
        else:
            resid_style = style
        model.setStyle({}, resid_style)
        # add residue label
        model.setHoverable(
            {},
            True,
            self.RESIDUE_HOVER_CALLBACK % resid,
            self.DISABLE_HOVER_CALLBACK,
        )
        self._models[resid] = self._mid

    @requires("IPython.display")
    def save_png(self) -> None:
        """Saves the current state of the 3D viewer to a PNG. Not available outside of a
        notebook.

        .. versionadded:: 2.1.0
        """
        if self._view is None:
            raise ValueError(
                "View not initialized, did you call `display`/`compare` first?",
            )
        uid = self._view.uniqueid
        display(
            Javascript(f"""
            var png = viewer_{uid}.pngURI()
            var a = document.createElement('a')
            a.href = png
            a.download = "prolif-3d.png"
            a.click()
            a.remove()
            """),
        )

    def _repr_html_(self) -> str | None:  # noqa: PLW3201
        if self._view:
            return self._view._repr_html_()  # type: ignore[no-any-return]
        return None
