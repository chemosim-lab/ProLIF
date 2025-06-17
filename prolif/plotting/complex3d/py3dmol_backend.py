""" """

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import py3Dmol
from rdkit import Chem

from prolif.plotting.complex3d.base import Backend, Settings

if TYPE_CHECKING:
    from rdkit.Geometry import Point3D

    from prolif.molecule import Molecule
    from prolif.residue import Residue, ResidueId


@dataclass
class Py3DMolSettings(Settings[dict[str, dict]]):
    __doc__ = (
        cast(str, Settings.__doc__)
        + """\
    RESIDUE_HOVER_CALLBACK : str
        JavaScript callback executed when hovering a residue involved in an interaction.
    INTERACTION_HOVER_CALLBACK : str
        JavaScript callback executed when hovering an interaction line.
    DISABLE_HOVER_CALLBACK : str
        JavaScript callback executed when the hovering event is finished.
    """
    )
    LIGAND_STYLE: dict[str, dict] = field(
        default_factory=lambda: {"stick": {"colorscheme": "cyanCarbon"}}
    )
    RESIDUES_STYLE: dict[str, dict] = field(default_factory=lambda: {"stick": {}})
    PROTEIN_STYLE: dict[str, dict] = field(
        default_factory=lambda: {"cartoon": {"style": "edged"}}
    )
    PEPTIDE_STYLE: dict[str, dict] = field(
        default_factory=lambda: {
            "cartoon": {"style": "edged", "colorscheme": "cyanCarbon"},
        }
    )
    INTERACTION_STYLE: dict = field(
        default_factory=lambda: {
            "radius": 0.15,
            "dashed": True,
            "fromCap": 1,
            "toCap": 1,
        }
    )
    RESIDUE_HOVER_CALLBACK: str = """
    function(atom,viewer) {
        if(!atom.label) {
            atom.label = viewer.addLabel(
                atom.resn+atom.resi+'.'+atom.chain+':'+(atom.atom ? atom.atom : atom.serial),
                {position: atom, backgroundColor: 'mintcream', fontColor:'black'}
            );
        }
    }"""
    INTERACTION_HOVER_CALLBACK: str = """
    function(shape,viewer) {
        if(!shape.label) {
            shape.label = viewer.addLabel('%s',
                {position: shape, backgroundColor: 'black', fontColor:'white'});
        }
    }"""
    DISABLE_HOVER_CALLBACK: str = """
    function(obj,viewer) {
        if(obj.label) {
            viewer.removeLabel(obj.label);
            delete obj.label;
        }
    }"""


class Py3DmolBackend(Backend[Py3DMolSettings, str, int]):
    def setup(
        self,
        viewergrid: tuple[int, int],
        **view_kwargs: Any,
    ) -> None:
        self.view = py3Dmol.view(viewergrid=viewergrid, **view_kwargs)
        super().setup()

    def prepare(
        self,
        position: tuple[int, int] = (0, 0),
        colormap: dict[ResidueId, str] | None = None,
    ) -> None:
        super().prepare()
        self.position = position
        self.colormap = {} if colormap is None else colormap

    def clear(self) -> None:
        self.viewcmd("removeAllModels")

    def finalize(self) -> None:
        self.viewcmd("zoomTo", {"model": self.models["ligand"]}, viewer=self.position)

    def viewcmd(self, cmd: str, /, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.view, cmd)(*args, **kwargs)

    def modelcmd(self, cmd: str, /, *args: Any, **kwargs: Any) -> Any:
        model_id = kwargs.pop("model_id", None)
        if model_id is None:
            model = self.viewcmd("getModel", viewer=self.position)
        else:
            model = self.viewcmd("getModel", model_id, viewer=self.position)
        return getattr(model, cmd)(*args, **kwargs)

    def load_molecule(
        self, mol: "Molecule", component: str, style: dict[str, dict]
    ) -> None:
        pdb_dump = Chem.MolToPDBBlock(mol, flavor=16 | 32)
        needs_dummy = "cartoon" in style
        if needs_dummy:
            # load dummy model to show requested style (cartoon)
            self.viewcmd(
                "addModel",
                pdb_dump,
                "pdb",
                {
                    "assignBonds": False,
                    "keepH": False,
                    "style": style,
                },
                viewer=self.position,
            )
            self._model_count += 1
        # load actual model with hydrogens and handling of events
        self.viewcmd(
            "addModel",
            pdb_dump,
            "pdb",
            {
                "assignBonds": False,
                "keepH": True,
                "style": {"clicksphere": {"radius": 0.4}} if needs_dummy else style,
            },
            viewer=self.position,
        )
        self.modelcmd(
            "setHoverable",
            {},
            True,
            self.settings.RESIDUE_HOVER_CALLBACK,
            self.settings.DISABLE_HOVER_CALLBACK,
        )
        self.models[component] = self._model_count
        self._model_count += 1

    def show_residue(
        self, residue: "Residue", component: str, style: dict[str, dict]
    ) -> None:
        super().show_residue(residue, component, style)
        model_id = self.models[component]
        resid = residue.resid
        if resid in self.colormap:
            resid_style = deepcopy(style)
            for key in resid_style:
                resid_style[key]["colorscheme"] = self.colormap[resid]
        else:
            resid_style = style
        selection = {
            "chain": resid.chain,
            "resn": resid.name,
            "resi": resid.number,
        }
        self.modelcmd("setStyle", selection, resid_style, True, model_id=model_id)

    def add_interaction(
        self,
        interaction: str,
        distance: float,
        points: tuple["Point3D", "Point3D"],
        residues: tuple["ResidueId", "ResidueId"],
        atoms: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    ) -> None:
        p1, p2 = points
        interaction_label = f"{interaction}: {distance:.2f}Å"
        self.viewcmd(
            "addCylinder",
            {
                "start": {"x": p1.x, "y": p1.y, "z": p1.z},
                "end": {"x": p2.x, "y": p2.y, "z": p2.z},
                "color": self.settings.COLORS.get(interaction, "grey"),
                **self.settings.INTERACTION_STYLE,
                "hoverable": True,
                "hover_callback": self.settings.INTERACTION_HOVER_CALLBACK
                % interaction_label,
                "unhover_callback": self.settings.DISABLE_HOVER_CALLBACK,
            },
            viewer=self.position,
        )

    def hide_hydrogens(self, component: str, keep_indices: list[int]) -> None:
        self.modelcmd(
            "setStyle",
            {"and": [{"elem": "H"}, {"not": {"index": keep_indices}}]},
            {"stick": {"hidden": True}},
            model_id=self.models[component],
        )
