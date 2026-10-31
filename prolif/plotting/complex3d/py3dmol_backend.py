""" """

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import py3Dmol
from IPython.display import display_javascript
from rdkit import Chem
from rdkit.Geometry import Point3D

from prolif.plotting.complex3d.base import (
    Complex3DBaseGridBackendSettings,
    Complex3DGridBackend,
)

if TYPE_CHECKING:
    from prolif.molecule import Molecule
    from prolif.residue import Residue, ResidueId
    from prolif.typeshed import Component


@dataclass
class Py3DMolSettings(Complex3DBaseGridBackendSettings[dict[str, dict]]):
    __doc__ = (
        cast(str, Complex3DBaseGridBackendSettings.__doc__)
        + """\
    residue_hover_callback : str
        JavaScript callback executed when hovering a residue involved in an interaction.
    interaction_hover_callback : str
        JavaScript callback executed when hovering an interaction line.
    disable_hover_callback : str
        JavaScript callback executed when the hovering event is finished.
    linked: bool = True
        Link mouse interactions (pan, zoom, translate) on the grid plot.
    """
    )
    ligand_style: dict[str, dict] = field(
        default_factory=lambda: {"stick": {"colorscheme": "cyanCarbon"}}
    )
    residues_style: dict[str, dict] = field(default_factory=lambda: {"stick": {}})
    protein_style: dict[str, dict] = field(
        default_factory=lambda: {"cartoon": {"style": "edged"}}
    )
    peptide_style: dict[str, dict] = field(
        default_factory=lambda: {
            "cartoon": {"style": "edged", "colorscheme": "cyanCarbon"},
        }
    )
    interaction_style: dict = field(
        default_factory=lambda: {
            "radius": 0.15,
            "dashed": True,
            "fromCap": 1,
            "toCap": 1,
        }
    )
    residue_hover_callback: str = """
    function(atom,viewer) {
        if(!atom.label) {
            atom.label = viewer.addLabel(
                atom.resn+atom.resi+'.'+atom.chain+' '+(atom.atom ? atom.atom : atom.serial),
                {position: atom, backgroundColor: 'mintcream', fontColor:'black'}
            );
        }
    }"""  # noqa: E501
    interaction_hover_callback: str = """
    function(shape,viewer) {
        if(!shape.label) {
            shape.label = viewer.addLabel('%s',
                {position: shape, backgroundColor: 'black', fontColor:'white'});
        }
    }"""
    disable_hover_callback: str = """
    function(obj,viewer) {
        if(obj.label) {
            viewer.removeLabel(obj.label);
            delete obj.label;
        }
    }"""
    grid_color_unique: str | None = "magentaCarbon"
    linked: bool = True


class Py3DmolBackend(
    Complex3DGridBackend[Py3DMolSettings, dict[str, dict], int, py3Dmol.view],  # type: ignore[no-any-unimported]
    settings=Py3DMolSettings,
):
    def setup(self) -> None:
        self.interface = py3Dmol.view(
            viewergrid=(1, 1), width=self.settings.size[0], height=self.settings.size[1]
        )
        super().setup()

    def prepare(self) -> None:
        super().prepare()
        self.position = (0, 0)
        self.mols: dict[Component, Molecule] = {}

    def grid_setup(self, n_rows: int, n_columns: int) -> None:
        self.interface = py3Dmol.view(
            viewergrid=(n_rows, n_columns),
            width=self.settings.grid_size[0],
            height=self.settings.grid_size[1],
            linked=self.settings.linked,
        )
        super().setup()

    def grid_prepare(self, row: int, column: int) -> None:
        super().prepare()
        self.position = (row, column)

    def clear(self) -> None:
        self._cmd("removeAllModels")

    def finalize(self) -> None:
        self._cmd("zoomTo", {"model": self.models["ligand"]}, viewer=self.position)

    def _cmd(self, cmd: str, /, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.interface, cmd)(*args, **kwargs)

    def _modelcmd(self, cmd: str, /, *args: Any, **kwargs: Any) -> Any:
        model_id = kwargs.pop("model_id", None)
        if model_id is None:
            model = self._cmd("getModel", viewer=self.position)
        else:
            model = self._cmd("getModel", model_id, viewer=self.position)
        return getattr(model, cmd)(*args, **kwargs)

    def load_molecule(
        self, mol: "Molecule", component: "Component", style: dict[str, dict]
    ) -> None:
        pdb_dump = Chem.MolToPDBBlock(mol, flavor=16 | 32)
        needs_dummy = "cartoon" in style
        if needs_dummy:
            # load dummy model to show requested style (cartoon)
            self._cmd(
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
        self._cmd(
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
        self._modelcmd(
            "setHoverable",
            {},
            True,
            self.settings.residue_hover_callback,
            self.settings.disable_hover_callback,
        )
        self.models[component] = self._model_count
        self._model_count += 1
        self.mols[component] = mol

    def show_residue(
        self, residue: "Residue", component: "Component", style: dict[str, dict]
    ) -> None:
        super().show_residue(residue, component, style)
        model_id = self.models[component]
        resid = residue.resid
        if color_override := self.settings.color_overrides.get(resid):
            resid_style = deepcopy(style)
            for key in resid_style:
                resid_style[key]["colorscheme"] = color_override
        else:
            resid_style = style
        selection = {
            "chain": resid.chain,
            "resn": resid.name,
            "resi": resid.number,
        }
        self._modelcmd("setStyle", selection, resid_style, True, model_id=model_id)

    def add_interaction(
        self,
        interaction: str,
        distance: float,
        components: tuple["Component", "Component"],
        residues: tuple["ResidueId", "ResidueId"],  # noqa: ARG002
        atoms: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    ) -> None:
        points: list[Point3D] = []
        for component, indices in zip(components, atoms, strict=True):
            p = (
                self._get_ring_centroid(component, indices)
                if isinstance(indices, tuple)
                else self.mols[component].GetConformer().GetAtomPosition(indices)
            )
            points.append(p)
        interaction_label = f"{interaction}: {distance:.2f}Å"
        self._cmd(
            "addCylinder",
            {
                "start": {"x": points[0].x, "y": points[0].y, "z": points[0].z},
                "end": {"x": points[1].x, "y": points[1].y, "z": points[1].z},
                "color": self.settings.colors.get(interaction, "grey"),
                **self.settings.interaction_style,
                "hoverable": True,
                "hover_callback": self.settings.interaction_hover_callback
                % interaction_label,
                "unhover_callback": self.settings.disable_hover_callback,
            },
            viewer=self.position,
        )

    def hide_hydrogens(self, component: "Component", keep_indices: list[int]) -> None:
        self._modelcmd(
            "setStyle",
            {"and": [{"elem": "H"}, {"not": {"index": keep_indices}}]},
            {"stick": {"hidden": True}},
            model_id=self.models[component],
        )

    def save_png(self, path: str) -> None:
        uid = self.interface.uniqueid
        display_javascript(
            f"""
            var png = viewer_{uid}.pngURI()
            var a = document.createElement('a')
            a.href = png
            a.download = "{path}"
            a.click()
            a.remove()
            """,
            raw=True,
        )

    def _get_ring_centroid(
        self, component: "Component", indices: tuple[int, ...]
    ) -> Point3D:
        """Get the centroid of a ring system."""
        centroid = self.mols[component].xyz[list(indices)].mean(axis=0)
        return Point3D(*centroid)
