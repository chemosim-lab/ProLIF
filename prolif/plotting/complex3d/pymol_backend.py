from __future__ import annotations

import atexit
import os
import struct
from base64 import b64encode
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from tempfile import mktemp
from time import sleep
from typing import TYPE_CHECKING, Any, cast
from xmlrpc.client import ServerProxy

from rdkit import Chem

from prolif.plotting.complex3d.base import Backend, Settings
from prolif.typeshed import PyMOLRPCServer

if TYPE_CHECKING:
    from rdkit.Geometry import Point3D

    from prolif.molecule import Molecule
    from prolif.residue import Residue, ResidueId


@dataclass
class PyMOLSettings(Settings):
    LIGAND_STYLE: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: {
            "stick": [],
        }
    )
    RESIDUES_STYLE: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: {
            "stick": [],
        }
    )
    PROTEIN_STYLE: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: {
            "cartoon": [],
        }
    )
    PEPTIDE_STYLE: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: {
            "cartoon": [("cartoon_color", "cyan")],
        }
    )


@cache
def get_rpc_server() -> PyMOLRPCServer:
    """Get proxy to the PyMOL RPC server."""
    host = os.environ.get("PYMOL_RPCHOST", "localhost")
    port = 9123
    proxy = cast(PyMOLRPCServer, ServerProxy(f"http://{host}:{port}/RPC2"))
    proxy.ping()
    atexit.register(proxy.__close)
    return proxy


class PyMOLBackend(Backend[PyMOLSettings, str, str]):
    def setup(
        self,
        handler: Callable[[str], None] | None = None,
        group: str = "prolif",
        **kwargs: Any,
    ) -> None:
        self.cmd = handler or get_rpc_server().do
        self.group = group
        self.view = PyMOLScreenshot(callback=self.cmd, kwargs=kwargs)
        super().setup()

    def prepare(self) -> None:
        super().prepare()
        self.interactions: set[str] = set()
        self.cmd("set group_auto_mode, 2")

    def clear(self) -> None:
        self.cmd("reinitialize")

    def finalize(self) -> None:
        self.cmd("valence guess, all")
        model_id = self.models["ligand"]
        self.cmd(f"zoom %{model_id}")

    @contextmanager
    def ignore_autozoom(self) -> Iterator[None]:
        """Context manager to ignore the automatic zoom when loading an object."""
        self.cmd("view rdinterface, store")
        yield
        self.cmd("view rdinterface, recall")

    def load_molecule(
        self, mol: "Molecule", component: str, style: dict[str, list[tuple[str, str]]]
    ) -> None:
        pdb_dump = Chem.MolToPDBBlock(mol, flavor=16 | 32)
        model_id = f"{self.group}.{component}"
        with self.ignore_autozoom():
            self.cmd(f"cmd.read_pdbstr({pdb_dump!r}, {model_id!r})")
        self._model_count += 1
        self.models[component] = model_id
        self.apply_style(f"%{model_id}", style)

    def show_residue(
        self,
        residue: "Residue",
        component: str,
        style: dict[str, list[tuple[str, str]]],
    ) -> None:
        super().show_residue(residue, component, style)
        model_id = self.models[component]
        resid = residue.resid
        selection = (
            f"%{model_id} and chain {resid.chain} and resid {resid.number} "
            f"and resname {resid.name}"
        )
        self.apply_style(selection, style)

    def apply_style(
        self, selection: str, style: dict[str, list[tuple[str, str]]]
    ) -> None:
        for representation, extras in style.items():
            self.cmd(f"show {representation}, {selection}")
            if extras:
                for key, value in extras:
                    self.cmd(f"set {key}, {value}, {selection}")

    def hide_hydrogens(self, component: str, keep_indices: list[int]) -> None:
        model_id = self.models[component]
        selection = " or ".join([f"rank {i}" for i in keep_indices])
        self.cmd(f"hide %{model_id} and elem H and not ({selection})")

    def add_interaction(
        self,
        interaction: str,
        distance: float,
        points: tuple["Point3D", "Point3D"],
        residues: tuple["ResidueId", "ResidueId"],
        atoms: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    ) -> None:
        lresid, presid = residues
        latoms, patoms = atoms
        colour = (
            self.settings.COLORS.get(interaction, "#dedede").upper().replace("#", "0x")
        )
        selections = []
        for component, indices in [("ligand", latoms), ("protein", patoms)]:
            if isinstance(indices, int):
                sel = f"%{self.group}.{component} and (rank {indices})"
            else:
                ring_indices = " or rank ".join(map(str, indices))
                sel = f"%{self.group}.{component} and (rank {ring_indices})"
            selections.append(sel)

        resids = f"{lresid}_{presid}".replace(".", "")
        atom_ids = "_".join(
            [
                "".join(map(str, (atoms,) if isinstance(atoms, int) else atoms))
                for atoms in (latoms, patoms)
            ]
        )
        name = f"{self.group}.interactions.{interaction}.{resids}.{atom_ids}"
        self.cmd(f"distance {name}, {', '.join(selections)}, mode=4")
        self.cmd(f"hide labels, {name}")
        self.cmd(f"color {colour}, {name}")
        self.interactions.add(interaction)


@dataclass
class PyMOLScreenshot:
    callback: Callable[[str], None]
    kwargs: dict

    def _repr_png_(self) -> bytes:
        # wait for all commands to be done
        self.callback("cmd.sync()")
        # write to temp PNG file
        path = Path(mktemp(suffix=".png"))
        kwargs = ""
        if self.kwargs:
            kwargs = ", "
            kwargs += ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        self.callback(f"png {path}{kwargs}")
        # wait for end chunk of PNG to be written
        while not (
            path.is_file()
            and path.stat().st_size > 12
            and struct.unpack("!I4sI", path.read_bytes()[-12:])
            == (0, b"IEND", 2923585666)
        ):
            sleep(0.1)
        return path.read_bytes()

    def _repr_html_(self) -> str:
        png = self._repr_png_()
        dump = b64encode(png).decode()
        return (
            f'<image alt="ProLIF PyMOL screenshot" src="data:image/png;base64, '
            f'{dump}" />'
        )
