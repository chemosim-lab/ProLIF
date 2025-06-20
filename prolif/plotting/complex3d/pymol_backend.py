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
    ligand_style: dict[str, list[str]] = field(
        default_factory=lambda: {
            "stick": ["util.cbac {}"],
        }
    )
    residues_style: dict[str, list[str]] = field(
        default_factory=lambda: {
            "stick": ["util.cbag {}"],
        }
    )
    protein_style: dict[str, list[str]] = field(
        default_factory=lambda: {
            "cartoon": ["set cartoon_color, green, {}"],
        }
    )
    peptide_style: dict[str, list[str]] = field(
        default_factory=lambda: {
            "cartoon": ["set cartoon_color, cyan, {}"],
        }
    )
    group_id: str = "complex"


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
        **kwargs: Any,
    ) -> None:
        self.cmd = handler or get_rpc_server().do
        self.interface = PyMOLScreenshot(
            callback=self.cmd, kwargs=kwargs, is_interactive=handler is None
        )
        super().setup()

    def prepare(self) -> None:
        super().prepare()
        self.interactions: set[str] = set()
        self.group_id = self.settings.group_id
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
        self, mol: "Molecule", component: str, style: dict[str, list[str]]
    ) -> None:
        pdb_dump = Chem.MolToPDBBlock(mol, flavor=16 | 32)
        model_id = f"{self.group_id}.{component}"
        with self.ignore_autozoom():
            self.cmd(f"cmd.read_pdbstr({pdb_dump!r}, {model_id!r})")
        self._model_count += 1
        self.models[component] = model_id
        self.apply_style(f"%{model_id}", style)

    def show_residue(
        self,
        residue: "Residue",
        component: str,
        style: dict[str, list[str]],
    ) -> None:
        super().show_residue(residue, component, style)
        model_id = self.models[component]
        resid = residue.resid
        selection = (
            f"%{model_id} and chain {resid.chain} and resid {resid.number} "
            f"and resname {resid.name}"
        )
        self.apply_style(selection, style)

    def apply_style(self, selection: str, style: dict[str, list[str]]) -> None:
        for representation, extras in style.items():
            self.cmd(f"show {representation}, {selection}")
            if extras:
                for extra in extras:
                    self.cmd(extra.format(selection))

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
            self.settings.colors.get(interaction, "#dedede").upper().replace("#", "0x")
        )
        selections = []
        for component, indices in [("ligand", latoms), ("protein", patoms)]:
            if isinstance(indices, int):
                sel = f"%{self.group_id}.{component} and (rank {indices})"
            else:
                ring_indices = " or rank ".join(map(str, indices))
                sel = f"%{self.group_id}.{component} and (rank {ring_indices})"
            selections.append(sel)

        resids = f"{lresid}_{presid}".replace(".", "")
        atom_ids = "_".join(
            [
                "".join(map(str, (atoms,) if isinstance(atoms, int) else atoms))
                for atoms in (latoms, patoms)
            ]
        )
        name = f"{self.group_id}.interactions.{interaction}.{resids}.{atom_ids}"
        self.cmd(f"distance {name}, {', '.join(selections)}, mode=4")
        self.cmd(f"hide labels, {name}")
        self.cmd(f"color {colour}, {name}")
        self.interactions.add(interaction)

    def save_png(self, name: str) -> None:
        path = Path(name)
        screenshot = cast(PyMOLScreenshot, self.interface)
        return screenshot.save_png(path, dpi=300, ray=1, **screenshot.kwargs)


@dataclass
class PyMOLScreenshot:
    callback: Callable[[str], None]
    kwargs: dict
    is_interactive: bool

    def save_png(self, path: Path, **kwargs: Any) -> None:
        # wait for all commands to be done
        self.callback("cmd.sync(timeout=3)")

        # build command
        if kwargs:
            params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            params = f", {params}"
        else:
            params = ""
        self.callback(f"png {path}{params}")

        # wait for end chunk of PNG to be written
        while not self._is_png_written(path):
            sleep(0.1)

    def _repr_png_(self) -> bytes | None:
        if not self.is_interactive:
            # only generate PNG if using RPC server
            return None

        # write to temp PNG file
        png_path = Path(mktemp(suffix=".png"))
        self.save_png(png_path, **self.kwargs)

        # read data and delete temp file
        data = png_path.read_bytes()
        png_path.unlink(missing_ok=True)
        return data

    @classmethod
    def _is_png_written(cls, png_path: Path) -> bool:
        """Returns whether the PNG has been completely written or not."""
        return (
            png_path.is_file()
            and png_path.stat().st_size > 12
            and struct.unpack("!I4sI", cls._last_n_bytes(png_path, 12))
            # IEND data chunk for PNG files (12 bytes, network-byte order):
            # 4 null bytes for length of data (always no data for this chunk)
            # 4 bytes for name of chunk (IEND for last chunk)
            # (0 bytes for data)
            # CRC-32 checksum for `b"IEND"`
            == (0, b"IEND", 2923585666)
        )

    @staticmethod
    def _last_n_bytes(path: Path, n: int) -> bytes:
        """Reads the last N bytes from a file."""
        with open(path, "rb") as fh:
            fh.seek(-n, os.SEEK_END)
            return fh.read(n)

    def _repr_html_(self) -> str | None:
        png = self._repr_png_()
        if not png:
            return None
        dump = b64encode(png).decode()
        return (
            '<image alt="ProLIF PyMOL screenshot" src="data:image/png;base64, '
            f'{dump}" />'
        )
