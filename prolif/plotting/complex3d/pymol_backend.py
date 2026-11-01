from __future__ import annotations

import atexit
import os
import struct
from base64 import b64encode
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from tempfile import mktemp
from time import sleep
from typing import TYPE_CHECKING, Any, Protocol, cast
from xmlrpc.client import ServerProxy

from rdkit import Chem

from prolif.plotting.complex3d.base import (
    Complex3DBackend,
    Complex3DBaseBackendSettings,
)

if TYPE_CHECKING:
    from prolif.molecule import Molecule
    from prolif.residue import Residue, ResidueId
    from prolif.typeshed import Component


class PyMOLRPCServer(Protocol):
    def ping(self) -> None: ...
    def do(self, command: str) -> None: ...
    def __close(self) -> None: ...


@dataclass
class PyMOLSettings(Complex3DBaseBackendSettings[dict[str, list[str]]]):
    __doc__ = (
        cast(str, Complex3DBaseBackendSettings.__doc__)
        + """\
    group_id: str = "complex"
        Name of the object in which all PyMOL objects will be loaded.
    handler: Callable[[str], None] | None = None
        Function to call for each command sent to PyMOL. If ``None``, the
        :class:`~prolif.plotting.complex3d.pymol_backend.PyMOLRPCServer` will be used to
        send commands to a running PyMOL instance.
    """
    )
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
    handler: Callable[[str], None] | None = None


@cache
def get_rpc_server(port: int = 9123) -> PyMOLRPCServer:
    """Get proxy to the PyMOL RPC server."""
    host = os.environ.get("PYMOL_RPCHOST", "localhost")
    proxy = cast(PyMOLRPCServer, ServerProxy(f"http://{host}:{port}/RPC2"))
    try:
        proxy.ping()
    except ConnectionRefusedError:
        raise RuntimeError(
            "Could not connect to PyMOL RPC server. "
            "Make sure you run PyMOL with the `-R` flag."
        ) from None
    atexit.register(proxy.__close)
    return proxy


_forbidden_table = str.maketrans({".": None, ":": None})


def _strip_forbidden_chars(s: str) -> str:
    return s.translate(_forbidden_table)


@dataclass
class PyMOLScreenshot:
    callback: Callable[[str], None]
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
        self.save_png(png_path)

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
        with path.open("rb") as fh:
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


class PyMOLBackend(
    Complex3DBackend[PyMOLSettings, dict[str, list[str]], str, PyMOLScreenshot],
    settings=PyMOLSettings,
):
    def setup(self) -> None:
        handler = self.settings.handler
        self.cmd = handler or get_rpc_server().do
        self.interface = PyMOLScreenshot(
            callback=self.cmd, is_interactive=handler is None
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
    def _ignore_autozoom(self) -> Iterator[None]:
        """Context manager to ignore the automatic zoom when loading an object."""
        self.cmd("view rdinterface, store")
        yield
        self.cmd("view rdinterface, recall")

    def load_molecule(
        self, mol: "Molecule", component: "Component", style: dict[str, list[str]]
    ) -> None:
        pdb_dump = Chem.MolToPDBBlock(mol, flavor=16 | 32)
        model_id = f"{self.group_id}.{component}"
        with self._ignore_autozoom():
            self.cmd(f"cmd.read_pdbstr({pdb_dump!r}, {model_id!r})")
        self._model_count += 1
        self.models[component] = model_id
        self.apply_style(f"%{model_id}", style)

    def show_residue(
        self,
        residue: "Residue",
        component: "Component",
        style: dict[str, list[str]],
    ) -> None:
        super().show_residue(residue, component, style)
        model_id = self.models[component]
        resid = residue.resid
        selection = (
            f"%{model_id} and chain {resid.chain} "
            f"and resid {resid.number} and resname {resid.name}"
        )
        if color_override := self.settings.color_overrides.get(resid):
            resid_style = deepcopy(style)
            for key in resid_style:
                resid_style[key].append(f"color {color_override}, {{}}")
        else:
            resid_style = style
        self.apply_style(selection, resid_style)

    def apply_style(self, selection: str, style: dict[str, list[str]]) -> None:
        for representation, extras in style.items():
            self.cmd(f"show {representation}, {selection}")
            if extras:
                for extra in extras:
                    self.cmd(extra.format(selection))

    def hide_hydrogens(self, component: "Component", keep_indices: list[int]) -> None:
        model_id = self.models[component]
        subset = " or ".join([f"rank {i}" for i in keep_indices])
        selection = f"%{model_id} and elem H"
        if subset:
            selection += f" and not ({subset})"
        self.cmd(f"hide ({selection})")

    def add_interaction(
        self,
        interaction: str,
        distance: float,  # noqa: ARG002
        components: tuple["Component", "Component"],
        residues: tuple["ResidueId", "ResidueId"],
        atoms: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    ) -> None:
        lresid, presid = residues
        colour = (
            self.settings.colors.get(interaction, "#dedede").upper().replace("#", "0x")
        )
        selections = []
        for component, indices in zip(components, atoms, strict=True):
            if isinstance(indices, int):
                sel = f"%{self.group_id}.{component} and (rank {indices})"
            else:
                ring_indices = " or rank ".join(map(str, indices))
                sel = f"%{self.group_id}.{component} and (rank {ring_indices})"
            selections.append(sel)

        resids = _strip_forbidden_chars(f"{lresid}_{presid}")
        atom_ids = "_".join(
            ["".join(map(str, (a,) if isinstance(a, int) else a)) for a in atoms]
        )
        name = f"{self.group_id}.interactions.{interaction}.{resids}.{atom_ids}"
        self.cmd(f"distance {name}, {', '.join(selections)}, mode=4")
        self.cmd(f"hide labels, {name}")
        self.cmd(f"color {colour}, {name}")
        self.interactions.add(interaction)

    def save_png(self, path: str) -> None:
        output = Path(path)
        return self.interface.save_png(output, dpi=300, ray=1)
