from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

from prolif.plotting.utils import separated_interaction_colors
from prolif.residue import ResidueId

if TYPE_CHECKING:
    from prolif.molecule import Molecule
    from prolif.residue import Residue

Component: TypeAlias = Literal["ligand", "protein", "water"]

_RING_SYSTEMS = ("PiStacking", "EdgeToFace", "FaceToFace")
_COMPLEX3D_BACKENDS: dict[
    str | type["Complex3DBaseBackendSettings"], "Complex3DBackendSpec"
] = {}


class Complex3DBackendSpec(NamedTuple):
    name: str
    backend: type["Complex3DBackend"]
    settings: type["Complex3DBaseBackendSettings"]


def get_backend(backend: "str | Complex3DBaseBackendSettings") -> Complex3DBackendSpec:
    """Returns the :class:`Complex3DBackendSpec` matching with ``backend``."""
    query = backend if isinstance(backend, str) else type(backend)
    spec = _COMPLEX3D_BACKENDS.get(query)
    if spec is None:
        raise ValueError(f"Unknown backend {backend!r}")
    return spec


StyleT = TypeVar("StyleT")
StyleT_contra = TypeVar("StyleT_contra", contravariant=True)


@dataclass
class Complex3DBaseBackendSettings(Generic[StyleT]):
    """
    Attributes
    ----------
    ligand_style : StyleT
        Style object for the ligand.
    residues_style : StyleT
        Style object for any residue involved in interactions.
    protein_style : StyleT
        Style object for the protein.
    peptide_style : StyleT
        Style object for the ligand as a peptide if appropriate.
    size : tuple[int, int]
        Size of the 3D viewer in pixels.
    colors : dict[str, str]
        Dictionnary of colors used in the plot for interactions.
    color_overrides : dict[ResidueId, str]
        Dictionnary of colors used to override the default color for specific residues.
        Note that this needs to be a valid color for the backend.
    peptide_threshold : int = 5
        Ligands with this number of residues or more will be displayed using
        :attr:`peptide_style` in addition to the :attr:`ligand_style`.
    ligand_displayed_atom : dict[str, int]
        Which atom should be used to display an atom-to-atom interaction for the ligand.
        Refers to the order defined in the SMARTS pattern used in interaction
        definition. Interactions not specified here use ``0`` by default.
    protein_displayed_atom : dict[str, int]
        Same as :attr:`ligand_displayed_atom` for the protein.
    ligand_ring_interactions : set[str]
        Which interactions should be displayed using the centroid instead of using
        :attr:`ligand_displayed_atom` for the ligand.
    protein_ring_interactions : set[str]
        Which interactions should be displayed using the centroid instead of using
        :attr:`protein_displayed_atom` for the protein.
    bridged_interactions : dict[str, str]
        For bridged-interactions such as WaterBridge. The key is the interaction name,
        and the value is the name of the molecule in the metadata indices dictionary.
    """

    ligand_style: StyleT
    residues_style: StyleT
    protein_style: StyleT
    peptide_style: StyleT
    size: tuple[int, int] = (650, 600)
    colors: dict[str, str] = field(
        default_factory=lambda: {**separated_interaction_colors}
    )
    color_overrides: dict[ResidueId, str] = field(default_factory=dict)
    peptide_threshold: int = 5
    ligand_displayed_atoms: dict = field(
        default_factory=lambda: {
            "HBDonor": 1,
            "XBDonor": 1,
        }
    )
    protein_displayed_atoms: dict = field(
        default_factory=lambda: {
            "HBAcceptor": 1,
            "XBAcceptor": 1,
        }
    )
    ring_systems: set[str] = field(default_factory=lambda: {*_RING_SYSTEMS})
    ligand_ring_interactions: set[str] = field(
        default_factory=lambda: {*_RING_SYSTEMS, "PiCation"}
    )
    protein_ring_interactions: set[str] = field(
        default_factory=lambda: {*_RING_SYSTEMS, "CationPi"}
    )
    bridged_interactions: dict[str, str] = field(
        default_factory=lambda: {"WaterBridge": "water"}
    )

    @contextmanager
    def overrides(self, **kwargs: Any) -> Iterator[None]:
        """
        Context manager to temporarily override settings.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to the attributes of the settings.
        """
        original_values = {key: getattr(self, key) for key in kwargs}
        try:
            for key, value in kwargs.items():
                setattr(self, key, value)
            yield
        finally:
            for key, value in original_values.items():
                setattr(self, key, value)


BackendSettingsT = TypeVar("BackendSettingsT", bound=Complex3DBaseBackendSettings)
ModelT = TypeVar("ModelT")
InterfaceT = TypeVar("InterfaceT")


@runtime_checkable
class Complex3DBackend(
    Protocol, Generic[BackendSettingsT, StyleT_contra, ModelT, InterfaceT]
):
    """
    Protocol for a 3D visualization backend.
    """

    settings: BackendSettingsT
    interface: InterfaceT
    models: dict[Component, ModelT]
    residues: dict["ResidueId", Component]
    _model_count: int

    def __init_subclass__(
        cls,
        settings: type[Complex3DBaseBackendSettings] | None,
        backend: str | None = None,
    ) -> None:
        """Registers new Complex3D backends.

        If no ``settings`` is specified, the class is not registered (e.g. for base
        classes).
        """
        if settings is not None:
            name = backend or cls.__name__.lower().replace("backend", "")
            spec = Complex3DBackendSpec(name=name, backend=cls, settings=settings)
            _COMPLEX3D_BACKENDS[name] = spec
            _COMPLEX3D_BACKENDS[settings] = spec
        return super().__init_subclass__()

    def __init__(self, settings: BackendSettingsT) -> None:
        """Initialize the backend with the given settings."""
        self.settings = settings

    def setup(self) -> None:
        """Setup the backend for a new plot."""
        self.prepare()
        self.clear()

    def prepare(self) -> None:
        """
        Prepare the backend for plotting new elements. This method must have defaults
        for all parameters.
        """
        self.models = {}
        self._model_count = 0
        self.residues = {}

    def clear(self) -> None:
        """Remove all elements from the plot."""

    def finalize(self) -> None:
        """Finalize the plot, e.g. zoom to the ligand."""

    def load_molecule(
        self, mol: "Molecule", component: Component, style: StyleT_contra
    ) -> None:
        """Load a molecule into the view with the given style."""

    def show_residue(
        self,
        residue: "Residue",
        component: Component,
        style: StyleT_contra,  # noqa: ARG002
    ) -> None:
        """Show a residue on the plot."""
        self.residues[residue.resid] = component

    def hide_hydrogens(self, component: Component, keep_indices: list[int]) -> None:
        """Hide non-polar hydrogens in the view."""

    def add_interaction(
        self,
        interaction: str,
        distance: float,
        components: tuple[Component, Component],
        residues: tuple["ResidueId", "ResidueId"],
        atoms: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    ) -> None:
        """Add an interaction to the plot."""

    def save_png(self, path: str) -> None:
        """Saves the current state of the 3D viewer to a PNG."""


@dataclass
class Complex3DBaseGridBackendSettings(Complex3DBaseBackendSettings[StyleT]):
    __doc__ = (
        cast(str, Complex3DBaseBackendSettings.__doc__)
        + """\
    grid_size : tuple[int, int]
        Size of the grid in number of rows and columns.
    grid_color_unique: str | None
        Which color to use for residues that have interactions that are found in one
        complex but not the other. Use ``None`` to disable the color override.
    """
    )

    grid_size: tuple[int, int] = (900, 600)
    grid_color_unique: str | None = None


@runtime_checkable
class Complex3DGridBackend(
    Complex3DBackend[BackendSettingsT, StyleT_contra, ModelT, InterfaceT],
    Protocol,
    settings=None,
):
    def grid_setup(self, n_rows: int, n_columns: int) -> None:
        """Setup the backend for a new grid plot."""

    def grid_prepare(self, row: int, column: int) -> None:
        """Prepare the backend for plotting new elements in a grid."""
