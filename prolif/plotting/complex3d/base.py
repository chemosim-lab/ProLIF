# setup
# show cartoon (protein, and optionally peptide ligand)
# show residue (for ligand and pocket)
# show interaction
# hide nonpolar hydrogens
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from prolif.plotting.utils import separated_interaction_colors

if TYPE_CHECKING:
    from rdkit.Geometry import Point3D

    from prolif.molecule import Molecule
    from prolif.residue import Residue, ResidueId

_RING_SYSTEMS = ("PiStacking", "EdgeToFace", "FaceToFace")

StyleT = TypeVar("StyleT")


@dataclass
class Settings(Generic[StyleT]):
    """
    Attributes
    ----------
    COLORS : dict
        Dictionnary of colors used in the plot for interactions.
    LIGAND_STYLE : dict
        Style object for the ligand.
    RESIDUES_STYLE : dict
        Style object for any residue involved in interactions.
    PROTEIN_STYLE : dict
        Style object for the protein.
    PEPTIDE_STYLE : dict
        Style object for the ligand as a peptide if appropriate.
    PEPTIDE_THRESHOLD : int = 5
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
    BRIDGED_INTERACTIONS : dict[str, str]
        For bridged-interactions such as WaterBridge. The key is the interaction name,
        and the value is the name of the molecule in the metadata indices dictionary.
    """

    LIGAND_STYLE: StyleT
    RESIDUES_STYLE: StyleT
    PROTEIN_STYLE: StyleT
    PEPTIDE_STYLE: StyleT
    COLORS: dict[str, str] = field(
        default_factory=lambda: {**separated_interaction_colors}
    )
    PEPTIDE_THRESHOLD: int = 5
    LIGAND_DISPLAYED_ATOM: dict = field(
        default_factory=lambda: {
            "HBDonor": 1,
            "XBDonor": 1,
        }
    )
    PROTEIN_DISPLAYED_ATOM: dict = field(
        default_factory=lambda: {
            "HBAcceptor": 1,
            "XBAcceptor": 1,
        }
    )
    RING_SYSTEMS: set[str] = field(default_factory=lambda: {*_RING_SYSTEMS})
    LIGAND_RING_INTERACTIONS: set[str] = field(
        default_factory=lambda: {*_RING_SYSTEMS, "PiCation"}
    )
    PROTEIN_RING_INTERACTIONS: set[str] = field(
        default_factory=lambda: {*_RING_SYSTEMS, "CationPi"}
    )
    BRIDGED_INTERACTIONS: dict[str, str] = field(
        default_factory=lambda: {"WaterBridge": "water"}
    )


SettingsT = TypeVar("SettingsT", bound=Settings)
ComponentT = TypeVar("ComponentT")
ModelT = TypeVar("ModelT")


class Backend(Protocol, Generic[SettingsT, ComponentT, ModelT]):
    """
    Protocol for a 3D visualization backend.
    """

    settings: SettingsT
    view: Any = None
    models: dict[ComponentT, ModelT]
    residues: dict["ResidueId", ComponentT]
    _model_count: int

    def __init__(self, settings: SettingsT) -> None:
        """Initialize the backend with the given settings."""
        self.settings = settings

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Setup the backend for a new plot."""
        self.prepare()
        self.clear()

    def prepare(self, *args: Any, **kwargs: Any) -> None:
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
        """Finalize the plot."""

    def load_molecule(self, mol: "Molecule", component: ComponentT, style: Any) -> None:
        """Load a molecule into the view with the given style."""

    def show_residue(
        self, residue: "Residue", component: ComponentT, style: Any
    ) -> None:
        """Show a residue on the plot."""
        self.residues[residue.resid] = component

    def hide_hydrogens(self, component: ComponentT, keep_indices: list[int]) -> None:
        """Hide non-polar hydrogens in the view."""

    def add_interaction(
        self,
        interaction: str,
        distance: float,
        points: tuple["Point3D", "Point3D"],
        residues: tuple["ResidueId", "ResidueId"],
        atoms: tuple[int | tuple[int, ...], int | tuple[int, ...]],
    ) -> None:
        """Add an interaction to the plot."""

    def save_png(self, name: str) -> None:
        """Saves the current state of the 3D viewer to a PNG."""
