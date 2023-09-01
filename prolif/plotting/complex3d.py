"""
Plot interactions in 3D --- :mod:`prolif.plotting.complex3d`
============================================================

.. versionadded:: 2.0.0

.. autoclass:: Complex3D
   :members:

"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, Set, Tuple

import py3Dmol
from rdkit import Chem
from rdkit.Geometry import Point3D

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import separated_interaction_colors
from prolif.utils import get_centroid

if TYPE_CHECKING:
    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP
    from prolif.molecule import Molecule


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
    LIGAND_STYLE : Dict[str, Dict] = {"stick": {"colorscheme": "cyanCarbon"}}
        Style object passed to ``3Dmol.js`` for the ligand.
    RESIDUES_STYLE : Dict[str, Dict] = {"stick": {}}
        Style object passed to ``3Dmol.js`` for the protein residues involved in
        interactions.
    PROTEIN_STYLE : Dict[str, Dict] = {"cartoon": {"style": "edged"}}
        Style object passed to ``3Dmol.js`` for the entire protein.
    PEPTIDE_STYLE : Dict[str, Dict] = "cartoon": {"style": "edged", "colorscheme": "cyanCarbon"}
        Style object passed to ``3Dmol.js`` for the ligand as a peptide if appropriate.
    PEPTIDE_THRESHOLD : int = 2
        Ligands with this number of residues or more will be displayed using
        ``PEPTIDE_STYLE`` in addition to the ``LIGAND_STYLE``.
    LIGAND_DISPLAYED_ATOM : Dict[str, int]
        Which atom should be used to display an atom-to-atom interaction for the ligand.
        Refers to the order defined in the SMARTS pattern used in interaction
        definition. Interactions not specified here use ``0`` by default.
    PROTEIN_DISPLAYED_ATOM : Dict[str, int]
        Same as :attr:`LIGAND_DISPLAYED_ATOM` for the protein.
    LIGAND_RING_INTERACTIONS : Set[str]
        Which interactions should be displayed using the centroid instead of using
        :attr:`LIGAND_DISPLAYED_ATOM` for the ligand.
    PROTEIN_RING_INTERACTIONS : Set[str]
        Which interactions should be displayed using the centroid instead of using
        :attr:`PROTEIN_DISPLAYED_ATOM` for the protein.
    RESIDUE_HOVER_CALLBACK : str
        JavaScript callback executed when hovering a residue involved in an interaction.
    INTERACTION_HOVER_CALLBACK : str
        JavaScript callback executed when hovering an interaction line.
    DISABLE_HOVER_CALLBACK : str
        JavaScript callback executed when the hovering event is finished.
    """

    COLORS: ClassVar[Dict[str, str]] = {**separated_interaction_colors}
    LIGAND_STYLE: ClassVar[Dict] = {"stick": {"colorscheme": "cyanCarbon"}}
    RESIDUES_STYLE: ClassVar[Dict] = {"stick": {}}
    PROTEIN_STYLE: ClassVar[Dict] = {"cartoon": {"style": "edged"}}
    PEPTIDE_STYLE: ClassVar[Dict] = {
        "cartoon": {"style": "edged", "colorscheme": "cyanCarbon"}
    }
    PEPTIDE_THRESHOLD: ClassVar[int] = 5
    LIGAND_DISPLAYED_ATOM = {
        "HBDonor": 1,
        "XBDonor": 1,
    }
    PROTEIN_DISPLAYED_ATOM = {
        "HBAcceptor": 1,
        "XBAcceptor": 1,
    }
    RING_SYSTEMS: ClassVar[Set[str]] = {
        "PiStacking",
        "EdgeToFace",
        "FaceToFace",
    }
    LIGAND_RING_INTERACTIONS: ClassVar[Set[str]] = {*RING_SYSTEMS, "PiCation"}
    PROTEIN_RING_INTERACTIONS: ClassVar[Set[str]] = {*RING_SYSTEMS, "CationPi"}
    RESIDUE_HOVER_CALLBACK: ClassVar[
        str
    ] = """function(atom,viewer) {
        if(!atom.label) {
            atom.label = viewer.addLabel('%s:'+atom.atom+atom.serial,
                {position: atom, backgroundColor: 'mintcream', fontColor:'black'});
        }
    }"""
    INTERACTION_HOVER_CALLBACK: ClassVar[
        str
    ] = """
    function(atom,viewer) {
        if(!atom.label) {
            atom.label = viewer.addLabel(atom.interaction,
                {position: atom, backgroundColor: 'black', fontColor:'white'});
        }
    }"""
    DISABLE_HOVER_CALLBACK: ClassVar[
        str
    ] = """
    function(atom,viewer) { 
        if(atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
        }
    }"""

    def __init__(self, ifp: IFP, lig_mol: Molecule, prot_mol: Molecule) -> None:
        self.ifp = ifp
        self.lig_mol = lig_mol
        self.prot_mol = prot_mol

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
                "Please run the fingerprint analysis before attempting to display results."
            )
        ifp = fp.ifp[frame]
        return cls(ifp, lig_mol, prot_mol)

    @staticmethod
    def get_ring_centroid(mol: Molecule, indices: Tuple[int, ...]) -> Point3D:
        centroid = mol.xyz[list(indices)].mean(axis=0)
        return Point3D(*centroid)

    def display(self, size: Tuple[int, int] = (650, 600), display_all: bool = False):
        """Display as a py3Dmol widget view.

        Parameters
        ----------
        size: Tuple[int, int] = (650, 600)
            The size of the py3Dmol widget view.
        display_all : bool
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Not relevant if ``count=False`` in the ``Fingerprint``
            object.
        """
        v = py3Dmol.view(width=size[0], height=size[1])
        v.removeAllModels()

        models = {}
        mid = -1
        for (lresid, presid), interactions in self.ifp.items():
            lres = self.lig_mol[lresid]
            pres = self.prot_mol[presid]
            # set model ids for reusing later
            for resid, res, style in [
                (lresid, lres, self.LIGAND_STYLE),
                (presid, pres, self.RESIDUES_STYLE),
            ]:
                if resid not in models:
                    mid += 1
                    v.addModel(Chem.MolToMolBlock(res), "sdf")
                    model = v.getModel()
                    model.setStyle({}, style)
                    # add residue label
                    model.setHoverable(
                        {},
                        True,
                        self.RESIDUE_HOVER_CALLBACK % resid,
                        self.DISABLE_HOVER_CALLBACK,
                    )
                    models[resid] = mid
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
                    # get coordinates for both points of the interaction
                    if interaction in self.LIGAND_RING_INTERACTIONS:
                        p1 = self.get_ring_centroid(lres, metadata["indices"]["ligand"])
                    else:
                        p1 = lres.GetConformer().GetAtomPosition(
                            metadata["indices"]["ligand"][
                                self.LIGAND_DISPLAYED_ATOM.get(interaction, 0)
                            ]
                        )
                    if interaction in self.PROTEIN_RING_INTERACTIONS:
                        p2 = self.get_ring_centroid(
                            pres, metadata["indices"]["protein"]
                        )
                    else:
                        p2 = pres.GetConformer().GetAtomPosition(
                            metadata["indices"]["protein"][
                                self.PROTEIN_DISPLAYED_ATOM.get(interaction, 0)
                            ]
                        )
                    # add interaction line
                    v.addCylinder(
                        {
                            "start": dict(x=p1.x, y=p1.y, z=p1.z),
                            "end": dict(x=p2.x, y=p2.y, z=p2.z),
                            "color": self.COLORS.get(interaction, "grey"),
                            "radius": 0.15,
                            "dashed": True,
                            "fromCap": 1,
                            "toCap": 1,
                        }
                    )
                    # add label when hovering the middle of the dashed line by adding a dummy atom
                    c = Point3D(*get_centroid([p1, p2]))
                    modelID = models[lresid]
                    model = v.getModel(modelID)
                    interaction_label = f"{interaction}: {metadata['distance']:.2f}Å"
                    model.addAtoms(
                        [
                            {
                                "elem": "Z",
                                "x": c.x,
                                "y": c.y,
                                "z": c.z,
                                "interaction": interaction_label,
                            }
                        ]
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

        # show protein
        mol = Chem.RemoveAllHs(self.prot_mol)
        pdb = Chem.MolToPDBBlock(mol, flavor=0x20 | 0x10)
        v.addModel(pdb, "pdb")
        model = v.getModel()
        model.setStyle({}, self.PROTEIN_STYLE)

        # do the same for ligand if multiple residues
        if self.lig_mol.n_residues >= self.PEPTIDE_THRESHOLD:
            mol = Chem.RemoveAllHs(self.lig_mol)
            pdb = Chem.MolToPDBBlock(mol, flavor=0x20 | 0x10)
            v.addModel(pdb, "pdb")
            model = v.getModel()
            model.setStyle({}, self.PEPTIDE_STYLE)

        v.zoomTo({"model": list(models.values())})
        return v
