"""
Plot a Ligand Interaction Network --- :mod:`prolif.plotting.network`
====================================================================

.. versionadded:: 0.3.2

.. versionchanged:: 2.0.0
    Replaced ``LigNetwork.from_ifp`` with ``LigNetwork.from_fingerprint`` which works
    without requiring a dataframe with atom indices.

.. autoclass:: LigNetwork
   :members:

"""

import json
import operator
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO, Union, cast
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor

from prolif.exceptions import RunRequiredError
from prolif.ifp import IFP
from prolif.plotting.utils import grouped_interaction_colors, metadata_iterator
from prolif.residue import ResidueId
from prolif.utils import requires

try:
    from IPython.display import Javascript, display
except ModuleNotFoundError:
    pass
else:
    warnings.filterwarnings(
        "ignore",
        "Consider using IPython.display.IFrame instead",  # pragma: no cover
    )

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP


class LigNetwork:
    """Creates a ligand interaction diagram

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a 4-level index (ligand, protein, interaction, atoms)
        and ``weight`` and ``distance`` columns for values
    lig_mol : rdkit.Chem.rdChem.Mol
        Ligand molecule
    use_coordinates : bool
        If ``True``, uses the coordinates of the molecule directly, otherwise generates
        2D coordinates from scratch. See also ``flatten_coordinates``.
    flatten_coordinates : bool
        If this is ``True`` and ``use_coordinates=True``, generates 2D coordinates that
        are constrained to fit the 3D conformation of the ligand as best as possible.
    kekulize : bool
        Kekulize the ligand
    molsize : int
        Multiply the coordinates by this number to create a bigger and
        more readable depiction
    rotation : float
        Rotate the structure on the XY plane
    carbon : float
        Size of the carbon atom dots on the depiction. Use `0` to hide the
        carbon dots

    Attributes
    ----------
    COLORS : dict
        Dictionnary of colors used in the diagram. Subdivided in several
        dictionaries:

        - "interactions": mapping between interactions types and colors
        - "atoms": mapping between atom symbol and colors
        - "residues": mapping between residues types and colors

    RESIDUE_TYPES : dict
        Mapping between residue names (3 letter code) and types. The types
        are then used to define how each residue should be colored.

    Notes
    -----
    You can customize the diagram by tweaking :attr:`LigNetwork.COLORS` and
    :attr:`LigNetwork.RESIDUE_TYPES` by adding or modifying the
    dictionaries inplace.

    .. versionchanged:: 2.0.0
        Replaced ``LigNetwork.from_ifp`` with ``LigNetwork.from_fingerprint`` which
        works without requiring a dataframe with atom indices. Replaced ``match3D``
        parameter with ``use_coordinates`` and ``flatten_coordinates`` to give users
        more control and allow them to provide their own 2D coordinates. Added support
        for displaying peptides as the "ligand". Changed the default color for
        VanDerWaals.

    .. versionchanged:: 2.1.0
        Added the ``show_interaction_data`` argument and exposed the ``fontsize`` in
        ``display``.
    """

    COLORS: ClassVar = {
        "interactions": {**grouped_interaction_colors},
        "atoms": {
            "C": "black",
            "N": "blue",
            "O": "red",
            "S": "#dece1b",
            "P": "orange",
            "F": "lime",
            "Cl": "lime",
            "Br": "lime",
            "I": "lime",
        },
        "residues": {
            "Aliphatic": "#59e382",
            "Aromatic": "#b559e3",
            "Acidic": "#e35959",
            "Basic": "#5979e3",
            "Polar": "#59bee3",
            "Sulfur": "#e3ce59",
            "Water": "#323aa8",
        },
    }
    RESIDUE_TYPES: ClassVar = {
        "ALA": "Aliphatic",
        "GLY": "Aliphatic",
        "ILE": "Aliphatic",
        "LEU": "Aliphatic",
        "PRO": "Aliphatic",
        "VAL": "Aliphatic",
        "PHE": "Aromatic",
        "TRP": "Aromatic",
        "TYR": "Aromatic",
        "ASP": "Acidic",
        "GLU": "Acidic",
        "ARG": "Basic",
        "HIS": "Basic",
        "HID": "Basic",
        "HIE": "Basic",
        "HIP": "Basic",
        "HSD": "Basic",
        "HSE": "Basic",
        "HSP": "Basic",
        "LYS": "Basic",
        "SER": "Polar",
        "THR": "Polar",
        "ASN": "Polar",
        "GLN": "Polar",
        "CYS": "Sulfur",
        "CYM": "Sulfur",
        "CYX": "Sulfur",
        "MET": "Sulfur",
        "WAT": "Water",
        "SOL": "Water",
        "H2O": "Water",
        "HOH": "Water",
        "OH2": "Water",
        "HHO": "Water",
        "OHH": "Water",
        "TIP": "Water",
        "T3P": "Water",
        "T4P": "Water",
        "T5P": "Water",
        "TIP2": "Water",
        "TIP3": "Water",
        "TIP4": "Water",
    }
    _FONTCOLORS: ClassVar = {
        "Water": "white",
    }
    _LIG_PI_INTERACTIONS: ClassVar = [
        "EdgeToFace",
        "FaceToFace",
        "PiStacking",
        "PiCation",
    ]
    _BRIDGED_INTERACTIONS: ClassVar[dict[str, str]] = {"WaterBridge": "water_residues"}
    _DISPLAYED_ATOM: ClassVar = {  # index 0 in indices tuple by default
        "HBDonor": 1,
        "XBDonor": 1,
    }

    _JS_FILE = Path(__file__).parent / "network.js"
    _HTML_FILE = Path(__file__).parent / "network.html"
    _CSS_FILE = Path(__file__).parent / "network.css"

    _HTML_TEMPLATE = _HTML_FILE.read_text()
    _JS_TEMPLATE = _JS_FILE.read_text()
    _CSS_TEMPLATE = _CSS_FILE.read_text()

    def __init__(
        self,
        df: pd.DataFrame,
        lig_mol: Chem.Mol,
        use_coordinates: bool = False,
        flatten_coordinates: bool = True,
        kekulize: bool = False,
        molsize: int = 35,
        rotation: float = 0,
        carbon: float = 0.16,
    ) -> None:
        self.df = df
        self._interacting_atoms: set[int] = {
            atom for atoms in df.index.get_level_values("atoms") for atom in atoms
        }
        mol = deepcopy(lig_mol)
        if kekulize:
            Chem.Kekulize(mol)
        if use_coordinates:
            if flatten_coordinates:
                rdDepictor.GenerateDepictionMatching3DStructure(mol, lig_mol)
        else:
            rdDepictor.Compute2DCoords(mol, clearConfs=True)
        xyz: "NDArray[np.float64]" = mol.GetConformer().GetPositions()
        if rotation:
            theta = np.radians(rotation)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, s], [-s, c]])
            xy, z = xyz[:, :2], xyz[:, 2:3]
            center = xy.mean(axis=0)
            xy = ((xy - center) @ R.T) + center
            xyz = np.concatenate([xy, z], axis=1)
        if carbon:
            self._carbon: dict[str, Any] = {
                "label": " ",
                "shape": "dot",
                "color": self.COLORS["atoms"]["C"],
                "size": molsize * carbon,
            }
        else:
            self._carbon = {"label": " ", "shape": "text"}
        self.xyz = molsize * xyz
        self.mol = mol
        self._multiplier = molsize
        self.options: dict[str, Any] = {}
        self._max_interaction_width = 6
        self._avoidOverlap = 0.8
        self._springConstant = 0.1
        self._bond_color = "black"
        self._default_atom_color = "grey"
        self._default_residue_color = "#dbdbdb"
        self._default_interaction_color = "#dbdbdb"
        self._non_single_bond_spacing = 0.06
        self._dash = [10]
        self._edge_title_formatter = "{interaction}: {distance:.2f}Å"
        self._edge_label_formatter = "{weight_pct:.0f}%"
        # regroup interactions of the same color
        temp = defaultdict(list)
        interactions = set(df.index.get_level_values("interaction").unique())
        for interaction in interactions:
            color = self.COLORS["interactions"].get(
                interaction,
                self._default_interaction_color,
            )
            temp[color].append(interaction)
        self._interaction_types = {
            interaction: "/".join(interaction_group)
            for interaction_group in temp.values()
            for interaction in interaction_group
        }
        # ID for saving to PNG with JS
        self.uuid = uuid4().hex
        self._iframe: str | None = None

    @classmethod
    def from_fingerprint(
        cls,
        fp: "Fingerprint",
        ligand_mol: Chem.Mol,
        kind: Literal["aggregate", "frame"] = "aggregate",
        frame: int = 0,
        display_all: bool = False,
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> "LigNetwork":
        """Helper method to create a ligand interaction diagram from a
        :class:`~prolif.fingerprint.Fingerprint` object.

        Notes
        -----
        Two kinds of diagrams can be rendered: either for a designated frame or
        by aggregating the results on the whole IFP and optionnally discarding
        interactions that occur less frequently than a threshold. In the latter
        case (aggregate), only the group of atoms most frequently involved in
        each interaction is used to draw the edge.

        Parameters
        ----------
        fp : prolif.fingerprint.Fingerprint
            The fingerprint object already executed using one of the ``run`` or
            ``run_from_iterable`` methods.
        lig : rdkit.Chem.rdChem.Mol
            Ligand molecule
        kind : str
            One of ``"aggregate"`` or ``"frame"``
        frame : int
            Frame number (see :attr:`~prolif.fingerprint.Fingerprint.ifp`). Only
            applicable for ``kind="frame"``
        display_all : bool
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Only applicable for ``kind="frame"``. Not relevant if
            ``count=False`` in the ``Fingerprint`` object.
        threshold : float
            Frequency threshold, between 0 and 1. Only applicable for
            ``kind="aggregate"``
        kwargs : object
            Other arguments passed to the :class:`LigNetwork` class


        .. versionchanged:: 2.0.0
            Added the ``display_all`` parameter.
        """
        if not hasattr(fp, "ifp"):
            raise RunRequiredError(
                "Please run the fingerprint analysis before attempting to display"
                " results.",
            )
        if kind == "frame":
            df = cls._make_frame_df_from_fp(fp, frame=frame, display_all=display_all)
            return cls(df, ligand_mol, **kwargs)
        if kind == "aggregate":
            df = cls._make_agg_df_from_fp(fp, threshold=threshold)
            return cls(df, ligand_mol, **kwargs)
        raise ValueError(f'{kind!r} must be "aggregate" or "frame"')

    @classmethod
    def _get_records(cls, ifp: "IFP", all_metadata: bool) -> list[dict[str, Any]]:
        records = []
        for (lig_resid, prot_resid), int_data in ifp.items():
            for int_name, metadata_tuple in int_data.items():
                is_bridged_interaction = cls._BRIDGED_INTERACTIONS.get(int_name, None)
                for metadata in metadata_iterator(metadata_tuple, all_metadata):
                    if is_bridged_interaction:
                        distances = [d for d in metadata if d.startswith("distance_")]
                        for distlabel in distances:
                            _, src, dest = distlabel.split("_")
                            if src == "ligand":
                                components = "ligand_water"
                                src = str(lig_resid)
                                atoms = metadata["parent_indices"]["ligand"]
                            elif dest == "protein":
                                components = "water_protein"
                                dest = str(prot_resid)
                                atoms = ()
                            else:
                                components = "water_water"
                                atoms = ()
                            records.append(
                                {
                                    "ligand": src,
                                    "protein": dest,
                                    "interaction": int_name,
                                    "components": components,
                                    "atoms": atoms,
                                    "distance": metadata[distlabel],
                                }
                            )
                    else:
                        records.append(
                            {
                                "ligand": str(lig_resid),
                                "protein": str(prot_resid),
                                "interaction": int_name,
                                "components": "ligand_protein",
                                "atoms": metadata["parent_indices"]["ligand"],
                                "distance": metadata.get("distance", 0),
                            }
                        )
        return records

    @classmethod
    def _make_agg_df_from_fp(
        cls, fp: "Fingerprint", threshold: float = 0.3
    ) -> pd.DataFrame:
        data = []
        for ifp in fp.ifp.values():
            data.extend(cls._get_records(ifp, all_metadata=False))
        df = pd.DataFrame(data)
        # add weight for each atoms, and average distance
        df["weight"] = 1
        df = df.groupby(["ligand", "protein", "interaction", "atoms"]).agg(
            weight=("weight", "sum"),
            distance=("distance", "mean"),
            components=("components", "first"),
        )
        df["weight"] /= len(fp.ifp)
        # merge different ligand atoms of the same residue/interaction group before
        # applying the threshold
        df = df.join(
            df.groupby(level=["ligand", "protein", "interaction"]).agg(
                weight_total=("weight", "sum"),
            ),
        )
        # threshold and keep most occuring ligand atom
        return (
            df[df["weight_total"] >= threshold]
            .drop(columns="weight_total")
            .sort_values("weight", ascending=False)
            .groupby(level=["ligand", "protein", "interaction"])
            .head(1)
            .sort_index()
        )

    @classmethod
    def _make_frame_df_from_fp(
        cls, fp: "Fingerprint", frame: int = 0, display_all: bool = False
    ) -> pd.DataFrame:
        ifp = fp.ifp[frame]
        data = cls._get_records(ifp, all_metadata=display_all)
        df = pd.DataFrame(data)
        df["weight"] = 1
        return df.set_index(["ligand", "protein", "interaction", "atoms"]).reindex(
            columns=["weight", "distance", "components"],
        )

    def _make_carbon(self) -> dict[str, Any]:
        return deepcopy(self._carbon)

    def _make_lig_node(self, atom: Chem.Atom) -> None:
        """Prepare ligand atoms"""
        idx = atom.GetIdx()
        elem = atom.GetSymbol()
        if elem == "H" and idx not in self._interacting_atoms:
            self.exclude.append(idx)
            return
        charge = atom.GetFormalCharge()
        if charge != 0:
            displayed_charge = "{}{}".format(
                "" if abs(charge) == 1 else str(charge),
                "+" if charge > 0 else "-",
            )
            label = f"{elem}{displayed_charge}"
            shape = "ellipse"
        else:
            label = elem
            shape = "circle"
        if elem == "C":
            node = self._make_carbon()
        else:
            node = {
                "label": label,
                "shape": shape,
                "color": "white",
                "font": {
                    "color": self.COLORS["atoms"].get(elem, self._default_atom_color),
                },
            }
        node.update(
            {
                "id": idx,
                "x": float(self.xyz[idx, 0]),
                "y": float(self.xyz[idx, 1]),
                "fixed": True,
                "group": "ligand",
                "borderWidth": 0,
            },
        )
        self._nodes[idx] = node

    def to_networkx_graph_object(self, include_ligand_bonds: bool = True) -> nx.Graph:
        """Export the interaction data directly as a NetworkX graph object.

        Parameters
        ----------
        include_ligand_bonds : bool, default=True
        Whether to include bonds between ligand atoms in the graph

        Returns
        -------
        nx.Graph
            A NetworkX graph representation of the ligand-protein interaction network.
        """
        G: nx.Graph = nx.Graph()

        # 1. Add protein residue nodes
        protein_residues = set(self.df.index.get_level_values("protein"))
        for protein_residue in protein_residues:
            resname = ResidueId.from_string(protein_residue).name
            restype = self.RESIDUE_TYPES.get(resname)
            G.add_node(
                protein_residue,
                node_type="protein",
                residue_type=restype,
                label=protein_residue,
            )

        # 2. Add ligand atom nodes
        if include_ligand_bonds:
            for atom in self.mol.GetAtoms():
                idx = atom.GetIdx()
                G.add_node(
                    idx,
                    node_type="ligand",
                    symbol=atom.GetSymbol(),
                    charge=atom.GetFormalCharge(),
                    coords=(
                        float(self.xyz[idx, 0]),
                        float(self.xyz[idx, 1]),
                        float(self.xyz[idx, 2]),
                    ),
                    label=atom.GetSymbol(),
                )

            # Add ligand bonds
            for bond in self.mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                G.add_edge(
                    begin_idx,
                    end_idx,
                    edge_type="bond",
                    bond_type=bond.GetBondType(),
                    bond_order=bond.GetBondTypeAsDouble(),
                )

        # 3. Add interaction edges
        for idx, row in self.df.iterrows():
            # Explicitly extract and type the index elements
            lig_res: str = idx[0]
            prot_res: str = idx[1]
            interaction: str = idx[2]
            lig_indices: tuple[int, ...] = idx[3]

            # Extract row values
            weight: float = float(row["weight"])
            distance: float = float(row["distance"])
            components: str = row["components"]

            # For interactions involving multiple ligand atoms, create edges for each
            for lig_atom_idx in lig_indices:
                if not include_ligand_bonds and lig_atom_idx not in G:
                    # If we're not including full ligand structure, add atom nodes as needed
                    atom = self.mol.GetAtomWithIdx(lig_atom_idx)
                    G.add_node(
                        lig_atom_idx,
                        node_type="ligand",
                        symbol=atom.GetSymbol(),
                        charge=atom.GetFormalCharge(),
                        coords=(
                            float(self.xyz[lig_atom_idx, 0]),
                            float(self.xyz[lig_atom_idx, 1]),
                            float(self.xyz[lig_atom_idx, 2]),
                        ),
                        label=atom.GetSymbol(),
                    )

                # Add the interaction edge
                G.add_edge(
                    lig_atom_idx,
                    prot_res,
                    edge_type="interaction",
                    interaction_type=interaction,
                    weight=weight,
                    distance=distance,
                    components=components,
                )

        return G

    def _make_lig_edge(self, bond: Chem.Bond) -> None:
        """Prepare ligand bonds"""
        idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        if any(i in self.exclude for i in idx):
            return
        btype = bond.GetBondTypeAsDouble()
        if btype == 1:
            self.edges.append(
                {
                    "from": idx[0],
                    "to": idx[1],
                    "color": self._bond_color,
                    "physics": False,
                    "group": "ligand",
                    "width": 4,
                },
            )
        else:
            self._make_non_single_bond(idx, btype)

    def _make_non_single_bond(self, ids: list[int], btype: float) -> None:
        """Prepare double, triple and aromatic bonds"""
        xyz = self.xyz[ids]
        d = xyz[1, :2] - xyz[0, :2]
        length = np.sqrt((d**2).sum())
        u = d / length
        p = np.array([-u[1], u[0]])
        nodes = []
        dist = self._non_single_bond_spacing * self._multiplier * np.ceil(btype)
        dashes = False if btype in {2, 3} else self._dash
        for perp in (p, -p):
            for point in xyz:
                xy = point[:2] + perp * dist
                id_ = hash(xy.tobytes())
                nodes.append(id_)
                self._nodes[id_] = {
                    "id": id_,
                    "x": xy[0],
                    "y": xy[1],
                    "shape": "text",
                    "label": " ",
                    "fixed": True,
                    "physics": False,
                }
        l1, l2, r1, r2 = nodes
        self.edges.extend(
            [
                {
                    "from": l1,
                    "to": l2,
                    "color": self._bond_color,
                    "physics": False,
                    "dashes": dashes,
                    "group": "ligand",
                    "width": 4,
                },
                {
                    "from": r1,
                    "to": r2,
                    "color": self._bond_color,
                    "physics": False,
                    "dashes": dashes,
                    "group": "ligand",
                    "width": 4,
                },
            ],
        )
        if btype == 3:
            self.edges.append(
                {
                    "from": ids[0],
                    "to": ids[1],
                    "color": self._bond_color,
                    "physics": False,
                    "group": "ligand",
                    "width": 4,
                },
            )

    def _make_interactions(self, mass: int = 2) -> None:
        """Prepare lig-prot interactions"""
        restypes: dict[str, str | None] = {}
        lig_prot_df = self.df[self.df["components"] == "ligand_protein"]
        prot_and_waters: set[str] = (
            set(self.df.index.get_level_values("protein"))
            .union(self.df.index.get_level_values("ligand"))
            .difference(lig_prot_df.index.get_level_values("ligand"))
        )
        for prot_res in prot_and_waters:
            resname = ResidueId.from_string(prot_res).name
            restype = self.RESIDUE_TYPES.get(resname)
            restypes[prot_res] = restype
            color = self.COLORS["residues"].get(restype, self._default_residue_color)
            node = {
                "id": prot_res,
                "label": prot_res,
                "color": color,
                "font": {"color": self._FONTCOLORS.get(restype, "black")},
                "shape": "box",
                "borderWidth": 0,
                "physics": True,
                "mass": mass,
                "group": "protein",
                "residue_type": restype,
            }
            self._nodes[prot_res] = node
        for (lig_res, prot_res, interaction, lig_indices), (
            weight,
            distance,
            components,
        ) in cast(
            Iterable[
                tuple[tuple[str, str, str, tuple[int, ...]], tuple[float, float, str]]
            ],
            self.df.iterrows(),
        ):
            if components.startswith("ligand"):
                if interaction in self._LIG_PI_INTERACTIONS:
                    centroid = self._get_ring_centroid(lig_indices)
                    origin = f"centroid({lig_res}, {prot_res}, {interaction})"
                    self._nodes[origin] = {
                        "id": origin,
                        "x": centroid[0],
                        "y": centroid[1],
                        "shape": "text",
                        "label": " ",
                        "fixed": True,
                        "physics": False,
                        "group": "ligand",
                    }
                else:
                    i = self._DISPLAYED_ATOM.get(interaction, 0)
                    origin = lig_indices[i]
            else:
                # water-water or water-protein
                origin = lig_res
            int_data = {
                "interaction": interaction,
                "distance": distance,
                "weight": weight,
                "weight_pct": weight * 100,
            }
            edge = {
                "from": origin,
                "to": prot_res,
                "title": self._edge_title_formatter.format_map(int_data),
                "interaction_type": self._interaction_types.get(
                    interaction,
                    interaction,
                ),
                "color": self.COLORS["interactions"].get(
                    interaction,
                    self._default_interaction_color,
                ),
                "smooth": {"type": "cubicBezier", "roundness": 0.2},
                "dashes": [10],
                "width": weight * self._max_interaction_width,
                "group": "interaction",
                "components": components,
            }
            if self.show_interaction_data:
                edge["label"] = self._edge_label_formatter.format_map(int_data)
                edge["font"] = self._edge_label_font
            self.edges.append(edge)

    def _get_ring_centroid(self, indices: tuple[int, ...]) -> "NDArray[np.float64]":
        """Find ring centroid coordinates using the indices of the ring atoms"""
        return self.xyz[list(indices)].mean(axis=0)  # type: ignore[no-any-return]

    def _patch_hydrogens(self) -> None:
        """Patch hydrogens on heteroatoms

        Hydrogen atoms that aren't part of any interaction have been hidden at
        this stage, but they should be added to the label of the heteroatom for
        clarity
        """
        to_patch: defaultdict[int, int] = defaultdict(int)
        for idx in self.exclude:
            h = self.mol.GetAtomWithIdx(idx)
            atom: Chem.Atom = h.GetNeighbors()[0]
            if atom.GetSymbol() != "C":
                to_patch[atom.GetIdx()] += 1
        for idx, nH in to_patch.items():
            node = self._nodes[idx]
            h_str = "H" if nH == 1 else f"H{nH}"
            label = re.sub(r"(\w+)(.*)", rf"\1{h_str}\2", node["label"])
            node["label"] = label
            node["shape"] = "ellipse"

    def _make_graph_data(self) -> None:
        """Prepares the nodes and edges"""
        self.exclude: list[int] = []
        self._nodes: dict[int | str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        # show residues
        self._make_interactions()
        # show ligand
        for atom in self.mol.GetAtoms():
            self._make_lig_node(atom)
        for bond in self.mol.GetBonds():
            self._make_lig_edge(bond)
        self._patch_hydrogens()
        self.nodes = list(self._nodes.values())

    def _get_js(
        self,
        width: str = "100%",
        height: str = "500px",
        div_id: str = "mynetwork",
        fontsize: int = 20,
        show_interaction_data: bool = False,
    ) -> dict[str, Any]:
        """Returns the JavaScript code to draw the network"""
        self.width = width
        self.height = height
        self.show_interaction_data = show_interaction_data
        self._edge_label_font = {"size": fontsize}
        self._make_graph_data()
        options = {
            "width": width,
            "height": height,
            "nodes": {
                "font": {"size": fontsize},
            },
            "physics": {
                "barnesHut": {
                    "avoidOverlap": self._avoidOverlap,
                    "springConstant": self._springConstant,
                },
            },
        }
        options.update(self.options)

        # get the legend buttons
        buttons = self._get_legend_buttons()

        return {
            "div_id": div_id,
            "nodes": json.dumps(self.nodes),
            "edges": json.dumps(self.edges),
            "options": json.dumps(options),
            "js_file_content": self._JS_TEMPLATE,
            "css_file_content": self._CSS_TEMPLATE,
            "buttons": json.dumps(buttons),
        }

    def _get_html(self, **kwargs: Any) -> str:
        """Returns the HTML code to draw the network"""
        js_data = self._get_js(**kwargs)
        return self._HTML_TEMPLATE % js_data

    def _get_legend_buttons(self, height: str = "90px") -> list[dict[str, Any]]:
        """Prepare the legend buttons data"""
        available = {}
        buttons = []
        map_color_restype = {c: t for t, c in self.COLORS["residues"].items()}
        map_color_interactions = {
            self.COLORS["interactions"].get(i, self._default_interaction_color): t
            for i, t in self._interaction_types.items()
        }
        # residues
        for node in self.nodes:
            if node.get("group", "") == "protein":
                color = node["color"]
                available[color] = map_color_restype.get(color, "Unknown")
        available = dict(sorted(available.items(), key=operator.itemgetter(1)))
        for i, (color, restype) in enumerate(available.items()):
            buttons.append(
                {
                    "index": i,
                    "label": restype,
                    "color": color,
                    "fontcolor": self._FONTCOLORS.get(restype, "black"),
                    "group": "residues",
                },
            )
        # interactions
        available.clear()
        for edge in self.edges:
            if edge.get("group", "") == "interaction":
                color = edge["color"]
                available[color] = map_color_interactions[color]
        available = dict(sorted(available.items(), key=operator.itemgetter(1)))
        for i, (color, interaction) in enumerate(available.items()):
            buttons.append(
                {
                    "index": i,
                    "label": interaction,
                    "color": color,
                    "fontcolor": "black",
                    "group": "interactions",
                },
            )

        # update height for legend
        if all("px" in h for h in [self.height, height]):
            h1 = int(re.findall(r"(\d+)\w+", self.height)[0])
            h2 = int(re.findall(r"(\d+)\w+", height)[0])
            self.height = f"{h1 + h2}px"

        return buttons

    @requires("IPython.display")
    def display(self, **kwargs: Any) -> "LigNetwork":
        """Prepare and display the network.

        Parameters
        ----------
        width: str = "100%"
        height: str = "500px"
        fontsize: int = 20
        show_interaction_data: bool = False
        """
        html = self._get_html(**kwargs)
        doc = escape(html)
        self._iframe = (
            f'<iframe id="{self.uuid}" width="{self.width}" height="{self.height}"'
            f' frameborder="0" srcdoc="{doc}"></iframe>'
        )
        return self

    @requires("IPython.display")
    def show(self, filename: str, **kwargs: Any) -> "LigNetwork":
        """Save the network as HTML and display the resulting file"""
        html = self._get_html(**kwargs)
        with open(filename, "w") as f:
            f.write(html)
        self._iframe = (
            f'<iframe id="{self.uuid}" width="{self.width}" height="{self.height}"'
            f' frameborder="0" src="{filename}"></iframe>'
        )
        return self

    def save(self, fp: Union[str, Path, "TextIO"], **kwargs: Any) -> None:
        """Save the network to an HTML file

        Parameters
        ----------
        fp : str or file-like object
            Name of the output file, or file-like object
        """
        html = self._get_html(**kwargs)
        if isinstance(fp, str | Path):
            with open(fp, "w") as f:
                f.write(html)
        elif hasattr(fp, "write") and callable(fp.write):
            fp.write(html)

    @requires("IPython.display")
    def save_png(self) -> Any:
        """Saves the current state of the ligplot to a PNG. Not available outside of a
        notebook.

        Notes
        -----
        Requires calling ``display`` or ``show`` first. The legend won't be exported.

        .. versionadded:: 2.1.0
        """
        return display(
            Javascript(f"""
            var iframe = document.getElementById("{self.uuid}");
            var iframe_doc = iframe.contentWindow.document;
            var canvas = iframe_doc.getElementsByTagName("canvas")[0];
            var link = document.createElement("a");
            link.href = canvas.toDataURL();
            link.download = "prolif-lignetwork.png"
            link.click();
            """),
        )

    def _repr_html_(self) -> str | None:
        if self._iframe:
            return self._iframe
        return None
