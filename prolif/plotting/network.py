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
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor

from prolif.exceptions import RunRequiredError
from prolif.plotting.utils import grouped_interaction_colors
from prolif.residue import ResidueId
from prolif.utils import requires

try:
    from IPython.display import HTML
except ModuleNotFoundError:
    pass
else:
    warnings.filterwarnings(
        "ignore", "Consider using IPython.display.IFrame instead"  # pragma: no cover
    )


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
    rotation : int
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
    """

    COLORS = {
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
        },
    }
    RESIDUE_TYPES = {
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
    }
    _LIG_PI_INTERACTIONS = ["EdgeToFace", "FaceToFace", "PiStacking", "PiCation"]
    _DISPLAYED_ATOM = {  # index 0 in indices tuple by default
        "HBDonor": 1,
        "XBDonor": 1,
    }
    _JS_TEMPLATE = """
        var ifp, legend, nodes, edges, legend_buttons;
        function drawGraph(_id, nodes, edges, options) {
            var container = document.getElementById(_id);
            nodes = new vis.DataSet(nodes);
            edges = new vis.DataSet(edges);
            var data = {nodes: nodes, edges: edges};
            var network = new vis.Network(container, data, options);
            network.on("stabilizationIterationsDone", function () {
                network.setOptions( { physics: false } );
            });
            return network;
        }
        nodes = %(nodes)s;
        edges = %(edges)s;
        ifp = drawGraph('%(div_id)s', nodes, edges, %(options)s);
    """
    _HTML_TEMPLATE = """
        <html>
        <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network@9.0.4/dist/vis-network.min.js"></script>
        <link href="https://unpkg.com/vis-network@9.0.4/dist/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
        <style type="text/css">
            body {
                padding: 0;
                margin: 0;
                background: #fff;
            }
            .legend-btn.residues.disabled {
                background: #b4b4b4 !important;
                color: #555 !important;
            }
            .legend-btn.interactions.disabled {
                border-color: #b4b4b4 !important;
                color: #555 !important;
            }
        </style>
        </head>
        <body>
        <div id="mynetwork"></div>
        <div id="networklegend"></div>
        <script type="text/javascript">
            %(js)s
        </script>
        </body>
        </html>
    """

    def __init__(
        self,
        df,
        lig_mol,
        use_coordinates=False,
        flatten_coordinates=True,
        kekulize=False,
        molsize=35,
        rotation=0,
        carbon=0.16,
    ):
        self.df = df
        self._interacting_atoms = set(
            [atom for atoms in df.index.get_level_values("atoms") for atom in atoms]
        )
        mol = deepcopy(lig_mol)
        if kekulize:
            Chem.Kekulize(mol)
        if use_coordinates:
            if flatten_coordinates:
                rdDepictor.GenerateDepictionMatching3DStructure(mol, lig_mol)
        else:
            rdDepictor.Compute2DCoords(mol, clearConfs=True)
        xyz = mol.GetConformer().GetPositions()
        if rotation:
            theta = np.radians(rotation)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, s], [-s, c]])
            xy, z = xyz[:, :2], xyz[:, 2:3]
            center = xy.mean(axis=0)
            xy = ((xy - center) @ R.T) + center
            xyz = np.concatenate([xy, z], axis=1)
        if carbon:
            self._carbon = {
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
        self.options = {}
        self._max_interaction_width = 6
        self._avoidOverlap = 0.8
        self._springConstant = 0.1
        self._bond_color = "black"
        self._default_atom_color = "grey"
        self._default_residue_color = "#dbdbdb"
        self._default_interaction_color = "#dbdbdb"
        # regroup interactions of the same color
        temp = defaultdict(list)
        interactions = set(df.index.get_level_values("interaction").unique())
        for interaction in interactions:
            color = self.COLORS["interactions"].get(
                interaction, self._default_interaction_color
            )
            temp[color].append(interaction)
        self._interaction_types = {
            interaction: "/".join(interaction_group)
            for interaction_group in temp.values()
            for interaction in interaction_group
        }

    @classmethod
    def from_fingerprint(
        cls,
        fp,
        ligand_mol,
        kind="aggregate",
        frame=0,
        display_all=False,
        threshold=0.3,
        **kwargs,
    ):
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
                "Please run the fingerprint analysis before attempting to display results."
            )
        if kind == "frame":
            df = cls._make_frame_df_from_fp(fp, frame=frame, display_all=display_all)
            return cls(df, ligand_mol, **kwargs)
        if kind == "aggregate":
            df = cls._make_agg_df_from_fp(fp, threshold=threshold)
            return cls(df, ligand_mol, **kwargs)
        raise ValueError(f'{kind!r} must be "aggregate" or "frame"')

    @staticmethod
    def _get_records(ifp, all_metadata):
        records = []
        for (lig_resid, prot_resid), int_data in ifp.items():
            for int_name, metadata_tuple in int_data.items():
                entry = {
                    "ligand": str(lig_resid),
                    "protein": str(prot_resid),
                    "interaction": int_name,
                }
                if all_metadata:
                    for metadata in metadata_tuple:
                        records.append(
                            {
                                **entry,
                                "atoms": metadata["parent_indices"]["ligand"],
                                "distance": metadata.get("distance", 0),
                            }
                        )
                else:
                    # extract interaction with shortest distance
                    metadata = min(
                        metadata_tuple, key=lambda m: m.get("distance", np.nan)
                    )
                    entry["atoms"] = metadata["parent_indices"]["ligand"]
                    entry["distance"] = metadata.get("distance", 0)
                    records.append(entry)
        return records

    @classmethod
    def _make_agg_df_from_fp(cls, fp, threshold=0.3):
        data = []
        for ifp in fp.ifp.values():
            data.extend(cls._get_records(ifp, all_metadata=False))
        df = pd.DataFrame(data)
        # add weight for each atoms, and average distance
        df["weight"] = 1
        df = df.groupby(["ligand", "protein", "interaction", "atoms"]).agg(
            weight=("weight", "sum"), distance=("distance", "mean")
        )
        df["weight"] = df["weight"] / len(fp.ifp)
        # merge different ligand atoms of the same residue/interaction group before
        # applying the threshold
        df = df.join(
            df.groupby(level=["ligand", "protein", "interaction"]).agg(
                weight_total=("weight", "sum")
            ),
        )
        # threshold and keep most occuring ligand atom
        df = (
            df.loc[df["weight_total"] >= threshold]
            .drop(columns="weight_total")
            .sort_values("weight", ascending=False)
            .groupby(level=["ligand", "protein", "interaction"])
            .head(1)
            .sort_index()
        )
        return df

    @classmethod
    def _make_frame_df_from_fp(cls, fp, frame=0, display_all=False):
        ifp = fp.ifp[frame]
        data = cls._get_records(ifp, all_metadata=display_all)
        df = pd.DataFrame(data)
        df["weight"] = 1
        df = df.set_index(["ligand", "protein", "interaction", "atoms"]).reindex(
            columns=["weight", "distance"]
        )
        return df

    def _make_carbon(self):
        return deepcopy(self._carbon)

    def _make_lig_node(self, atom):
        """Prepare ligand atoms"""
        idx = atom.GetIdx()
        elem = atom.GetSymbol()
        if elem == "H" and idx not in self._interacting_atoms:
            self.exclude.append(idx)
            return
        charge = atom.GetFormalCharge()
        if charge != 0:
            charge = "{}{}".format(
                "" if abs(charge) == 1 else str(charge), "+" if charge > 0 else "-"
            )
            label = f"{elem}{charge}"
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
                    "color": self.COLORS["atoms"].get(elem, self._default_atom_color)
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
            }
        )
        self.nodes[idx] = node

    def _make_lig_edge(self, bond):
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
                }
            )
        else:
            self._make_non_single_bond(idx, btype)

    def _make_non_single_bond(self, ids, btype, bdist=0.06, dash=[10]):
        """Prepare double, triple and aromatic bonds"""
        xyz = self.xyz[ids]
        d = xyz[1, :2] - xyz[0, :2]
        length = np.sqrt((d**2).sum())
        u = d / length
        p = np.array([-u[1], u[0]])
        nodes = []
        dist = bdist * self._multiplier * np.ceil(btype)
        dashes = False if btype in [2, 3] else dash
        for perp in (p, -p):
            for point in xyz:
                xy = point[:2] + perp * dist
                _id = hash(xy.tobytes())
                nodes.append(_id)
                self.nodes[_id] = {
                    "id": _id,
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
            ]
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
                }
            )

    def _make_interactions(self, mass=2):
        """Prepare lig-prot interactions"""
        restypes = {}
        for prot_res in self.df.index.get_level_values("protein").unique():
            resname = ResidueId.from_string(prot_res).name
            restype = self.RESIDUE_TYPES.get(resname)
            restypes[prot_res] = restype
            color = self.COLORS["residues"].get(restype, self._default_residue_color)
            node = {
                "id": prot_res,
                "label": prot_res,
                "color": color,
                "shape": "box",
                "borderWidth": 0,
                "physics": True,
                "mass": mass,
                "group": "protein",
                "residue_type": restype,
            }
            self.nodes[prot_res] = node
        for (lig_res, prot_res, interaction, lig_indices), (
            weight,
            distance,
        ) in self.df.iterrows():
            if interaction in self._LIG_PI_INTERACTIONS:
                centroid = self._get_ring_centroid(lig_indices)
                origin = f"centroid({lig_res}, {prot_res}, {interaction})"
                self.nodes[origin] = {
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
            edge = {
                "from": origin,
                "to": prot_res,
                "title": f"{interaction}: {distance:.2f}Å",
                "interaction_type": self._interaction_types.get(
                    interaction, interaction
                ),
                "color": self.COLORS["interactions"].get(
                    interaction, self._default_interaction_color
                ),
                "smooth": {"type": "cubicBezier", "roundness": 0.2},
                "dashes": [10],
                "width": weight * self._max_interaction_width,
                "group": "interaction",
            }
            self.edges.append(edge)

    def _get_ring_centroid(self, indices):
        """Find ring centroid coordinates using the indices of the ring atoms"""
        return self.xyz[list(indices)].mean(axis=0)

    def _patch_hydrogens(self):
        """Patch hydrogens on heteroatoms

        Hydrogen atoms that aren't part of any interaction have been hidden at
        this stage, but they should be added to the label of the heteroatom for
        clarity
        """
        to_patch = defaultdict(int)
        for idx in self.exclude:
            h = self.mol.GetAtomWithIdx(idx)
            atom = h.GetNeighbors()[0]
            if atom.GetSymbol() != "C":
                to_patch[atom.GetIdx()] += 1
        for idx, nH in to_patch.items():
            node = self.nodes[idx]
            h_str = "H" if nH == 1 else f"H{nH}"
            label = re.sub(r"(\w+)(.*)", rf"\1{h_str}\2", node["label"])
            node["label"] = label
            node["shape"] = "ellipse"

    def _make_graph_data(self):
        """Prepares the nodes and edges"""
        self.exclude = []
        self.nodes = {}
        self.edges = []
        # show residues
        self._make_interactions()
        # show ligand
        for atom in self.mol.GetAtoms():
            self._make_lig_node(atom)
        for bond in self.mol.GetBonds():
            self._make_lig_edge(bond)
        self._patch_hydrogens()
        self.nodes = list(self.nodes.values())

    def _get_js(self, width="100%", height="500px", div_id="mynetwork", fontsize=20):
        """Returns the JavaScript code to draw the network"""
        self.width = width
        self.height = height
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
                }
            },
        }
        options.update(self.options)
        js = self._JS_TEMPLATE % dict(
            div_id=div_id,
            nodes=json.dumps(self.nodes),
            edges=json.dumps(self.edges),
            options=json.dumps(options),
        )
        js += self._get_legend()
        return js

    def _get_html(self, **kwargs):
        """Returns the HTML code to draw the network"""
        return self._HTML_TEMPLATE % dict(js=self._get_js(**kwargs))

    def _get_legend(self, height="90px"):
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
        available = {
            k: v for k, v in sorted(available.items(), key=lambda item: item[1])
        }
        for i, (color, restype) in enumerate(available.items()):
            buttons.append(
                {"index": i, "label": restype, "color": color, "group": "residues"}
            )
        # interactions
        available.clear()
        for edge in self.edges:
            if edge.get("group", "") == "interaction":
                color = edge["color"]
                available[color] = map_color_interactions[color]
        available = {
            k: v for k, v in sorted(available.items(), key=lambda item: item[1])
        }
        for i, (color, interaction) in enumerate(available.items()):
            buttons.append(
                {
                    "index": i,
                    "label": interaction,
                    "color": color,
                    "group": "interactions",
                }
            )
        # JS code
        if all("px" in h for h in [self.height, height]):
            h1 = int(re.findall(r"(\d+)\w+", self.height)[0])
            h2 = int(re.findall(r"(\d+)\w+", height)[0])
            self.height = f"{h1+h2}px"
        return """
        legend_buttons = %(buttons)s;
        legend = document.getElementById('%(div_id)s');
        var div_residues = document.createElement('div');
        var div_interactions = document.createElement('div');
        var disabled = [];
        var legend_callback = function() {
            this.classList.toggle("disabled");
            var hide = this.classList.contains("disabled");
            var show = !hide;
            var btn_label = this.innerHTML;
            if (hide) {
                disabled.push(btn_label);
            } else {
                disabled = disabled.filter(x => x !== btn_label);
            }
            var node_update = [],
                edge_update = [];
            // click on residue type
            if (this.classList.contains("residues")) {
                nodes.forEach((node) => {
                    // find nodes corresponding to this type
                    if (node.residue_type === btn_label) {
                        // if hiding this type and residue isn't already hidden
                        if (hide && !node.hidden) {
                            node.hidden = true;
                            node_update.push(node);
                        // if showing this type and residue isn't already visible
                        } else if (show && node.hidden) {
                            // display if there's at least one of its edge that isn't hidden
                            num_edges_active = edges.filter(x => x.to === node.id)
                                                    .map(x => Boolean(x.hidden))
                                                    .filter(x => !x)
                                                    .length;
                            if (num_edges_active > 0) {
                                node.hidden = false;
                                node_update.push(node);
                            }
                        }
                    }
                });
                ifp.body.data.nodes.update(node_update);
            // click on interaction type
            } else {
                edges.forEach((edge) => {
                    // find edges corresponding to this type
                    if (edge.interaction_type === btn_label) {
                        edge.hidden = !edge.hidden;
                        edge_update.push(edge);
                        // number of active edges for the corresponding residue
                        var num_edges_active = edges.filter(x => x.to === edge.to)
                                               .map(x => Boolean(x.hidden))
                                               .filter(x => !x)
                                               .length;
                        // find corresponding residue
                        var ix = nodes.findIndex(x => x.id === edge.to);
                        // only change visibility if residue_type not being hidden
                        if (!(disabled.includes(nodes[ix].residue_type))) {
                            // hide if no edge being shown for this residue
                            if (hide && (num_edges_active === 0)) {
                                nodes[ix].hidden = true;
                                node_update.push(nodes[ix]);
                            // show if edges are being shown
                            } else if (show && (num_edges_active > 0)) {
                                nodes[ix].hidden = false;
                                node_update.push(nodes[ix]);
                            }
                        }
                    }
                });
                ifp.body.data.nodes.update(node_update);
                ifp.body.data.edges.update(edge_update);
            }
        };
        legend_buttons.forEach(function(v,i) {
            if (v.group === "residues") {
                var div = div_residues;
                var border = "none";
                var color = v.color;
            } else {
                var div = div_interactions;
                var border = "3px dashed " + v.color;
                var color = "white";
            }
            var button = div.appendChild(document.createElement('button'));
            button.classList.add("legend-btn", v.group);
            button.innerHTML = v.label;
            Object.assign(button.style, {
                "cursor": "pointer",
                "background-color": color,
                "border": border,
                "border-radius": "5px",
                "padding": "5px",
                "margin": "5px",
                "font": "14px 'Arial', sans-serif",
            });
            button.onclick = legend_callback;
        });
        legend.appendChild(div_residues);
        legend.appendChild(div_interactions);
        """ % dict(
            div_id="networklegend", buttons=json.dumps(buttons)
        )

    @requires("IPython.display")
    def display(self, **kwargs):
        """Prepare and display the network"""
        html = self._get_html(**kwargs)
        iframe = (
            '<iframe width="{width}" height="{height}" frameborder="0" '
            'srcdoc="{doc}"></iframe>'
        )
        return HTML(
            iframe.format(width=self.width, height=self.height, doc=escape(html))
        )

    @requires("IPython.display")
    def show(self, filename, **kwargs):
        """Save the network as HTML and display the resulting file"""
        html = self._get_html(**kwargs)
        with open(filename, "w") as f:
            f.write(html)
        iframe = (
            '<iframe width="{width}" height="{height}" frameborder="0" '
            'src="{filename}"></iframe>'
        )
        return HTML(
            iframe.format(width=self.width, height=self.height, filename=filename)
        )

    def save(self, fp, **kwargs):
        """Save the network to an HTML file

        Parameters
        ----------
        fp : str or file-like object
            Name of the output file, or file-like object
        """
        html = self._get_html(**kwargs)
        if isinstance(fp, (str, Path)):
            with open(fp, "w") as f:
                f.write(html)
        elif hasattr(fp, "write") and callable(fp.write):
            fp.write(html)
