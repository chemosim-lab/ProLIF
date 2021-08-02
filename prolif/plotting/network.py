"""
Plot a Ligand Interaction Network --- :mod:`prolif.plotting.network`
====================================================================

.. versionadded:: 0.3.2

.. autoclass:: LigNetwork
   :members:

"""
from copy import deepcopy
from collections import defaultdict
import warnings
import json
import re
from html import escape
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from ..residue import ResidueId
from ..utils import requires
try:
    from IPython.display import HTML
except ModuleNotFoundError:
    pass
else:
    warnings.filterwarnings("ignore",  # pragma: no cover
                            "Consider using IPython.display.IFrame instead")


class LigNetwork:
    """Creates a ligand interaction diagram

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a 4-level index (ligand, protein, interaction, atom)
        and a weight column for values
    lig_mol : rdkit.Chem.rdChem.Mol
        Ligand molecule
    match3D : bool
        If ``True``, generates 2D coordines that are constrained to fit the
        3D conformation of the ligand as best as possible. Else, generate 2D
        coordinates from scratch
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
    """
    COLORS = {
        "interactions": {
            "Hydrophobic": "#59e382",
            "HBAcceptor": "#59bee3",
            "HBDonor": "#59bee3",
            "XBAcceptor": "#59bee3",
            "XBDonor": "#59bee3",
            "Cationic": "#e35959",
            "Anionic": "#5979e3",
            "CationPi": "#e359d8",
            "PiCation": "#e359d8",
            "PiStacking": "#b559e3",
            "EdgeToFace": "#b559e3",
            "FaceToFace": "#b559e3",
        },
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
            "Sulfur": "#e3ce59"
        }
    }
    RESIDUE_TYPES = {
        'ALA': "Aliphatic",
        'GLY': "Aliphatic",
        'ILE': "Aliphatic",
        'LEU': "Aliphatic",
        'PRO': "Aliphatic",
        'VAL': "Aliphatic",
        'PHE': "Aromatic",
        'TRP': "Aromatic",
        'TYR': "Aromatic",
        'ASP': "Acidic",
        'GLU': "Acidic",
        'ARG': "Basic",
        'HIS': "Basic",
        'HID': "Basic",
        'HIE': "Basic",
        'HIP': "Basic",
        'HSD': "Basic",
        'HSE': "Basic",
        'HSP': "Basic",
        'LYS': "Basic",
        'SER': "Polar",
        'THR': "Polar",
        'ASN': "Polar",
        'GLN': "Polar",
        'CYS': "Sulfur",
        'CYM': "Sulfur",
        'CYX': "Sulfur",
        'MET': "Sulfur",
    }
    _LIG_PI_INTERACTIONS = ["EdgeToFace", "FaceToFace", "PiStacking",
                            "PiCation"]
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

    def __init__(self, df, lig_mol, match3D=True, kekulize=False, molsize=35,
                 rotation=0, carbon=.16):
        self.df = df
        mol = deepcopy(lig_mol)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        self._ring_info = mol.GetRingInfo()
        if kekulize:
            Chem.Kekulize(mol)
        if match3D:
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
                'label': " ",
                'shape': "dot",
                'color': self.COLORS["atoms"]["C"],
                'size': molsize * carbon,
            }
        else:
            self._carbon = {
                'label': " ",
                'shape': "text"
            }
        self.xyz = molsize * xyz
        self.mol = mol
        self._multiplier = molsize
        self.options = {}
        self._max_interaction_width = 6
        self._avoidOverlap = .8
        self._springConstant = .1
        self._bond_color = "black"
        self._default_atom_color = "grey"
        self._default_residue_color = "#dbdbdb"
        self._default_interaction_color = "#dbdbdb"
        # regroup interactions of the same color
        temp = defaultdict(list)
        interactions = set(df.index.get_level_values("interaction").unique())
        for interaction, color in self.COLORS["interactions"].items():
            if interaction in interactions:
                temp[color].append(interaction)
        self._interaction_types = {i: "/".join(t)
                                   for c, t in temp.items()
                                   for i in t}

    @classmethod
    def from_ifp(cls, ifp, lig, kind="aggregate", frame=0, threshold=.3,
                 **kwargs):
        """Helper method to create a ligand interaction diagram from an IFP
        DataFrame obtained with ``fp.to_dataframe(return_atoms=True)``

        Notes
        -----
        Two kinds of diagrams can be rendered: either for a designated frame or
        by aggregating the results on the whole IFP and optionnally discarding
        interactions that occur less frequently than a threshold. In the latter
        case (aggregate), only the most frequent ligand atom interaction is
        rendered.

        Parameters
        ----------
        ifp : pandas.DataFrame
            The result of ``fp.to_dataframe(return_atoms=True)``
        lig : rdkit.Chem.rdChem.Mol
            Ligand molecule
        kind : str
            One of "aggregate" or "frame"
        frame : int or str
            Frame number, as read in ``ifp.index``. Only applicable for
            ``kind="frame"``
        threshold : float
            Frequency threshold, between 0 and 1. Only applicable for
            ``kind="aggregate"``
        kwargs : object
            Other arguments passed to the :class:`LigNetwork` class
        """
        if kind == "aggregate":
            data = (pd.get_dummies(ifp.applymap(lambda x: x[0])
                                      .astype(object),
                                   prefix_sep=", ")
                      .rename(columns=lambda x:
                              x.translate({ord(c): None for c in "()'"}))
                      .mean())
            index = [i.split(", ") for i in data.index]
            index = [[j for j in i[:-1]+[int(float(i[-1]))]] for i in index]
            data.index = pd.MultiIndex.from_tuples(
                index,
                names=["ligand", "protein", "interaction", "atom"])
            data = data.to_frame()
            data.rename(columns={data.columns[-1]: "weight"}, inplace=True)
            # merge different ligand atoms before applying the threshold
            data = data.join(
                data.groupby(level=["ligand", "protein", "interaction"]).sum(),
                rsuffix="_total")
            # threshold and keep most occuring atom
            data = (data
                    .loc[data["weight_total"] >= threshold]
                    .drop(columns="weight_total")
                    .sort_values("weight", ascending=False)
                    .groupby(level=["ligand", "protein", "interaction"])
                    .head(1)
                    .sort_index())
            return cls(data, lig, **kwargs)
        elif kind == "frame":
            data = (ifp
                    .loc[ifp.index == frame]
                    .T
                    .applymap(lambda x: x[0])
                    .dropna()
                    .astype(int)
                    .reset_index())
            data.rename(columns={data.columns[-1]: "atom"}, inplace=True)
            data["weight"] = 1
            data.set_index(["ligand", "protein", "interaction", "atom"],
                           inplace=True)
            return cls(data, lig, **kwargs)
        else:
            raise ValueError(f'{kind!r} must be "aggregate" or "frame"')

    def _make_carbon(self):
        return deepcopy(self._carbon)

    def _make_lig_node(self, atom):
        """Prepare ligand atoms"""
        idx = atom.GetIdx()
        elem = atom.GetSymbol()
        if elem == "H" and idx not in self.df.index.get_level_values("atom"):
            self.exclude.append(idx)
            return
        charge = atom.GetFormalCharge()
        if charge != 0:
            charge = "{}{}".format('' if abs(charge) == 1 else str(charge),
                                   '+' if charge > 0 else '-')
            label = f"{elem}{charge}"
            shape = "ellipse"
        else:
            label = elem
            shape = "circle"
        if elem == "C":
            node = self._make_carbon()
        else:
            node = {
                'label': label,
                'shape': shape,
                'color': "white",
                'font': {
                    'color': self.COLORS["atoms"].get(elem,
                                                      self._default_atom_color)
                }
            }
        node.update({
            'id': idx,
            'x': float(self.xyz[idx, 0]),
            'y': float(self.xyz[idx, 1]),
            'fixed': True,
            'group': "ligand",
            'borderWidth': 0
        })
        self.nodes[idx] = node

    def _make_lig_edge(self, bond):
        """Prepare ligand bonds"""
        idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        if any(i in self.exclude for i in idx):
            return
        btype = bond.GetBondTypeAsDouble()
        if btype == 1:
            self.edges.append({
                'from': idx[0], 'to': idx[1],
                'color': self._bond_color,
                'physics': False,
                'group': "ligand",
                'width': 4,
            })
        else:
            self._make_non_single_bond(idx, btype)

    def _make_non_single_bond(self, ids, btype, bdist=.06,
                              dash=[10]):
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
                    'id': _id, 'x': xy[0], 'y': xy[1],
                    'shape': "text", "label": " ",
                    'fixed': True, 'physics': False}
        l1, l2, r1, r2 = nodes
        self.edges.extend([
            {'from': l1, 'to': l2,
             'color': self._bond_color,
             'physics': False,
             'dashes': dashes,
             'group': "ligand",
             'width': 4},
            {'from': r1, 'to': r2,
             'color': self._bond_color,
             'physics': False,
             'dashes': dashes,
             'group': "ligand",
             'width': 4}
        ])
        if btype == 3:
            self.edges.append({
                'from': ids[0], 'to': ids[1],
                'color': self._bond_color,
                'physics': False,
                'group': "ligand",
                'width': 4
            })

    def _make_interactions(self, mass=2):
        """Prepare lig-prot interactions"""
        restypes = {}
        for prot_res in self.df.index.get_level_values("protein").unique():
            resname = ResidueId.from_string(prot_res).name
            restype = self.RESIDUE_TYPES.get(resname)
            restypes[prot_res] = restype
            color = self.COLORS["residues"].get(restype,
                                                self._default_residue_color)
            node = {
                'id': prot_res,
                'label': prot_res,
                'color': color,
                'shape': "box",
                'borderWidth': 0,
                'physics': True,
                'mass': mass,
                'group': "protein",
                'residue_type': restype,
            }
            self.nodes[prot_res] = node
        for ((lig_res, prot_res, interaction, lig_id),
             (weight,)) in self.df.iterrows():
            if interaction in self._LIG_PI_INTERACTIONS:
                centroid = self._get_ring_centroid(lig_id)
                origin = str((lig_res, prot_res, interaction))
                self.nodes[origin] = {'id': origin,
                                      'x': centroid[0], 'y': centroid[1],
                                      'shape': "text", "label": " ",
                                      'fixed': True, 'physics': False}
            else:
                origin = int(lig_id)
            edge = {
                'from': origin, 'to': prot_res,
                'title': interaction,
                'interaction_type': self._interaction_types[interaction],
                'color': self.COLORS["interactions"].get(
                    interaction,
                    self._default_interaction_color),
                'smooth': {
                    'type': 'cubicBezier',
                    'roundness': .2},
                'dashes': [10],
                'width': weight * self._max_interaction_width,
                'group': "interaction",
            }
            self.edges.append(edge)

    def _get_ring_centroid(self, index):
        """Find ring coordinates using the index of one of the ring atoms"""
        for r in self._ring_info.AtomRings():
            if index in r:
                break
        else:
            raise ValueError("No ring containing this atom index was found in "
                             "the given molecule")
        return self.xyz[list(r)].mean(axis=0)

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
            label = re.sub(r'(\w+)(.*)', fr'\1{h_str}\2', node["label"])
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

    def _get_js(self, width="100%", height="500px", div_id="mynetwork",
                fontsize=20):
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
            }
        }
        options.update(self.options)
        js = self._JS_TEMPLATE % dict(div_id=div_id,
                                      nodes=json.dumps(self.nodes),
                                      edges=json.dumps(self.edges),
                                      options=json.dumps(options))
        js += self._get_legend()
        return js

    def _get_html(self, **kwargs):
        """Returns the HTML code to draw the network"""
        return self._HTML_TEMPLATE % dict(js=self._get_js(**kwargs))

    def _get_legend(self, height="90px"):
        available = {}
        buttons = []
        map_color_restype = {c: t for t, c in self.COLORS["residues"].items()}
        map_color_interactions = {self.COLORS["interactions"][i]: t
                                  for i, t in self._interaction_types.items()}
        # residues
        for node in self.nodes:
            if node.get("group", "") == "protein":
                color = node["color"]
                available[color] = map_color_restype.get(color, "Unknown")
        available = {k: v for k, v in sorted(available.items(),
                                             key=lambda item: item[1])}
        for i, (color, restype) in enumerate(available.items()):
            buttons.append({
                "index": i,
                "label": restype,
                "color": color,
                "group": "residues"
            })
        # interactions
        available.clear()
        for edge in self.edges:
            if edge.get("group", "") == "interaction":
                color = edge["color"]
                available[color] = map_color_interactions.get(color, "Unknown")
        available = {k: v for k, v in sorted(available.items(),
                                             key=lambda item: item[1])}
        for i, (color, interaction) in enumerate(available.items()):
            buttons.append({
                    "index": i,
                    "label": interaction,
                    "color": color,
                    "group": "interactions"
                })
        # JS code
        if all("px" in h for h in [self.height, height]):
            h1 = int(re.findall(r'(\d+)\w+', self.height)[0])
            h2 = int(re.findall(r'(\d+)\w+', height)[0])
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
        """ % dict(div_id="networklegend",
                   buttons=json.dumps(buttons))

    @requires("IPython.display")
    def display(self, **kwargs):
        """Prepare and display the network"""
        html = self._get_html(**kwargs)
        iframe = ('<iframe width="{width}" height="{height}" frameborder="0" '
                  'srcdoc="{doc}"></iframe>')
        return HTML(iframe.format(width=self.width, height=self.height,
                                  doc=escape(html)))

    @requires("IPython.display")
    def show(self, filename, **kwargs):
        """Save the network as HTML and display the resulting file"""
        html = self._get_html(**kwargs)
        with open(filename, "w") as f:
            f.write(html)
        iframe = ('<iframe width="{width}" height="{height}" frameborder="0" '
                  'src="{filename}"></iframe>')
        return HTML(iframe.format(width=self.width, height=self.height,
                                  filename=filename))

    def save(self, fp, **kwargs):
        """Save the network to an HTML file

        Parameters
        ----------
        fp : str or file-like object
            Name of the output file, or file-like object
        """
        html = self._get_html(**kwargs)
        try:
            fp.write(html)
        except AttributeError:
            with open(fp, "w") as f:
                f.write(html)
