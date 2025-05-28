/**
 * ProLIF Ligand Interaction Network Visualization
 *
 * JavaScript for rendering the ligand–interaction network,
 * used by network.py to make an interactive graph.
 */

// Main function to draw the network graph using vis-network
// _id: DOM element ID where the network will be rendered
// nodes: array of node objects
// edges: array of edge objects
// options: network configuration options
function drawGraph(_id, nodes, edges, options) {
    // get the container element by ID
    var container = document.getElementById(_id);

    // wrap nodes and edges in vis.DataSet for dynamic updates
    nodes = new vis.DataSet(nodes);
    edges = new vis.DataSet(edges);

    // prepare data object for the network
    var data = { nodes: nodes, edges: edges };

    // instantiate the network
    var network = new vis.Network(container, data, options);

    // once the graph has stabilized, disable physics to lock layout
    network.on("stabilizationIterationsDone", function () {
        network.setOptions({ physics: false });
    });

    return network;
}

// Function to create and attach an interactive legend
// legendId: DOM element ID for the legend container
// buttons: array of legend button definitions (label, group, color)
function createLegend(legendId, buttons) {
    var legend = document.getElementById(legendId);

    // separate containers for residue-type and interaction-type buttons
    var div_residues = document.createElement('div');
    var div_interactions = document.createElement('div');
    var disabled = [];  // track which labels are currently disabled

    // callback to handle click events on legend buttons
    var legend_callback = function() {
        // toggle disabled state on the button
        this.classList.toggle("disabled");
        var hide = this.classList.contains("disabled");
        var show = !hide;
        var btn_label = this.innerHTML;

        // maintain disabled list for filtering logic
        if (hide) {
            disabled.push(btn_label);
        } else {
            disabled = disabled.filter(x => x !== btn_label);
        }

        var node_update = [], edge_update = [];

        // if the button belongs to residue types
        if (this.classList.contains("residues")) {
            nodes.forEach((node) => {
                if (node.residue_type === btn_label) {
                    if (hide && !node.hidden) {
                        // hide node when its residue type is disabled
                        node.hidden = true;
                        node_update.push(node);
                    } else if (show && node.hidden) {
                        // show node only if it still has at least one visible edge
                        var num_edges_active = edges
                            .filter(x => x.to === node.id)
                            .map(x => !x.hidden)
                            .filter(Boolean)
                            .length;
                        if (num_edges_active > 0) {
                            node.hidden = false;
                            node_update.push(node);
                        }
                    }
                } else if (btn_label === "Water") {
                    // hide residues only shown for water bridge
                    res_edges = edges.filter(x => x.to === node.id);
                    water_res_edges = res_edges.filter(
                        x => x.components === "water_protein"
                    );
                    if (
                        (water_res_edges.length)
                        && (water_res_edges.length == res_edges.length)
                    ) {
                        node.hidden = hide;
                        water_res_edges.forEach((edge) => {
                            edge.hidden = hide;
                            edge_update.push(edge);
                        });
                        node_update.push(node);
                    }
                }
            });
        } else {
            // button belongs to interaction types
            edges.forEach((edge) => {
                if (edge.interaction_type === btn_label) {
                    // toggle edge hidden property
                    edge.hidden = hide;
                    edge_update.push(edge);

                    // count remaining visible edges for the target node
                    var num_edges_active = edges
                        .filter(x => x.to === edge.to && !x.hidden)
                        .length;

                    // find index of the target node
                    var ix = nodes.findIndex(x => x.id === edge.to);

                    // only adjust node if its residue type is not disabled
                    if (!disabled.includes(nodes[ix].residue_type)) {
                        if (hide && num_edges_active === 0) {
                            // hide node if no visible edges remain
                            nodes[ix].hidden = true;
                            node_update.push(nodes[ix]);
                        } else if (show && num_edges_active > 0) {
                            // show node if at least one edge is visible
                            nodes[ix].hidden = false;
                            node_update.push(nodes[ix]);
                        }
                    }
                }
            });
        }
        // apply batch updates for both nodes and edges
        ifp.body.data.nodes.update(node_update);
        ifp.body.data.edges.update(edge_update);
    };
    
    // build and style buttons for the legend
    buttons.forEach(function(v) {
        var div = (v.group === "residues") ? div_residues : div_interactions;
        var border = (v.group === "interactions") ? "3px dashed " + v.color : "none";
        var bg = (v.group === "residues") ? v.color : "white";

        var button = document.createElement('button');
        button.classList.add("legend-btn", v.group);
        button.innerHTML = v.label;

        Object.assign(button.style, {
            cursor: "pointer",
            backgroundColor: bg,
            color: v.fontcolor,
            border: border,
            borderRadius: "5px",
            padding: "5px",
            margin: "5px",
            font: "14px 'Arial', sans-serif",
        });
        button.onclick = legend_callback;
        div.appendChild(button);
    });

    // attach the two button groups to the legend container
    legend.appendChild(div_residues);
    legend.appendChild(div_interactions);
}
