import builtins

IS_NOTEBOOK = hasattr(builtins, "__IPYTHON__")

separated_interaction_colors = {
    "Hydrophobic": "#59e382",
    "VdWContact": "#dfab43",
    "HBAcceptor": "#59bee3",
    "HBDonor": "#239fcd",
    "XBAcceptor": "#ff9f02",
    "XBDonor": "#ce8000",
    "Cationic": "#e35959",
    "Anionic": "#5979e3",
    "CationPi": "#e359d8",
    "PiCation": "#ea85e2",
    "PiStacking": "#b559e3",
    "EdgeToFace": "#c885ea",
    "FaceToFace": "#a22ddc",
    "MetalAcceptor": "#7da982",
    "MetalDonor": "#609267",
}

grouped_interaction_colors = {
    "Hydrophobic": "#59e382",
    "VdWContact": "#dfab43",
    "HBAcceptor": "#59bee3",
    "HBDonor": "#59bee3",
    "XBAcceptor": "#ff9f02",
    "XBDonor": "#ff9f02",
    "Cationic": "#e35959",
    "Anionic": "#5979e3",
    "CationPi": "#e359d8",
    "PiCation": "#e359d8",
    "PiStacking": "#b559e3",
    "EdgeToFace": "#b559e3",
    "FaceToFace": "#b559e3",
    "MetalAcceptor": "#7da982",
    "MetalDonor": "#7da982",
}
