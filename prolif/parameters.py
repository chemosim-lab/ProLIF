# Geometric rules used to detect interactions
# Mostly adapted from Marcou & Rognan; JCIM 2007 (doi: 10.1021/ci600342e)
# and from Schrödinger's Ligand Interaction Diagram (https://www.schrodinger.com/kb/1556)
# For Halogen bonds, see Varadwaj et al.; Inorganics 2019 (doi: 10.3390/inorganics7030040)
RULES = {
  "HBond": {                    # Donor-Hydrogen ... Acceptor
    "donor":    "[O,N,S][H]",   # SMARTS query for [D][H]
    "acceptor": "[O,N,*-;!+]",  # SMARTS query for [A]
    "distance": 3.0,            # between H and A, in Angstroms
    "angle":    [0, 50]         # between DH and HA, in degrees
  },
  "XBond": {                                                # Donor-Halogen ... Acceptor
    "donor":    "[#6,#7,Si,F,Cl,Br,I]-[F,Cl,Br,I,At]",      # [D][X]
    "acceptor": "[F-,Cl-,Br-,I-,#7,O,P,S,Se,Te,a&R;!+]",    # [A]
    "distance": 3.3,                                        # between X and A
    "angle":    [0, 25]                                     # between DX and XA
  },
  "Ionic": {
    "cation":   "[*+]",
    "anion":    "[*-]",
    "distance": 5.0
  },
  "Aromatic": {
    "smarts": [
        "[a]1:[a]:[a]:[a]:[a]:[a]:1",
        "[a]1:[a]:[a]:[a]:[a]:1",
    ],
    "FaceToFace": {
        "angle": [0, 30],   # between ring planes
        "distance": 4.4,    # between ring centroids
    },
    "EdgeToFace": {
        "angle": [60, 90],  # between ring planes
        "distance": 5.5,    # between ring centroids
    },
  },
  "Cation-Pi": {
    "distance": 5.0,       # between cation and centroid
    "angle":    [0, 30]    # between normal to ring plane, and line going from the ring centroid to the cation
  },
  "Hydrophobic": {
    "smarts":   "[C,S,F,Cl,Br,I]",
    "distance": 4.5
  },
  "Metallic": {
    "metal":   "[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]",
    "ligand":  "[O,N,*-;!+]",
    "distance": 2.8
  }
}
