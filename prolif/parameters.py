# Geometric rules used to detect interactions
# Mostly adapted from Marcou & Rognan; JCIM 2007 (doi: 10.1021/ci600342e)
# and from Schrödinger's Ligand Interaction Diagram (https://www.schrodinger.com/kb/1556)
# For Halogen bonds, see Auffinger et al.; PNAS 2004 (doi: 10.1073/pnas.0407607101)
# and Scholfield et al.; Protein Sci. 2012 (doi: 10.1002/pro.2201)
RULES = {
  "HBond": {                    # Donor-Hydrogen ... Acceptor
    "donor":    "[O,N,S]-[H]",   # SMARTS query for [D][H]
    "acceptor": "[O,N,F,*-;!+]",  # SMARTS query for [A]
    "distance": 3.0,            # between H and A, in Angstroms
    "angle":    [0, 50]         # between DH and HA, in degrees
  },
  "XBond": {                                                # Donor-Halogen ... Acceptor
    "donor":    "[#6,#7,Si,F,Cl,Br,I][F,Cl,Br,I,At]",       # [D][X]
    "acceptor": "[F-,Cl-,Br-,I-,#7,O,P,S,Se,Te,a&R;!+][*]", # [A][R]
    "distance": 3.2,                                        # between X and A
    "angle":    {
        "AXD": [160, 180],                                  # between XD and XA
        "XAR": [90, 130],                                   # between AX and AR
    },
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
    "smarts":   "[#6,S,F,Cl,Br,I;!+;!-]",
    "distance": 4.5
  },
  "Metallic": {
    "metal":   "[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]",
    "ligand":  "[O,N,*-;!+]",
    "distance": 2.8
  }
}
