[![PyPI - Version](https://badge.fury.io/py/prolif.svg)](https://pypi.org/project/prolif/)
[![PyPI - License](https://img.shields.io/pypi/l/prolif.svg)](https://pypi.org/project/prolif/)
[![PyPI - Status](https://img.shields.io/pypi/status/prolif.svg)](https://pypi.org/project/prolif/)
[![Build Status](https://travis-ci.org/cbouy/ProLIF.svg?branch=master)](https://travis-ci.org/cbouy/ProLIF)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/cbouy)

# ProLIF
Protein-Ligand Interaction Fingerprints

:warning: This project is currently under active development, and might be subject to drastic changes. :warning:

## Description

ProLIF is a tool designed to generate Interaction FingerPrints (IFP) and compute similarity scores for protein-ligand interactions, given a reference ligand.

## Installation

ProLIF is written in Python 3, and uses the following non-standard libraries:
* numpy
* [rdkit](http://www.rdkit.org/docs/Install.html)


Make sure RDKit is installed before proceeding to the next step:
```
conda install -c rdkit rdkit
```

Once this is done, you can download and install the package with Pip:
```
pip install prolif
```

## Usage

ProLIF is a command-line tool, so open a terminal and use the `prolif` command with the following arguments:

```
INPUT arguments:
  -r fileName, --reference fileName
                        Path to your reference ligand.
  -l fileName [fileName ...], --ligand fileName [fileName ...]
                        Path to your ligand(s).
  -p fileName, --protein fileName
                        Path to your protein.
  --residues RESIDUES [RESIDUES ...]
                        Residues chosen for the interactions. Default: automatically detect residues within --cutoff of the reference ligand
  --cutoff float        Cutoff distance for automatic detection of binding site residues. Default: 5.0 Å
  --json fileName       Path to a custom parameters file.

OUTPUT arguments:
  -o filename, --output filename
                        Path to the output CSV file
  --log level           Set the level of the logger. Default: ERROR
  -v, --version         Show version and exit

Other arguments:
  --interactions bit [bit ...]
                        List of interactions used to build the fingerprint.
                                      │          Class         Ligand        Residue
                                      │―――――――――――――――――――――――――――――――――――――――――――――
                              HBdonor │  Hydrogen bond          donor       acceptor
                           HBacceptor │  Hydrogen bond       acceptor          donor
                              XBdonor │   Halogen bond          donor       acceptor
                           XBacceptor │   Halogen bond       acceptor          donor
                               cation │          Ionic         cation          anion
                                anion │          Ionic          anion         cation
                          hydrophobic │    Hydrophobic    hydrophobic    hydrophobic
                           FaceToFace │    Pi-stacking       aromatic       aromatic
                           FaceToEdge │    Pi-stacking       aromatic       aromatic
                            pi-cation │      Pi-cation       aromatic         cation
                            cation-pi │      Pi-cation         cation       aromatic
                              MBdonor │          Metal          metal         ligand
                           MBacceptor │          Metal         ligand          metal
                        Default: HBdonor HBacceptor cation anion FaceToFace FaceToEdge hydrophobic
  --score {tanimoto,dice,tversky}
                        Similarity score between molecule A and B :
                        Let 'a' and 'b' be the number of bits activated in molecules A and B, and 'c' the number of activated bits in common.
                            -) tanimoto : c/(a+b-c). Used by default
                            -) dice     : 2c/(a+b)
                            -) tversky  : c/(alpha*(a-c)+beta*(b-c)+c)
  --alpha int           Alpha parameter for Tversky. Default: 0.7
  --beta int            Beta parameter for Tversky. Default: 0.3

Mandatory arguments: --reference --ligand --protein
MOL2 files only.
```

## License

Unless otherwise noted, all files in this directory and all subdirectories are distributed under the Apache License, Version 2.0:
```
   Copyright 2017-2018 Cédric BOUYSSET

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
