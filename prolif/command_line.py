"""
   Copyright 2017 Cédric BOUYSSET

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import argparse, textwrap, os, sys
from . import prolif
from .version import __version__

class Range:
    """Class to raise an exception if the value is not between start and end.
    Used for alpha and beta in Tversky"""
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
    def __repr__(self):
        return '{} to {}'.format(self.start, self.end)

def cli():
    jsonpath = os.path.join(os.path.dirname(__file__),'parameters.json')
    description = 'ProLIF: Protein Ligand Interaction Fingerprints\nGenerates Interaction FingerPrints (IFP) and a similarity score for protein-ligand interactions'
    epilog = 'Mandatory arguments: --reference --ligand --protein\nMOL2 files only.'
    # Argparse
    parser = argparse.ArgumentParser(description=description, epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter)

    group_input = parser.add_argument_group('INPUT arguments')
    group_input.add_argument("-r", "--reference", metavar='fileName', type=str, required=True,
        help="Path to your reference ligand.")
    group_input.add_argument("-l", "--ligand", metavar='fileName', type=str, nargs='+', required=True,
        help="Path to your ligand(s).")
    group_input.add_argument("-p", "--protein", metavar='fileName', type=str, required=True,
        help="Path to your protein.")
    group_input.add_argument("--residues", type=str, nargs='+', default=None,
        help="Residues chosen for the interactions. Default: automatically detect residues within --cutoff of the reference ligand")
    group_input.add_argument("--cutoff", metavar='float', type=float, required=False, default=5.0,
        help="Cutoff distance for automatic detection of binding site residues. Default: 5.0 Å")
    group_input.add_argument("--json", metavar='fileName', type=str, default=jsonpath,
        help="Path to a custom parameters file. Default: {}".format(jsonpath))

    group_output = parser.add_argument_group('OUTPUT arguments')
    group_output.add_argument("-o", "--output", metavar='filename', type=str,
        help="Path to the output CSV file")
    group_output.add_argument("--log", metavar="level", help="Set the level of the logger. Default: ERROR",
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], default='ERROR')
    group_output.add_argument("-v", "--version", action="version",
        version='ProLIF {}'.format(__version__), help="Show version and exit")

    group_args = parser.add_argument_group('Other arguments')
    table = [
        ['', 'Class','Ligand','Residue'],
        ['','―'*15,'―'*15,'―'*15],
        ['HBdonor', 'Hydrogen bond', 'donor', 'acceptor'],
        ['HBacceptor', 'Hydrogen bond', 'acceptor', 'donor'],
        ['XBdonor', 'Halogen bond', 'donor', 'acceptor'],
        ['XBacceptor', 'Halogen bond', 'acceptor', 'donor'],
        ['cation', 'Ionic', 'cation', 'anion'],
        ['anion', 'Ionic', 'anion', 'cation'],
        ['hydrophobic', 'Hydrophobic', 'hydrophobic', 'hydrophobic'],
        ['FaceToFace', 'Pi-stacking', 'aromatic', 'aromatic'],
        ['EdgeToFace', 'Pi-stacking', 'aromatic', 'aromatic'],
        ['pi-cation', 'Pi-cation', 'aromatic', 'cation'],
        ['cation-pi', 'Pi-cation', 'cation', 'aromatic'],
        ['MBdonor', 'Metal', 'metal', 'ligand'],
        ['MBacceptor', 'Metal', 'ligand', 'metal'],
    ]
    defaults = ['HBdonor','HBacceptor','cation','anion','FaceToFace','EdgeToFace','hydrophobic']
    table_as_str = '\n'.join(['{:>13} │{:>15}{:>15}{:>15}'.format(*line) for line in table])
    group_args.add_argument("--interactions", metavar="bit", nargs='+',
        choices=[line[0] for line in table[2:]],
        default=defaults,
        help=textwrap.dedent("""List of interactions used to build the fingerprint.
            {}\nDefault: {}""").format(table_as_str, ' '.join(defaults)))
    group_args.add_argument("--score", choices=['tanimoto', 'dice', 'tversky'], default='tanimoto',
        help=textwrap.dedent("""Similarity score between molecule A and B :
            Let 'a' and 'b' be the number of bits activated in molecules A and B, and 'c' the number of activated bits in common.
                -) tanimoto : c/(a+b-c). Used by default
                -) dice     : 2c/(a+b)
                -) tversky  : c/(alpha*(a-c)+beta*(b-c)+c)
                """))
    group_args.add_argument("--alpha", metavar="int", type=float, choices=[Range(0.0, 1.0)], default=0.7,
        help="Alpha parameter for Tversky. Default: 0.7")
    group_args.add_argument("--beta", metavar="int", type=float, choices=[Range(0.0, 1.0)], default=0.3,
        help="Beta parameter for Tversky. Default: 0.3")

    # Parse arguments from command line
    args = parser.parse_args()

    prolif.main(args)
