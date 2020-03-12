import argparse, textwrap, os, sys
import logging
from rdkit import RDLogger
from .ligand import Ligand
from .protein import Protein
from .fingerprint import Fingerprint
from .utils import get_resnumber
from .version import __version__

logger = logging.getLogger("prolif")

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

def parse_args():
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
        ['Cation', 'Ionic', 'cation', 'anion'],
        ['Anion', 'Ionic', 'anion', 'cation'],
        ['Hydrophobic', 'Hydrophobic', 'hydrophobic', 'hydrophobic'],
        ['PiStacking', 'π-stacking', 'aromatic', 'aromatic'],
        ['FaceToFace', 'π-stacking', 'aromatic', 'aromatic'],
        ['EdgeToFace', 'π-stacking', 'aromatic', 'aromatic'],
        ['PiCation', 'Cation-π', 'aromatic', 'cation'],
        ['CationPi', 'Cation-π', 'cation', 'aromatic'],
        ['MBdonor', 'Metallic', 'metal', 'ligand'],
        ['MBacceptor', 'Metallic', 'ligand', 'metal'],
    ]
    defaults = ['HBdonor','HBacceptor','Cation','Anion','PiStacking','Hydrophobic']
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

    return args

def main():
    # parse command line arguments
    args = parse_args()

    # set logger level
    lg = RDLogger.logger()
    if args.log == 'CRITICAL':
        lg.setLevel(RDLogger.CRITICAL)
        logger.setLevel(logging.CRITICAL)
    elif args.log == 'ERROR':
        lg.setLevel(RDLogger.ERROR)
        logger.setLevel(logging.ERROR)
    elif args.log == 'WARNING':
        lg.setLevel(RDLogger.WARNING)
        logger.setLevel(logging.WARNING)
    elif args.log == 'INFO':
        lg.setLevel(RDLogger.INFO)
        logger.setLevel(logging.INFO)
    elif args.log == 'DEBUG':
        lg.setLevel(RDLogger.DEBUG)
        logger.setLevel(logging.DEBUG)
    logger.info('Using {} to compute similarity between fingerprints'.format(args.score))
    # Read files
    fingerprint = Fingerprint(args.json, args.interactions)
    reference   = Ligand(args.reference)
    protein     = Protein(args.protein, reference, cutoff=args.cutoff, residueList=args.residues)
    residues    = [protein.residues[residue] for residue in sorted(protein.residues, key=get_resnumber)]
    # Print residues on terminal:
    bitstr_length  = len(args.interactions)
    print(''.join('{resname: <{bitstr_length}s}'.format(
        resname=residue.resname, bitstr_length=bitstr_length
        ) for residue in residues))
    # Generate the IFP between the reference ligand and the protein
    fingerprint.generate_ifp(reference, protein)
    ifp_list = [reference.IFP[i:i+bitstr_length] for i in range(0, len(reference.IFP), bitstr_length)]
    print(''.join('{ifp: <{size}s}'.format(
        ifp=ifp_list[i], size=len(residues[i].resname)
        ) for i in range(len(ifp_list))), reference.inputFile)

    # Loop over ligands:
    ligandList = []
    for lig in args.ligand:
        ligand = Ligand(lig)
        # Generate the IFP between a ligand and the protein
        fingerprint.generate_ifp(ligand, protein)
        # Calculate similarity
        score = ligand.getSimilarity(reference, args.score, args.alpha, args.beta)
        ligand.setSimilarity(score)
        ifp_list = [ligand.IFP[i:i+bitstr_length] for i in range(0, len(ligand.IFP), bitstr_length)]
        print(''.join('{ifp: <{size}s}'.format(ifp=ifp_list[i], size=len(residues[i].resname)) for i in range(len(ifp_list)) ),
              '{:.3f}'.format(ligand.score), ligand.inputFile)
        ligandList.append(ligand)

    # Output
    if args.output:
        logger.info('Writing CSV formatted output to ' + args.output)
        with open(args.output, 'w') as f:
            f.write('File,SimilarityScore')
            for residue in residues:
                f.write(',{}'.format(protein.residues[residue].resname))
            f.write('\n')
            CSIFP = ','.join(reference.IFP[i:i+bitstr_length] for i in range(0, len(reference.IFP), bitstr_length))
            f.write('{},,{}\n'.format(args.reference, CSIFP))
            for ligand in ligandList:
                CSIFP = ','.join(ligand.IFP[i:i+bitstr_length] for i in range(0, len(ligand.IFP), bitstr_length))
                f.write('{},{:.3f},{}\n'.format(ligand.inputFile, ligand.score, CSIFP))
