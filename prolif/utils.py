import re, textwrap
from rdkit import Chem
from .logger import logger

periodic_table = Chem.GetPeriodicTable()


def get_resnumber(resname):
    pattern = re.search(r'(\d+)', resname)
    return int(pattern.group(0))


def getCentroid(coordinates):
    """Centroid XYZ coordinates for an array of XYZ coordinates"""
    return [sum([c[i] for c in coordinates])/len(coordinates) for i in range(3)]


def getNormalVector(v):
    """Get a vector perpendicular to the input vector"""
    if (v.x == 0) and (v.y == 0):
        if v.z == 0:
            raise ValueError('Null vector has no normal vector')
        return [0,1,0]
    return [-v.y, v.x, 0]


def isinAngleLimits(angle, min_angle, max_angle):
    """Check if an angle value is between min and max angles in degrees"""
    if (min_angle <= angle <= max_angle) or (min_angle <= 180 - angle <= max_angle):
        return True
    return False


def get_mol2_records(lines):
    """Search for the MOLECULE, ATOM and BOND records of a mol2 file.
    Returns 3 lists of lines where the record starts"""
    mol_lines        = []
    first_atom_lines = []
    first_bond_lines = []

    for i, line in enumerate(lines):
        search_molecule = re.search(r'@<TRIPOS>MOLECULE', line)
        search_atom     = re.search(r'@<TRIPOS>ATOM', line)
        search_bond     = re.search(r'@<TRIPOS>BOND', line)

        if search_molecule:
            # line with the number of atoms and bonds
            mol_lines.append(i)
        elif search_atom:
            # first line with atom coordinates
            first_atom_lines.append(i)
        elif search_bond:
            # first line with bond information
            first_bond_lines.append(i)

    return mol_lines, first_atom_lines, first_bond_lines


def mol2_reader(mol2_file, ignoreH=False):
    '''A simple MOL2 file reader. Returns a list of residues.
    Each residue is an RDkit molecule'''
    molecules = []

    # Read file
    with open(mol2_file, "r") as f:
        lines = f.readlines()

    # Search for the molecule, atom and bonds records
    mol_lines, first_atom_lines, first_bond_lines = get_mol2_records(lines)

    for mol_line, first_atom_line, first_bond_line in zip(mol_lines, first_atom_lines, first_bond_lines):
        residues = get_residues_from_mol2(mol_line, first_atom_line, first_bond_line, lines, ignoreH)
        molecules.extend(residues)
    return molecules


def get_residues_from_mol2(mol_line, first_atom_line, first_bond_line, lines, ignoreH=False):
    """Extracts a molecule from a mol2 file.
    mol_line: index of the line containing the number of atoms, bonds...etc.
    first_atom_line: index of the first line of the ATOM record for the molecule to be extracted
    first_bond_line: Same for the BOND record
    Returns a MOL2 block of text coresponding to the molecule"""
    # Read number of atoms directly from the corresponding line
    data      = lines[mol_line+2].split()
    num_atoms = int(data[0])
    num_bonds = int(data[1])

    block = lines[mol_line : mol_line +first_atom_line-mol_line +num_atoms+1 +num_bonds+1]
    residues = residues_from_block(block, ignoreH)

    return residues


def residues_from_block(block, ignoreH=False):
    """Create residues as rdkit molecules.
    block: a MOL2 block of text with the MOLECULE, ATOM and BOND records
    Returns a list of residues as RDkit molecules"""

    residues = []
    # get the records lines
    mol_lines, first_atom_lines, first_bond_lines = get_mol2_records(block)
    mol_line, first_atom_line, first_bond_line = [item[0] for item in [mol_lines, first_atom_lines, first_bond_lines]]

    # read the molecule record
    data      = block[mol_line+2].split()
    num_atoms = int(data[0])
    num_bonds = int(data[1])

    # read the atom record to get the residues unique names
    resnames = set([line.split()[7] for line in block[first_atom_line+1:first_atom_line+num_atoms+1]])

    # Assign atoms to residues
    atoms  = {}
    map_id_to_res = {}
    for resname in resnames:
        atoms[resname] = []

    for line in block[first_atom_line+1:first_atom_line+num_atoms+1]:
        data = line.split()
        atom_id = data[0]
        resname = data[7]
        atoms[resname].append(line)
        map_id_to_res[atom_id] = resname

    # assign bonds to residues
    bonds = {}
    for resname in resnames:
        bonds[resname] = []

    for line in block[first_bond_line+1:first_bond_line+num_bonds+1]:
        data = line.split()
        atom1 = data[1]
        atom2 = data[2]
        # check if atom1 and atom2 belong to the same residue
        if map_id_to_res[atom1] == map_id_to_res[atom2]:
            resname = map_id_to_res[atom1]
            bonds[resname].append(line)

    # create residues as molecules
    for resname in resnames:
        # fix atom and bond numbering
        new_atoms = []
        new_bonds = []
        map_atoms = {}
        for i,line in enumerate(atoms[resname]):
            atom_id = line.split()[0]
            map_atoms[atom_id] = i+1
            new_line = str(i+1) + line[len(atom_id):]
            new_atoms.append(new_line)
        for i, line in enumerate(bonds[resname]):
            data = line.split()
            atom1 = data[1]
            atom2 = data[2]
            bond_type = data[3]
            new_line = '{} {} {} {}\n'.format(str(i+1), map_atoms[atom1], map_atoms[atom2], bond_type)
            new_bonds.append(new_line)

        residue_mol2_block = """\
@<TRIPOS>MOLECULE
{mol_name}
{num_atoms} {num_bonds}
SMALL
USER_CHARGES
@<TRIPOS>ATOM
{atoms}\
@<TRIPOS>BOND
{bonds}""".format(
    mol_name=resname,
    num_atoms=len(new_atoms),
    num_bonds=len(new_bonds),
    atoms=''.join(new_atoms),
    bonds=''.join(new_bonds),
    )
        residue = Chem.MolFromMol2Block(residue_mol2_block, sanitize=True, removeHs=ignoreH)
        if residue:
            residue.SetProp('resname', resname)
            residues.append(residue)
    return residues
