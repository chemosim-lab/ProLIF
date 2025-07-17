"""
The below function is directly copied from pdbinf's _parse_altnames,
which parses the XML file containing alternative names for residues and
atoms in PDB files.

Source:
https://github.com/OpenFreeEnergy/pdbinf/blob/c0ddf00bd068d7860b2e99b9f03847c890e3efb5/src/pdbinf/_aliases.py#L294

"""

from xml.etree import ElementTree as ET


def parse_altnames(xml_string: str) -> tuple:
    residues = ET.fromstring(xml_string)

    resname_aliases = {}
    atomname_aliases = {}

    for res in residues:
        resname = res.attrib["name"]

        for k, alt_resname in res.attrib.items():
            if k.startswith("alt"):
                resname_aliases[alt_resname] = resname

        aliases = {}
        for atom in res:
            atomname = atom.attrib["name"]

            for k, alt_atomname in atom.attrib.items():
                if k.startswith("alt"):
                    aliases[alt_atomname] = atomname

        atomname_aliases[resname] = aliases

    return resname_aliases, atomname_aliases
