import numpy as np
from MDAnalysis import Universe
from MDAnalysis.topology.guessers import guess_atom_element

from prolif.datafiles import datapath
from prolif.molecule import Molecule


def from_mol2(f):
    path = str(datapath / f)
    u = Universe(path)
    elements = [guess_atom_element(n) for n in u.atoms.names]
    u.add_TopologyAttr("elements", np.array(elements, dtype=object))
    u.atoms.types = np.array([x.upper() for x in u.atoms.types], dtype=object)
    return Molecule.from_mda(u, force=True)


def benzene():
    return from_mol2("benzene.mol2")


def cation():
    return from_mol2("cation.mol2")


def cation_false():
    return from_mol2("cation_false.mol2")


def anion():
    return from_mol2("anion.mol2")


def ftf():
    return from_mol2("facetoface.mol2")


def etf():
    return from_mol2("edgetoface.mol2")


def chlorine():
    return from_mol2("chlorine.mol2")


def bromine():
    return from_mol2("bromine.mol2")


def hb_donor():
    return from_mol2("donor.mol2")


def hb_acceptor():
    return from_mol2("acceptor.mol2")


def hb_acceptor_false():
    return from_mol2("acceptor_false.mol2")


def xb_donor():
    return from_mol2("xbond_donor.mol2")


def xb_acceptor():
    return from_mol2("xbond_acceptor.mol2")


def xb_acceptor_false_xar():
    return from_mol2("xbond_acceptor_false_xar.mol2")


def xb_acceptor_false_axd():
    return from_mol2("xbond_acceptor_false_axd.mol2")


def ligand():
    return from_mol2("ligand.mol2")


def metal():
    return from_mol2("metal.mol2")


def metal_false():
    return from_mol2("metal_false.mol2")
