import multiprocessing as mp
from ctypes import c_int32
from time import sleep

from tqdm.auto import tqdm

from .molecule import Molecule


def process_chunk(args):
    """Generates a fingerprint for a chunk of frame indices"""
    traj, lig, prot, chunk = args
    ifp = []
    for ts in traj[chunk]:
        lig_mol = Molecule.from_mda(lig)
        prot_mol = Molecule.from_mda(prot)
        data = fp.generate(lig_mol, prot_mol, residues=residues,
                           return_atoms=True)
        data["Frame"] = ts.frame
        ifp.append(data)
        if display_progress:
            with pcount.lock:
                pcount.counter.value += 1
    return ifp


def declare_shared_objs_for_chunk(fingerprint, resid_list, show_progressbar,
                                  progress_counter):
    """Declares global objects that are available to the pool of workers for
    a trajectory"""
    global fp, residues, display_progress, pcount
    fp = fingerprint
    residues = resid_list
    display_progress = show_progressbar
    pcount = progress_counter


def process_mol(args):
    """Generates a fingerprint for a single molecule"""
    index, mol = args
    data = fp.generate(mol, prot_mol, residues=residues, return_atoms=True)
    data["Frame"] = index
    return data


def declare_shared_objs_for_mol(fingerprint, pmol, resid_list):
    """Declares global objects that are available to the pool of workers for
    an interable of ligands"""
    global fp, residues, prot_mol
    fp = fingerprint
    residues = resid_list
    prot_mol = pmol


class ProgressCounter:
    """Tracks the progress of the fingerprint analysis accross the pool of
    workers"""
    def __init__(self):
        self.lock = mp.Lock()
        self.counter = mp.Value(c_int32)


class Progress:
    """Handles tracking the progress of the ProgressCounter and updating the
    tqdm progress bar, from within an independent thread"""
    def __init__(self, pcount, *args, **kwargs):
        self.pbar = tqdm(*args, **kwargs)
        self.pcount = pcount

    def __call__(self):
        while self.pbar.n < self.pbar.total:
            if self.pcount.counter.value != 0:
                with self.pcount.lock:
                    n_processed = self.pcount.counter.value
                    self.pcount.counter.value = 0
                self.pbar.update(n_processed)
            sleep(1)
