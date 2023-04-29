from ctypes import c_uint32
from threading import Event, Thread
from time import sleep

from multiprocess import Value
from multiprocess.pool import Pool
from rdkit import Chem
from tqdm.auto import tqdm

from prolif.molecule import Molecule


class Progress:
    """Tracks the number of frames processed by the :class:`TrajectoryPool`."""

    delay = 0.5

    def __init__(self, killswitch, tracker, *args, **kwargs):
        self.tqdm_pbar = tqdm(*args, **kwargs)
        self.killswitch = killswitch
        self.tracker = tracker

    def close(self):
        self.update()
        self.tqdm_pbar.close()

    def update(self):
        if self.tracker.value != 0:
            with self.tracker.get_lock():
                increment = self.tracker.value
                self.tracker.value = 0
            self.tqdm_pbar.update(increment)

    def __call__(self):
        while self.tqdm_pbar.n < self.tqdm_pbar.total:
            if self.killswitch.is_set():
                break
            self.update()
            sleep(self.delay)


class TrajectoryPool:
    """Pool of workers for a parallelized IFP analysis on an MD trajectory."""

    def __init__(
        self, n_processes, fingerprint, residues, tqdm_kwargs, rdkitconverter_kwargs
    ):
        self.tqdm_kwargs = tqdm_kwargs
        self.tracker = Value(c_uint32, lock=True)
        self.pool = Pool(
            n_processes,
            initializer=self.initializer,
            initargs=(
                self.tracker,
                fingerprint,
                residues,
                rdkitconverter_kwargs,
            ),
        )

    @classmethod
    def initializer(cls, tracker, fingerprint, residues, rdkitconverter_kwargs):
        cls.tracker = tracker
        cls.fp = fingerprint
        cls.residues = residues
        cls.converter_kwargs = rdkitconverter_kwargs

    @classmethod
    def executor(cls, args):
        traj, lig, prot, chunk = args
        ifp = {}
        for ts in traj[chunk]:
            lig_mol = Molecule.from_mda(lig, **cls.converter_kwargs[0])
            prot_mol = Molecule.from_mda(prot, **cls.converter_kwargs[1])
            data = cls.fp.generate(
                lig_mol, prot_mol, residues=cls.residues, metadata=True
            )
            ifp[ts.frame] = data
            with cls.tracker.get_lock():
                cls.tracker.value += 1
        return ifp

    def process(self, args_iterable):
        return self.pool.map(self.executor, args_iterable, chunksize=1)

    def __enter__(self):
        self.killswitch = Event()
        self.progress = Progress(self.killswitch, self.tracker, **self.tqdm_kwargs)
        self.progress_thread = Thread(target=self.progress)
        self.progress_thread.start()
        return self

    def __exit__(self, *exc):
        self.progress.close()
        self.killswitch.set()


class MolIterablePool:
    """Pool of workers for a parallelized IFP analysis on an iterable of ligands."""

    def __init__(self, n_processes, fingerprint, prot_mol, residues, tqdm_kwargs):
        self.tqdm_kwargs = tqdm_kwargs
        self.pool = Pool(
            n_processes,
            initializer=self.initializer,
            initargs=(fingerprint, prot_mol, residues),
        )

    @classmethod
    def initializer(cls, fingerprint, prot_mol, residues):
        cls.fp = fingerprint
        cls.pmol = prot_mol
        cls.residues = residues

    @classmethod
    def executor(cls, mol):
        return cls.fp.generate(mol, cls.pmol, residues=cls.residues, metadata=True)

    def process(self, args_iterable):
        results = self.pool.imap(self.executor, args_iterable, chunksize=1)
        return tqdm(results, **self.tqdm_kwargs)

    def __enter__(self):
        self.previous_pkl_props = Chem.GetDefaultPickleProperties()
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        return self

    def __exit__(self, *exc):
        Chem.SetDefaultPickleProperties(self.previous_pkl_props)
