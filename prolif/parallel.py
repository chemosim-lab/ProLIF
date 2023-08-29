"""
Generating an IFP in parallel --- :mod:`prolif.parallel`
========================================================

This module provides two classes, :class:`TrajectoryPool` and :class:`MolIterablePool`
to execute the analysis in parallel. These are used in the parallel implementation used
in :meth:`~prolif.fingerprint.Fingerprint.run` and
:meth:`~prolif.fingerprint.Fingerprint.run_from_iterable` respectively.
"""
from ctypes import c_uint32
from threading import Event, Thread
from time import sleep

from multiprocess import Value
from multiprocess.pool import Pool
from tqdm.auto import tqdm

from prolif.molecule import Molecule
from prolif.pickling import PICKLE_HANDLER


class Progress:
    """Helper class to track the number of frames processed by the
    :class:`TrajectoryPool`.

    Parameters
    ----------
    killswitch : threading.Event
        A threading.Event instance created and controlled by the :class:`TrajectoryPool`
        to kill the thread updating the :class:`~tqdm.std.tqdm` progress bar.
    tracker : multiprocess.Value
        Value holding a :class:`ctypes.c_uint32` ctype updated by the
        :class:`TrajectoryPool`, storing how many frames were processed since the last
        progress bar update.
    **kwargs : object
        Used to create the :class:`~tqdm.std.tqdm` progress bar.

    Attributes
    ----------
    delay : float, default = 0.5
        Delay between progress bar updates. This requires locking access to the
        ``tracker`` object which the :class:`TrajectoryPool` needs access to, so too
        small values might cause delays in the analysis.
    tqdm_pbar : tqdm.std.tqdm
        The progress bar displayed
    """

    delay = 0.5

    def __init__(self, killswitch, tracker, **kwargs):
        self.tqdm_pbar = tqdm(**kwargs)
        self.killswitch = killswitch
        self.tracker = tracker

    def close(self):
        """Cleanup after the :class:`TrajectoryPool` computation is done"""
        self.update()
        self.tqdm_pbar.close()

    def update(self):
        """Update the value displayed by the progress bar"""
        if self.tracker.value != 0:
            with self.tracker.get_lock():
                increment = self.tracker.value
                self.tracker.value = 0
            self.tqdm_pbar.update(increment)

    def event_loop(self):
        """Event loop targeted by a separate thread"""
        while True:
            if self.killswitch.is_set():
                break
            self.update()
            sleep(self.delay)


class TrajectoryPool:
    """Process pool for a parallelized IFP analysis on an MD trajectory. Must be used
    in a ``with`` statement.

    Parameters
    ----------
    n_processes : int
        Max number of processes
    fingerprint : prolif.fingerprint.Fingerprint
        Fingerprint instance used to generate the IFP
    residues : list, optional
        List of protein residues considered for the IFP
    tqdm_kwargs : dict
        Parameters for the :class:`~tqdm.std.tqdm` progress bar
    rdkitconverter_kwargs : tuple[dict, dict]
        Parameters for the :class:`~MDAnalysis.converters.RDKit.RDKitConverter`
        from MDAnalysis: the first for the ligand, and the second for the protein

    Attributes
    ----------
    tracker : multiprocess.Value
        Value holding a :class:`ctypes.c_uint32` ctype storing how many frames were
        processed since the last progress bar update.
    pool : multiprocess.pool.Pool
        The underlying pool instance.
    """

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
        """Initializer classmethod passed to the pool so that each child process can
        access these objects without copying them."""
        cls.tracker = tracker
        cls.fp = fingerprint
        cls.residues = residues
        cls.converter_kwargs = rdkitconverter_kwargs

    @classmethod
    def executor(cls, args):
        """Classmethod executed by each child process on a chunk of the trajectory

        Returns
        -------
        ifp_chunk: dict[int, prolif.ifp.IFP]
            A dictionary of :class:`~prolif.ifp.IFP` indexed by frame number
        """
        traj, lig, prot, chunk = args
        ifp = {}
        for ts in traj[chunk]:
            lig_mol = Molecule.from_mda(lig, **cls.converter_kwargs[0])
            prot_mol = Molecule.from_mda(prot, **cls.converter_kwargs[1])
            data = cls.fp.generate(
                lig_mol, prot_mol, residues=cls.residues, metadata=True
            )
            ifp[int(ts.frame)] = data
            with cls.tracker.get_lock():
                cls.tracker.value += 1
        return ifp

    def process(self, args_iterable):
        """Maps the input iterable of arguments to the executor function.

        Parameters
        ----------
        args_iterable : typing.Iterable[tuple]
            Iterable of tuple of trajectory, ligand atomgroup, protein atomgroup, and
            array of frame indices.

        Returns
        -------
        ifp: typing.Iterable[dict[int, prolif.ifp.IFP]]
            An iterable of dictionaries of :class:`~prolif.ifp.IFP` indexed by frame
            number.
        """
        return self.pool.map(self.executor, args_iterable, chunksize=1)

    def __enter__(self):
        """Sets up the :class:`Progress` instance and associated killswitch event, and
        starts the progress event loop in a separate thread."""
        self.killswitch = Event()
        self.progress = Progress(self.killswitch, self.tracker, **self.tqdm_kwargs)
        self.progress_thread = Thread(target=self.progress.event_loop)
        self.progress_thread.start()
        return self

    def __exit__(self, *exc):
        """Call the killswitch and close the progress."""
        self.killswitch.set()
        self.progress.close()


class MolIterablePool:
    """Process pool for a parallelized IFP analysis on an iterable of ligands. Must be
    used in a ``with`` statement.

    Parameters
    ----------
    n_processes : int
        Max number of processes
    fingerprint : prolif.fingerprint.Fingerprint
        Fingerprint instance used to generate the IFP
    prot_mol : prolif.molecule.Molecule
        Protein molecule
    residues : list, optional
        List of protein residues considered for the IFP
    tqdm_kwargs : dict
        Parameters for the :class:`~tqdm.std.tqdm` progress bar

    Attributes
    ----------
    pool : multiprocess.pool.Pool
        The underlying pool instance.
    """

    def __init__(self, n_processes, fingerprint, prot_mol, residues, tqdm_kwargs):
        self.tqdm_kwargs = tqdm_kwargs
        self.pool = Pool(
            n_processes,
            initializer=self.initializer,
            initargs=(fingerprint, prot_mol, residues),
        )

    @classmethod
    def initializer(cls, fingerprint, prot_mol, residues):
        """Initializer classmethod passed to the pool so that each child process can
        access these objects without copying them."""
        cls.fp = fingerprint
        cls.pmol = prot_mol
        cls.residues = residues

    @classmethod
    def executor(cls, mol):
        """Classmethod executed by each child process on a single ligand molecule from
        the input iterable.

        Returns
        -------
        ifp_data : prolif.ifp.IFP
            A dictionary indexed by ``(ligand, protein)`` residue pairs, and each value
            is a sparse dictionary of metadata indexed by interaction name.
        """
        return cls.fp.generate(mol, cls.pmol, residues=cls.residues, metadata=True)

    def process(self, args_iterable):
        """Maps the input iterable of molecules to the executor function.

        Parameters
        ----------
        args_iterable : typing.Iterable[prolif.molecule.Molecule]
            An iterable yielding ligand molecules

        Returns
        -------
        ifp : typing.Iterable[prolif.ifp.IFP]
            An iterable of :class:`~prolif.ifp.IFP` dictionaries.
        """
        results = self.pool.imap(self.executor, args_iterable, chunksize=1)
        return tqdm(results, **self.tqdm_kwargs)

    def __enter__(self):
        """Sets up which properties will be pickled by RDKit by default"""
        PICKLE_HANDLER.set()
        return self

    def __exit__(self, *exc):
        """Resets RDKit's default pickled properties"""
        PICKLE_HANDLER.reset()
