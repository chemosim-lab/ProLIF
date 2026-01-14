"""
Generating an IFP in parallel --- :mod:`prolif.parallel`
========================================================

This module provides two classes, :class:`TrajectoryPool` and :class:`MolIterablePool`
to execute the analysis in parallel. These are used in the parallel implementation used
in :meth:`~prolif.fingerprint.Fingerprint.run` and
:meth:`~prolif.fingerprint.Fingerprint.run_from_iterable` respectively.
"""

import os
from collections.abc import Iterable
from ctypes import c_uint32
from threading import Event, Thread
from time import sleep
from typing import TYPE_CHECKING, Any, ClassVar, cast

import psutil
from multiprocess import Value
from multiprocess.pool import Pool
from tqdm.auto import tqdm

from prolif.molecule import Molecule
from prolif.pickling import PICKLE_HANDLER

if TYPE_CHECKING:
    from multiprocessing.pool import Pool as BuiltinPool
    from multiprocessing.sharedctypes import Synchronized

    from numpy.typing import NDArray

    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP
    from prolif.typeshed import IFPResults, MDAObject, ResidueSelection, Trajectory

PROLIF_MAX_JOBS = int(os.getenv("PROLIF_MAX_JOBS", "10"))
"""
Limits the max number of processes (unless the number of jobs is specified by the
user directly) to avoid oversubscription as IO tends to be the bottleneck.
"""


def get_n_jobs(n_jobs: int | None = None) -> int | None:
    """Get the number of parallel jobs to use.

    Prioritizes the ``n_jobs`` parameter, then the ``PROLIF_N_JOBS`` environment
    variable, then the minimum between the number of logical cores and
    :const:`PROLIF_MAX_JOBS` (8 by default), finally ``None`` if
    :func:`psutil.cpu_count` couldn't retrieve the number of logical cores.
    """
    if n_jobs is not None:
        if n_jobs < 1:
            raise ValueError("n_jobs must be > 0 or None")
        return n_jobs
    if env_n_jobs := os.getenv("PROLIF_N_JOBS"):
        return int(env_n_jobs)
    if n_logical_cores := psutil.cpu_count():
        return min(n_logical_cores, PROLIF_MAX_JOBS)
    return None


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

    delay: ClassVar[float] = 0.5

    def __init__(
        self, killswitch: Event, tracker: "Synchronized", **kwargs: Any
    ) -> None:
        self.tqdm_pbar = tqdm(**kwargs)
        self.killswitch = killswitch
        self.tracker = tracker

    def close(self) -> None:
        """Cleanup after the :class:`TrajectoryPool` computation is done"""
        self.update()
        self.tqdm_pbar.close()

    def update(self) -> None:
        """Update the value displayed by the progress bar"""
        if self.tracker.value != 0:
            with self.tracker.get_lock():
                increment = self.tracker.value
                self.tracker.value = 0
            self.tqdm_pbar.update(increment)

    def event_loop(self) -> None:
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
    n_processes : int | None
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
    use_segid: bool
        Use the segment number rather than the chain identifier as a chain.

    Attributes
    ----------
    tracker : multiprocess.Value
        Value holding a :class:`ctypes.c_uint32` ctype storing how many frames were
        processed since the last progress bar update.
    pool : multiprocess.pool.Pool
        The underlying pool instance.

    .. versionchanged:: 2.1.0
        Added `use_segid`.
    """

    tracker: ClassVar["Synchronized"] = cast("Synchronized", Value(c_uint32, lock=True))
    fp: ClassVar["Fingerprint"]
    residues: ClassVar["ResidueSelection"]
    converter_kwargs: ClassVar[tuple[dict, dict]]
    use_segid: bool

    def __init__(
        self,
        n_processes: int | None,
        fingerprint: "Fingerprint",
        residues: "ResidueSelection",
        tqdm_kwargs: dict,
        rdkitconverter_kwargs: tuple[dict, dict],
        use_segid: bool,
    ) -> None:
        self.tqdm_kwargs = tqdm_kwargs
        self.pool = cast(
            "BuiltinPool",
            Pool(
                n_processes,
                initializer=self.initializer,
                initargs=(
                    self.tracker,
                    fingerprint,
                    residues,
                    rdkitconverter_kwargs,
                    use_segid,
                ),
            ),
        )

    @classmethod
    def initializer(
        cls,
        tracker: "Synchronized",
        fingerprint: "Fingerprint",
        residues: "ResidueSelection",
        rdkitconverter_kwargs: tuple[dict, dict],
        use_segid: bool,
    ) -> None:
        """Initializer classmethod passed to the pool so that each child process can
        access these objects without copying them."""
        cls.tracker = tracker
        cls.fp = fingerprint
        cls.residues = residues
        cls.converter_kwargs = rdkitconverter_kwargs
        cls.use_segid = use_segid

    @classmethod
    def executor(
        cls, args: tuple["Trajectory", "MDAObject", "MDAObject", "NDArray"]
    ) -> "IFPResults":
        """Classmethod executed by each child process on a chunk of the trajectory

        Returns
        -------
        ifp_chunk: dict[int, prolif.ifp.IFP]
            A dictionary of :class:`~prolif.ifp.IFP` indexed by frame number
        """
        traj, lig, prot, chunk = args
        ifp: "IFPResults" = {}
        for ts in traj[chunk]:
            lig_mol = Molecule.from_mda(
                lig, use_segid=cls.use_segid, **cls.converter_kwargs[0]
            )
            prot_mol = Molecule.from_mda(
                prot, use_segid=cls.use_segid, **cls.converter_kwargs[1]
            )
            data = cls.fp.generate(
                lig_mol,
                prot_mol,
                residues=cls.residues,
                metadata=True,
            )
            ifp[int(ts.frame)] = data
            with cls.tracker.get_lock():
                cls.tracker.value += 1
        return ifp

    def process(
        self,
        args_iterable: Iterable[
            tuple["Trajectory", "MDAObject", "MDAObject", "NDArray"]
        ],
    ) -> list["IFPResults"]:
        """Maps the input iterable of arguments to the executor function.

        Parameters
        ----------
        args_iterable : typing.Iterable[tuple]
            Iterable of tuple of trajectory, ligand atomgroup, protein atomgroup, and
            array of frame indices.

        Returns
        -------
        ifp: list[dict[int, prolif.ifp.IFP]]
            An iterable of dictionaries of :class:`~prolif.ifp.IFP` indexed by frame
            number.
        """
        return self.pool.map(self.executor, args_iterable, chunksize=1)

    def __enter__(self) -> "TrajectoryPool":
        """Sets up the :class:`Progress` instance and associated killswitch event, and
        starts the progress event loop in a separate thread."""
        self.killswitch = Event()
        self.tracker.value = 0
        self.progress = Progress(self.killswitch, self.tracker, **self.tqdm_kwargs)
        self.progress_thread = Thread(target=self.progress.event_loop)
        self.progress_thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Call the killswitch and close the progress."""
        self.killswitch.set()
        self.progress.close()


class MolIterablePool:
    """Process pool for a parallelized IFP analysis on an iterable of ligands. Must be
    used in a ``with`` statement.

    Parameters
    ----------
    n_processes : int | None
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

    fp: ClassVar["Fingerprint"]
    pmol: ClassVar[Molecule]
    residues: ClassVar["ResidueSelection"]

    def __init__(
        self,
        n_processes: int | None,
        fingerprint: "Fingerprint",
        prot_mol: Molecule,
        residues: "ResidueSelection",
        tqdm_kwargs: dict,
    ) -> None:
        self.tqdm_kwargs = tqdm_kwargs
        self.pool = cast(
            "BuiltinPool",
            Pool(
                n_processes,
                initializer=self.initializer,
                initargs=(fingerprint, prot_mol, residues),
            ),
        )

    @classmethod
    def initializer(
        cls,
        fingerprint: "Fingerprint",
        prot_mol: Molecule,
        residues: "ResidueSelection",
    ) -> None:
        """Initializer classmethod passed to the pool so that each child process can
        access these objects without copying them."""
        cls.fp = fingerprint
        cls.pmol = prot_mol
        cls.residues = residues

    @classmethod
    def executor(cls, mol: Molecule) -> "IFP":
        """Classmethod executed by each child process on a single ligand molecule from
        the input iterable.

        Returns
        -------
        ifp_data : prolif.ifp.IFP
            A dictionary indexed by ``(ligand, protein)`` residue pairs, and each value
            is a sparse dictionary of metadata indexed by interaction name.
        """
        return cls.fp.generate(mol, cls.pmol, residues=cls.residues, metadata=True)

    def process(self, args_iterable: Iterable[Molecule]) -> Iterable["IFP"]:
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

    def __enter__(self) -> "MolIterablePool":
        """Sets up which properties will be pickled by RDKit by default"""
        PICKLE_HANDLER.set()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Resets RDKit's default pickled properties"""
        PICKLE_HANDLER.reset()


class TrajectoryPoolQueue:
    """Queue-based process pool for IFP analysis on an MD trajectory.

    This implementation uses a producer-consumer pattern where:
    - A producer thread iterates over the trajectory and converts frames to Molecules
    - Worker processes consume Molecule pairs from a queue and compute fingerprints
    - This avoids pickling MDAnalysis objects, which is the bottleneck in TrajectoryPool

    Must be used in a ``with`` statement.

    Parameters
    ----------
    n_processes : int | None
        Max number of worker processes
    fingerprint : prolif.fingerprint.Fingerprint
        Fingerprint instance used to generate the IFP
    residues : list, optional
        List of protein residues considered for the IFP
    tqdm_kwargs : dict
        Parameters for the :class:`~tqdm.std.tqdm` progress bar
    rdkitconverter_kwargs : tuple[dict, dict]
        Parameters for the :class:`~MDAnalysis.converters.RDKit.RDKitConverter`
        from MDAnalysis: the first for the ligand, and the second for the protein
    use_segid: bool
        Use the segment number rather than the chain identifier as a chain.

    .. versionadded:: 2.2.0
    """

    fp: ClassVar["Fingerprint"]
    residues: ClassVar["ResidueSelection"]

    def __init__(
        self,
        n_processes: int | None,
        fingerprint: "Fingerprint",
        residues: "ResidueSelection",
        tqdm_kwargs: dict,
        rdkitconverter_kwargs: tuple[dict, dict],
        use_segid: bool,
    ) -> None:
        self.n_processes = n_processes
        self.tqdm_kwargs = tqdm_kwargs
        self.converter_kwargs = rdkitconverter_kwargs
        self.use_segid = use_segid
        self.pool = cast(
            "BuiltinPool",
            Pool(
                n_processes,
                initializer=self.initializer,
                initargs=(fingerprint, residues),
            ),
        )

    @classmethod
    def initializer(
        cls,
        fingerprint: "Fingerprint",
        residues: "ResidueSelection",
    ) -> None:
        """Initializer classmethod passed to the pool so that each child process can
        access these objects without copying them."""
        cls.fp = fingerprint
        cls.residues = residues

    @classmethod
    def executor(cls, args: tuple[int, Molecule, Molecule]) -> tuple[int, "IFP"]:
        """Classmethod executed by each child process on a single frame.

        Parameters
        ----------
        args : tuple[int, Molecule, Molecule]
            Tuple of (frame_number, ligand_mol, protein_mol)

        Returns
        -------
        result : tuple[int, prolif.ifp.IFP]
            Tuple of (frame_number, IFP data)
        """
        frame, lig_mol, prot_mol = args
        data = cls.fp.generate(
            lig_mol,
            prot_mol,
            residues=cls.residues,
            metadata=True,
        )
        return frame, data

    def process(
        self,
        traj: "Trajectory",
        lig: "MDAObject",
        prot: "MDAObject",
    ) -> "IFPResults":
        """Process the trajectory using a producer-consumer pattern.

        Parameters
        ----------
        traj : Trajectory
            MDAnalysis trajectory or sliced trajectory
        lig : MDAObject
            Ligand AtomGroup
        prot : MDAObject
            Protein AtomGroup

        Returns
        -------
        ifp : dict[int, prolif.ifp.IFP]
            A dictionary of :class:`~prolif.ifp.IFP` indexed by frame number
        """

        def frame_generator() -> Iterable[tuple[int, Molecule, Molecule]]:
            """Generator that yields (frame, lig_mol, prot_mol) tuples."""
            for ts in traj:
                lig_mol = Molecule.from_mda(
                    lig, use_segid=self.use_segid, **self.converter_kwargs[0]
                )
                prot_mol = Molecule.from_mda(
                    prot, use_segid=self.use_segid, **self.converter_kwargs[1]
                )
                yield int(ts.frame), lig_mol, prot_mol

        ifp: "IFPResults" = {}
        # Use imap to stream results as they complete
        results = self.pool.imap_unordered(self.executor, frame_generator())
        pbar = tqdm(results, **self.tqdm_kwargs)

        for frame, data in pbar:
            ifp[frame] = data

        return ifp

    def __enter__(self) -> "TrajectoryPoolQueue":
        """Sets up RDKit pickle properties."""
        PICKLE_HANDLER.set()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Cleanup pool and reset RDKit pickle properties."""
        self.pool.close()
        self.pool.join()
        PICKLE_HANDLER.reset()
