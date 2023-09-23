"""
Calculate a Protein-Ligand Interaction Fingerprint --- :mod:`prolif.fingerprint`
================================================================================

.. ipython:: python
    :okwarning:

    import MDAnalysis as mda
    from rdkit.DataStructs import TanimotoSimilarity
    import prolif
    u = mda.Universe(prolif.datafiles.TOP, prolif.datafiles.TRAJ)
    prot = u.select_atoms("protein")
    lig = u.select_atoms("resname LIG")
    fp = prolif.Fingerprint(["HBDonor", "HBAcceptor", "PiStacking", "CationPi",
                             "Cationic"])
    fp.run(u.trajectory[:5], lig, prot)
    df = fp.to_dataframe()
    df
    bv = fp.to_bitvectors()
    TanimotoSimilarity(bv[0], bv[1])
    # save results
    fp.to_pickle("fingerprint.pkl")
    # load
    fp = prolif.Fingerprint.from_pickle("fingerprint.pkl")

"""
import warnings
from collections.abc import Sized
from functools import wraps
from typing import Literal, Optional, Tuple

import dill
import multiprocess as mp
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm

from prolif.ifp import IFP
from prolif.interactions.base import _BASE_INTERACTIONS, _INTERACTIONS
from prolif.molecule import Molecule
from prolif.parallel import MolIterablePool, TrajectoryPool
from prolif.plotting.utils import IS_NOTEBOOK
from prolif.utils import (
    get_residues_near_ligand,
    to_bitvectors,
    to_countvectors,
    to_dataframe,
)


def first_occurence(interaction):
    @wraps(interaction)
    def wrapped(*args, **kwargs):
        return next(interaction(*args, **kwargs), None)

    return wrapped


def all_occurences(interaction):
    @wraps(interaction)
    def wrapped(*args, **kwargs):
        return tuple(interaction(*args, **kwargs))

    return wrapped


class Fingerprint:
    """Class that generates an interaction fingerprint between two molecules

    While in most cases the fingerprint will be generated between a ligand and
    a protein, it is also possible to use this class for protein-protein
    interactions, or host-guest systems.

    Parameters
    ----------
    interactions : list
        List of names (str) of interaction classes as found in the
        :mod:`prolif.interactions` module.
    parameters : dict, optional
        New parameters for the interactions. Mapping between an interaction name and a
        dict of parameters as they appear in the interaction class.
    count : bool
        For a given interaction class and pair of residues, there might be multiple
        combinations of atoms that satisfy the interaction constraints. This parameter
        controls how the fingerprint treats these different combinations:

        * ``False``: returns the first combination that satisfy the constraints (fast)
        * ``True``: returns all the combinations (slow)

    vicinity_cutoff : float
        Automatically restrict the analysis to residues within this range of the ligand.
        This parameter is ignored if the ``residues`` parameter of the ``run`` methods
        is set to anything other than ``None``.

    Attributes
    ----------
    interactions : dict
        Dictionary of interaction classes indexed by class name. For more
        details, see :mod:`prolif.interactions`
    n_interactions : int
        Number of interaction functions registered by the fingerprint
    vicinity_cutoff : float
        Used when calling :func:`prolif.utils.get_residues_near_ligand`.
    count : bool
        Whether to keep track of all interaction occurences or just the first one.
    ifp : dict, optional
        Dict of interaction fingerprints in a sparse format for the given trajectory or
        docking poses: ``{<frame number>: <IFP>}``. See the :class:`~prolif.ifp.IFP`
        class for more information.

    Raises
    ------
    NameError
        Unknown interaction in the ``interactions`` or ``parameters`` parameters.

    Notes
    -----
    You can use the fingerprint generator in multiple ways:

    - On a trajectory directly from MDAnalysis objects:

    .. ipython:: python

        prot = u.select_atoms("protein")
        lig = u.select_atoms("resname LIG")
        fp = prolif.Fingerprint(["HBDonor", "HBAcceptor", "PiStacking",
                                 "Hydrophobic"])
        fp.run(u.trajectory[:5], lig, prot)
        fp.to_dataframe()

    - On two single structures (from RDKit or MDAnalysis):

    .. ipython:: python

        u.trajectory[0]  # use coordinates of the first frame
        prot = prolif.Molecule.from_mda(prot)
        lig = prolif.Molecule.from_mda(lig)
        ifp = fp.generate(lig, prot, metadata=True)
        prolif.to_dataframe({0: ifp}, fp.interactions)

    - On a specific pair of residues for a specific interaction:

    .. ipython:: python

        # ligand-protein
        next(fp.hbdonor(lig, prot["ASP129.A"]))
        # protein-protein
        next(fp.hbacceptor(prot["ASP129.A"], prot["CYS133.A"]))

    You can also obtain the indices of atoms responsible for the interaction:

    .. ipython:: python

        fp.metadata(lig, prot["ASP129.A"])
        next(fp.hbdonor(lig, prot["ASP129.A"], metadata=True), None)


    .. versionchanged:: 1.0.0
        Added pickle support

    .. versionchanged:: 2.0.0
        Changed the format of the :attr:`~Fingerprint.ifp` attribute to be a dictionary
        containing more complete interaction metadata instead of just atom indices.
        Removed the ``return_atoms`` argument in :meth:`~Fingerprint.to_dataframe`.
        Users should directly use :attr:`~Fingerprint.ifp` instead.
        Added the :meth:`~Fingerprint.plot_lignetwork` method to generate the
        :class:`~prolif.plotting.network.LigNetwork` plot.
        Replaced the ``Fingerprint.bitvector_atoms`` method with
        :meth:`Fingerprint.metadata`.
        Added a ``vicinity_cutoff`` parameter controlling the distance used
        to automatically restrict the IFP calculation to residues within the specified
        range of the ligand.
        Removed the ``__wrapped__`` attribute on interaction methods that are available
        from the fingerprint object. These methods now accept a ``metadata`` parameter
        instead.
        Added the ``parameters`` argument to set the interaction classes parameters
        without having to create a new class.
        Added the ``count`` argument to control for a given pair of residues whether to
        return all occurences of an interaction or only the first one.

    """

    def __init__(
        self,
        interactions=[
            "Hydrophobic",
            "HBDonor",
            "HBAcceptor",
            "PiStacking",
            "Anionic",
            "Cationic",
            "CationPi",
            "PiCation",
            "VdWContact",
        ],
        parameters=None,
        count=False,
        vicinity_cutoff=6.0,
    ):
        self.count = count
        self._set_interactions(interactions, parameters)
        self.vicinity_cutoff = vicinity_cutoff

    def _set_interactions(self, interactions, parameters):
        # read interactions to compute
        parameters = parameters or {}
        if interactions == "all":
            interactions = self.list_available()
        # sanity check
        self._check_valid_interactions(interactions, "interactions")
        self._check_valid_interactions(parameters, "parameters")
        # add interaction methods
        self.interactions = {}
        wrapper = all_occurences if self.count else first_occurence
        for name, interaction_cls in _INTERACTIONS.items():
            # create instance with custom parameters if available
            interaction = interaction_cls(**parameters.get(name, {}))
            setattr(self, name.lower(), interaction)
            if name in interactions:
                self.interactions[name] = wrapper(interaction)

    def _check_valid_interactions(self, interactions_iterable, varname):
        """Raises a NameError if an unknown interaction is given."""
        unsafe = set(interactions_iterable)
        unknown = unsafe.symmetric_difference(_INTERACTIONS.keys()) & unsafe
        if unknown:
            raise NameError(
                f"Unknown interaction(s) in {varname!r}: {', '.join(unknown)}"
            )

    def __repr__(self):  # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_interactions} interactions: {list(self.interactions.keys())}"
        return f"<{name}: {params} at {id(self):#x}>"

    @staticmethod
    def list_available(show_hidden=False):
        """List interactions available to the Fingerprint class.

        Parameters
        ----------
        show_hidden : bool
            Show hidden classes (base classes meant to be inherited from to create
            custom interactions).
        """
        if show_hidden:
            return sorted(_BASE_INTERACTIONS) + sorted(_INTERACTIONS)
        return sorted(_INTERACTIONS)

    @property
    def n_interactions(self):
        return len(self.interactions)

    def bitvector(self, res1, res2):
        """Generates the complete bitvector for the interactions between two
        residues. To access metadata for each interaction, see
        :meth:`~Fingerprint.metadata`.

        Parameters
        ----------
        res1 : prolif.residue.Residue
            A residue, usually from a ligand
        res2 : prolif.residue.Residue
            A residue, usually from a protein

        Returns
        -------
        bitvector : numpy.ndarray
            An array storing the encoded interactions between res1 and res2. Depending
            on :attr:`Fingerprint.count`, the dtype of the array will be bool or uint8.
        """
        bitvector = [
            interaction(res1, res2) for interaction in self.interactions.values()
        ]
        if self.count:
            return np.array([sum(bits) for bits in bitvector], dtype=np.uint8)
        return np.array(bitvector, dtype=bool)

    def metadata(self, res1, res2):
        """Generates a metadata dictionary for the interactions between two residues.

        Parameters
        ----------
        res1 : prolif.residue.Residue
            A residue, usually from a ligand
        res2 : prolif.residue.Residue
            A residue, usually from a protein

        Returns
        -------
        metadata : dict[str, tuple[dict, ...]]
            Dict containing tuples of metadata dictionaries indexed by interaction
            name. If a specific interaction is not present between residues, it is
            filtered out of the dictionary.


        .. versionchanged:: 0.3.2
            Atom indices are returned as two separate lists instead of a single
            list of tuples

        .. versionchanged:: 2.0.0
            Returns a dictionnary with all available metadata for each interaction,
            rather than just atom indices.

        """
        return {
            name: metadata if self.count else (metadata,)
            for name, interaction in self.interactions.items()
            if (metadata := interaction(res1, res2, metadata=True))
        }

    def generate(self, lig, prot, residues=None, metadata=False):
        """Generates the interaction fingerprint between 2 molecules

        Parameters
        ----------
        lig : prolif.molecule.Molecule
            Molecule for the ligand
        prot : prolif.molecule.Molecule
            Molecule for the protein
        residues : list or "all" or None
            A list of protein residues (:class:`str`, :class:`int` or
            :class:`~prolif.residue.ResidueId`) to take into account for
            the fingerprint extraction. If ``"all"``, all residues will be
            used. If ``None``, at each frame the
            :func:`~prolif.utils.get_residues_near_ligand` function is used to
            automatically use protein residues that are distant of 6.0 Å or
            less from each ligand residue (see :attr:`~Fingerprint.vicinity_cutoff`)
        metadata : bool
            For each residue pair and interaction, return an interaction metadata
            dictionary instead of bits.

        Returns
        -------
        ifp : prolif.ifp.IFP
            A dictionary indexed by ``(ligand, protein)`` residue pairs. The
            format for values will depend on ``metadata``:

            - A single numpy array if ``metadata=False``
            - A sparse dictionary of metadata tuples indexed by interaction name if
              ``metadata=True``

        Example
        -------
        ::

            >>> u = mda.Universe("complex.pdb")
            >>> lig = prolif.Molecule.from_mda(u, "resname LIG")
            >>> prot = prolif.Molecule.from_mda(u, "protein")
            >>> fp = prolif.Fingerprint()
            >>> ifp = fp.generate(lig, prot)

        .. versionadded:: 0.3.2

        .. versionchanged:: 2.0.0
            ``return_atoms`` replaced by ``metadata``, and it now returns a sparse
            dictionary of metadata tuples indexed by interaction name instead of a
            tuple of arrays.
        """
        ifp = IFP()
        prot_residues = prot.residues if residues == "all" else residues
        get_interactions = self.metadata if metadata else self.bitvector
        for lresid, lres in lig.residues.items():
            if residues is None:
                prot_residues = get_residues_near_ligand(
                    lres, prot, self.vicinity_cutoff
                )
            for prot_key in prot_residues:
                pres = prot[prot_key]
                key = (lresid, pres.resid)
                interactions = get_interactions(lres, pres)
                if any(interactions):
                    ifp[key] = interactions
        return ifp

    def run(
        self,
        traj,
        lig,
        prot,
        *,
        residues=None,
        converter_kwargs=None,
        progress=True,
        n_jobs=None,
    ):
        """Generates the fingerprint on a trajectory for a ligand and a protein

        Parameters
        ----------
        traj : MDAnalysis.coordinates.base.ProtoReader or MDAnalysis.coordinates.base.FrameIteratorSliced
            Iterate over this Universe trajectory or sliced trajectory object
            to extract the frames used for the fingerprint extraction
        lig : MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the ligand
        prot : MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the protein (with multiple residues)
        residues : list or "all" or None
            A list of protein residues (:class:`str`, :class:`int` or
            :class:`~prolif.residue.ResidueId`) to take into account for
            the fingerprint extraction. If ``"all"``, all residues will be
            used. If ``None``, at each frame the
            :func:`~prolif.utils.get_residues_near_ligand` function is used to
            automatically use protein residues that are distant of 6.0 Å or
            less from each ligand residue.
        converter_kwargs : tuple[dict, dict], optional
            Tuple of kwargs passed to the underlying :class:`~MDAnalysis.converters.RDKit.RDKitConverter`
            from MDAnalysis: the first for the ligand, and the second for the protein
        progress : bool
            Display a :class:`~tqdm.std.tqdm` progressbar while running the calculation
        n_jobs : int or None
            Number of processes to run in parallel. If ``n_jobs=None``, the
            analysis will use all available CPU threads, while if ``n_jobs=1``,
            the analysis will run in serial.

        Raises
        ------
        ValueError
            If ``n_jobs <= 0``

        Returns
        -------
        prolif.fingerprint.Fingerprint
            The Fingerprint instance that generated the fingerprint

        Example
        -------
        ::

            >>> u = mda.Universe("top.pdb", "traj.nc")
            >>> lig = u.select_atoms("resname LIG")
            >>> prot = u.select_atoms("protein")
            >>> fp = prolif.Fingerprint().run(u.trajectory[:10], lig, prot)

        .. seealso::

            - :meth:`Fingerprint.generate` to generate the fingerprint between
              two single structures.
            - :meth:`Fingerprint.run_from_iterable` to generate the fingerprint
              between a protein and a collection of ligands.

        .. versionchanged:: 0.3.2
            Moved the ``return_atoms`` parameter from the ``run`` method to the
            dataframe conversion code

        .. versionchanged:: 1.0.0
            Added support for multiprocessing

        .. versionchanged:: 1.1.0
            Added support for passing kwargs to the RDKitConverter through
            the ``converter_kwargs`` parameter

        .. versionchanged:: 2.0.0
            Changed the format of the :attr:`~Fingerprint.ifp` attribute to be a
            dictionary containing more complete interaction metadata instead of just
            atom indices.

        """
        if n_jobs is not None and n_jobs < 1:
            raise ValueError("n_jobs must be > 0 or None")
        if converter_kwargs is not None and len(converter_kwargs) != 2:
            raise ValueError("converter_kwargs must be a list of 2 dicts")

        converter_kwargs = converter_kwargs or ({}, {})
        if residues == "all":
            residues = list(Molecule.from_mda(prot, **converter_kwargs[1]).residues)
        if n_jobs != 1:
            return self._run_parallel(
                traj,
                lig,
                prot,
                residues=residues,
                converter_kwargs=converter_kwargs,
                progress=progress,
                n_jobs=n_jobs,
            )

        iterator = tqdm(traj) if progress else traj
        ifp = {}
        for ts in iterator:
            lig_mol = Molecule.from_mda(lig, **converter_kwargs[0])
            prot_mol = Molecule.from_mda(prot, **converter_kwargs[1])
            ifp[int(ts.frame)] = self.generate(
                lig_mol, prot_mol, residues=residues, metadata=True
            )
        self.ifp = ifp
        return self

    def _run_parallel(
        self,
        traj,
        lig,
        prot,
        residues=None,
        converter_kwargs=None,
        progress=True,
        n_jobs=None,
    ):
        """Parallel implementation of :meth:`~Fingerprint.run`"""
        n_chunks = n_jobs if n_jobs else mp.cpu_count()
        try:
            n_frames = traj.n_frames
        except AttributeError:
            # sliced trajectory
            frames = range(traj.start, traj.stop, traj.step)
            traj = lig.universe.trajectory
        else:
            frames = range(n_frames)
        chunks = np.array_split(frames, n_chunks)
        args_iterable = [(traj, lig, prot, chunk) for chunk in chunks]
        ifp = {}

        with TrajectoryPool(
            n_jobs,
            fingerprint=self,
            residues=residues,
            tqdm_kwargs={"total": len(frames), "disable": not progress},
            rdkitconverter_kwargs=converter_kwargs,
        ) as pool:
            for ifp_data_chunk in pool.process(args_iterable):
                ifp.update(ifp_data_chunk)

        self.ifp = ifp
        return self

    def run_from_iterable(
        self, lig_iterable, prot_mol, *, residues=None, progress=True, n_jobs=None
    ):
        """Generates the fingerprint between a list of ligands and a protein

        Parameters
        ----------
        lig_iterable : list or generator
            An iterable yielding ligands as :class:`~prolif.molecule.Molecule`
            objects
        prot_mol : prolif.molecule.Molecule
            The protein
        residues : list or "all" or None
            A list of protein residues (:class:`str`, :class:`int` or
            :class:`~prolif.residue.ResidueId`) to take into account for
            the fingerprint extraction. If ``"all"``, all residues will be
            used. If ``None``, at each frame the
            :func:`~prolif.utils.get_residues_near_ligand` function is used to
            automatically use protein residues that are distant of 6.0 Å or
            less from each ligand residue.
        progress : bool
            Display a :class:`~tqdm.std.tqdm` progressbar while running the calculation
        n_jobs : int or None
            Number of processes to run in parallel. If ``n_jobs=None``, the
            analysis will use all available CPU threads, while if ``n_jobs=1``,
            the analysis will run in serial.

        Raises
        ------
        ValueError
            If ``n_jobs <= 0``

        Returns
        -------
        prolif.fingerprint.Fingerprint
            The Fingerprint instance that generated the fingerprint

        Example
        -------
        ::

            >>> prot = mda.Universe("protein.pdb")
            >>> prot = prolif.Molecule.from_mda(prot)
            >>> lig_iter = prolif.mol2_supplier("docking_output.mol2")
            >>> fp = prolif.Fingerprint()
            >>> fp.run_from_iterable(lig_iter, prot)

        .. seealso::

            :meth:`Fingerprint.generate` to generate the fingerprint between
            two single structures

        .. versionchanged:: 0.3.2
            Moved the ``return_atoms`` parameter from the ``run_from_iterable``
            method to the dataframe conversion code

        .. versionchanged:: 1.0.0
            Added support for multiprocessing

        .. versionchanged:: 2.0.0
            Changed the format of the :attr:`~Fingerprint.ifp` attribute to be a
            dictionary containing more complete interaction metadata instead of just
            atom indices.

        """
        if n_jobs is not None and n_jobs < 1:
            raise ValueError("n_jobs must be > 0 or None")
        if residues == "all":
            residues = list(prot_mol.residues)
        if n_jobs != 1:
            return self._run_iter_parallel(
                lig_iterable=lig_iterable,
                prot_mol=prot_mol,
                residues=residues,
                progress=progress,
                n_jobs=n_jobs,
            )

        iterator = tqdm(lig_iterable) if progress else lig_iterable
        ifp = {}
        for i, lig_mol in enumerate(iterator):
            ifp[i] = self.generate(lig_mol, prot_mol, residues=residues, metadata=True)
        self.ifp = ifp
        return self

    def _run_iter_parallel(
        self, lig_iterable, prot_mol, residues=None, progress=True, n_jobs=None
    ):
        """Parallel implementation of :meth:`~Fingerprint.run_from_iterable`"""
        total = (
            len(lig_iterable)
            if isinstance(lig_iterable, (Chem.SDMolSupplier, Sized))
            else None
        )
        ifp = {}

        with MolIterablePool(
            n_jobs,
            fingerprint=self,
            prot_mol=prot_mol,
            residues=residues,
            tqdm_kwargs={"total": total, "disable": not progress},
        ) as pool:
            for i, ifp_data in enumerate(pool.process(lig_iterable)):
                ifp[i] = ifp_data

        self.ifp = ifp
        return self

    def to_dataframe(
        self, *, count=None, dtype=None, drop_empty=True, index_col="Frame"
    ):
        """Converts fingerprints to a pandas DataFrame

        Parameters
        ----------
        count : bool or None
            Whether to output a count fingerprint or not.
        dtype : object or None
            Cast the dataframe values to this type. If ``None``, uses ``np.uint8`` if
            ``count=True``, else ``bool``.
        drop_empty : bool
            Drop columns with only empty values
        index_col : str
            Name of the index column in the DataFrame

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame storing the results frame by frame and residue by
            residue. See :meth:`~prolif.utils.to_dataframe` for more
            information

        Raises
        ------
        AttributeError
            If the :meth:`run` method hasn't been used

        Example
        -------
        ::

            >>> df = fp.to_dataframe(dtype=np.uint8)
            >>> print(df)
            ligand             LIG1.G
            protein             ILE59                  ILE55       TYR93
            interaction   Hydrophobic HBAcceptor Hydrophobic Hydrophobic PiStacking
            Frame
            0                       0          1           0           0          0
            ...

        .. versionchanged:: 2.0.0
            Removed the ``return_atoms`` parameter. You can access more metadata
            information directly through :attr:`~Fingerprint.ifp`. Added the ``count``
            parameter.
        """
        if hasattr(self, "ifp"):
            return to_dataframe(
                self.ifp,
                self.interactions,
                count=self.count if count is None else count,
                dtype=dtype,
                drop_empty=drop_empty,
                index_col=index_col,
            )
        raise AttributeError("Please use the `run` method before")

    def to_bitvectors(self):
        """Converts fingerprints to a list of RDKit ExplicitBitVector

        Returns
        -------
        bvs : list
            A list of :class:`~rdkit.DataStructs.cDataStructs.ExplicitBitVect`
            for each frame

        Raises
        ------
        AttributeError
            If the :meth:`run` method hasn't been used

        Example
        -------
        ::

            >>> from rdkit.DataStructs import TanimotoSimilarity
            >>> bv = fp.to_bitvectors()
            >>> TanimotoSimilarity(bv[0], bv[1])
            0.42

        """
        if hasattr(self, "ifp"):
            df = self.to_dataframe()
            return to_bitvectors(df)
        raise AttributeError("Please use the `run` method before")

    def to_countvectors(self):
        """Converts fingerprints to a list of RDKit UIntSparseIntVect

        Returns
        -------
        cvs : list
            A list of :class:`~rdkit.DataStructs.cDataStructs.UIntSparseIntVect`
            for each frame

        Raises
        ------
        AttributeError
            If the :meth:`run` method hasn't been used

        Example
        -------
        ::

            >>> from rdkit.DataStructs import TanimotoSimilarity
            >>> cv = fp.to_countvectors()
            >>> TanimotoSimilarity(cv[0], cv[1])
            0.42


        .. versionadded: 2.0.0
        """
        if hasattr(self, "ifp"):
            df = self.to_dataframe()
            return to_countvectors(df)
        raise AttributeError("Please use the `run` method before")

    def to_pickle(self, path=None):
        """Dumps the fingerprint object as a pickle.

        Parameters
        ----------
        path : str, pathlib.Path or None
            Output path. If ``None``, the method returns the pickle as bytes.

        Returns
        -------
        obj : None or bytes
            ``None`` if ``path`` is set, else the bytes corresponding to the
            pickle

        Example
        -------
        ::

            >>> dump = fp.to_pickle()
            >>> saved_fp = Fingerprint.from_pickle(dump)
            >>> fp.to_pickle("data/fp.pkl")
            >>> saved_fp = Fingerprint.from_pickle("data/fp.pkl")

        .. seealso::

            :meth:`~Fingerprint.from_pickle` for loading the pickle dump.

        .. versionadded:: 1.0.0

        .. versionchanged:: 2.0.0
            Switched to dill instead of pickle

        """
        if path:
            with open(path, "wb") as f:
                dill.dump(self, f)
            return None
        return dill.dumps(self)

    @classmethod
    def from_pickle(cls, path_or_bytes):
        """Creates a fingerprint object from a pickle dump.

        Parameters
        ----------
        path_or_bytes : str, pathlib.Path or bytes
            The path to the pickle file, or bytes corresponding to a pickle
            dump.


        .. seealso::

            :meth:`~Fingerprint.to_pickle` for creating the pickle dump.

        .. versionadded:: 1.0.0

        .. versionchanged:: 2.0.0
            Switched to dill instead of pickle

        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                r"The .+ interaction has been superseded by a new class",  # pragma: no cover
            )
            if isinstance(path_or_bytes, bytes):
                return dill.loads(path_or_bytes)
            with open(path_or_bytes, "rb") as f:
                return dill.load(f)

    def plot_lignetwork(
        self,
        ligand_mol,
        *,
        kind: Literal["aggregate", "frame"] = "aggregate",
        frame: int = 0,
        display_all: bool = False,
        threshold: float = 0.3,
        use_coordinates: bool = False,
        flatten_coordinates: bool = True,
        kekulize: bool = False,
        molsize: int = 35,
        rotation: float = 0,
        carbon: float = 0.16,
        width: str = "100%",
        height: str = "500px",
    ):
        """Generate and display a :class:`~prolif.plotting.network.LigNetwork` plot from
        a fingerprint object that has been used to run an analysis.

        Parameters
        ----------
        ligand_mol : rdkit.Chem.rdChem.Mol
            Ligand molecule.
        kind : str
            One of ``"aggregate"`` or ``"frame"``.
        frame : int
            Frame number (see :attr:`~prolif.fingerprint.Fingerprint.ifp`). Only
            applicable for ``kind="frame"``.
        display_all : bool
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Only applicable for ``kind="frame"``. Not relevant if
            ``count=False`` in the ``Fingerprint`` object.
        threshold : float
            Frequency threshold, between 0 and 1. Only applicable for
            ``kind="aggregate"``.
        use_coordinates : bool
            If ``True``, uses the coordinates of the molecule directly, otherwise generates
            2D coordinates from scratch. See also ``flatten_coordinates``.
        flatten_coordinates : bool
            If this is ``True`` and ``use_coordinates=True``, generates 2D coordinates that
            are constrained to fit the 3D conformation of the ligand as best as possible.
        kekulize : bool
            Kekulize the ligand.
        molsize : int
            Multiply the coordinates by this number to create a bigger and
            more readable depiction.
        rotation : int
            Rotate the structure on the XY plane.
        carbon : float
            Size of the carbon atom dots on the depiction. Use `0` to hide the
            carbon dots.
        width : str
            Width of the IFrame window.
        height : str
            Height of the IFrame window.

        Notes
        -----
        Two kinds of diagrams can be rendered: either for a designated frame or
        by aggregating the results on the whole IFP and optionnally discarding
        interactions that occur less frequently than a threshold. In the latter
        case (aggregate), only the group of atoms most frequently involved in
        each interaction is used to draw the edge.

        See Also
        --------
        :class:`prolif.plotting.network.LigNetwork`

        .. versionadded:: 2.0.0
        """
        from prolif.plotting.network import LigNetwork

        ligplot = LigNetwork.from_fingerprint(
            fp=self,
            ligand_mol=ligand_mol,
            kind=kind,
            frame=frame,
            display_all=display_all,
            threshold=threshold,
            use_coordinates=use_coordinates,
            flatten_coordinates=flatten_coordinates,
            kekulize=kekulize,
            molsize=molsize,
            rotation=rotation,
            carbon=carbon,
        )
        return ligplot.display(width=width, height=height)

    def plot_barcode(
        self,
        *,
        figsize: Tuple[int, int] = (8, 10),
        dpi: int = 100,
        interactive: bool = IS_NOTEBOOK,
        n_frame_ticks: int = 10,
        residues_tick_location: Literal["top", "bottom"] = "top",
        xlabel: str = "Frame",
        subplots_kwargs: Optional[dict] = None,
        tight_layout_kwargs: Optional[dict] = None,
    ):
        """Generate and display a :class:`~prolif.plotting.barcode.Barcode` plot from
        a fingerprint object that has been used to run an analysis.

        Parameters
        ----------
        figsize: Tuple[int, int] = (8, 10)
            Size of the matplotlib figure.
        dpi: int = 100
            DPI used for the matplotlib figure.
        interactive: bool
            Add hover interactivity to the plot (only relevant for notebooks). You may
            need to add ``%matplotlib notebook`` or ``%matplotlib ipympl`` for it to
            work as expected.
        n_frame_ticks: int = 10
            Number of ticks on the X axis. May use ±1 tick to have them evenly spaced.
        residues_tick_location: Literal["top", "bottom"] = "top"
            Whether the Y ticks appear at the top or at the bottom of the series of
            interactions of each residue.
        xlabel: str = "Frame"
            Label displayed for the X axis.
        subplots_kwargs: Optional[dict] = None
            Other parameters passed to :func:`matplotlib.pyplot.subplots`.
        tight_layout_kwargs: Optional[dict] = None
            Other parameters passed to :meth:`matplotlib.figure.Figure.tight_layout`.

        See Also
        --------
        :class:`prolif.plotting.barcode.Barcode`

        .. versionadded:: 2.0.0
        """
        from prolif.plotting.barcode import Barcode

        barcode = Barcode.from_fingerprint(self)
        return barcode.display(
            figsize=figsize,
            dpi=dpi,
            interactive=interactive,
            n_frame_ticks=n_frame_ticks,
            residues_tick_location=residues_tick_location,
            xlabel=xlabel,
            subplots_kwargs=subplots_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
        )

    def plot_3d(
        self,
        ligand_mol: Molecule,
        protein_mol: Molecule,
        *,
        frame: int,
        size: Tuple[int, int] = (650, 600),
        display_all: bool = False,
    ):
        """Generate and display the complex in 3D with py3Dmol from a fingerprint object
        that has been used to run an analysis.

        Parameters
        ----------
        ligand_mol : Molecule
            The ligand molecule to display.
        protein_mol : Molecule
            The protein molecule to display.
        frame : int
            The frame number chosen to select which interactions are going to be
            displayed.
        size: Tuple[int, int] = (650, 600)
            The size of the py3Dmol widget view.
        display_all : bool
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Not relevant if ``count=False`` in the ``Fingerprint``
            object.

        See Also
        --------
        :class:`prolif.plotting.complex3d.Complex3D`

        .. versionadded:: 2.0.0
        """
        from prolif.plotting.complex3d import Complex3D

        plot3d = Complex3D.from_fingerprint(
            self, frame=frame, lig_mol=ligand_mol, prot_mol=protein_mol
        )
        return plot3d.display(size=size, display_all=display_all)
