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
import multiprocessing as mp
import pickle
from collections.abc import Iterable
from inspect import isgenerator
from threading import Thread

import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm

from .ifp import IFP
from .interactions import _INTERACTIONS
from .molecule import Molecule
from .parallel import (
    Progress,
    ProgressCounter,
    declare_shared_objs_for_chunk,
    declare_shared_objs_for_mol,
    process_chunk,
    process_mol,
)
from .utils import get_residues_near_ligand, to_bitvectors, to_dataframe


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
    ifp : dict, optional
        Dict of interaction fingerprints in a sparse format for the given trajectory or
        docking poses: ``{<frame number>: <IFP>}``. See the :class:`~prolif.ifp.IFP`
        class for more information.

    Raises
    ------
    NameError : Unknown interaction in the ``interactions`` parameter

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
        ifp = fp.generate(lig, prot)
        prolif.to_dataframe({0: ifp}, fp.interactions.keys())

    - On a specific pair of residues for a specific interaction:

    .. ipython:: python

        # ligand-protein
        fp.hbdonor(lig, prot["ASP129.A"])
        # protein-protein
        fp.hbacceptor(prot["ASP129.A"], prot["CYS133.A"])

    You can also obtain the indices of atoms responsible for the interaction:

    .. ipython:: python

        fp.metadata(lig, prot["ASP129.A"])
        fp.hbdonor(lig, prot["ASP129.A"], metadata=True)


    .. versionchanged:: 1.0.0
        Added pickle support

    .. versionchanged:: 2.0.0
        Changed the format of the :attr:`~Fingerprint.ifp` attribute to be a dictionary
        containing more complete interaction metadata instead of just atom indices.
        Removed the ``return_atoms`` argument in :meth:`~Fingerprint.to_dataframe`.
        Users should directly use :attr:`~Fingerprint.ifp` instead.
        Added the :meth:`~Fingerprint.to_ligplot` method to generate the
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
        vicinity_cutoff=6.0,
    ):
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
        for name, interaction_cls in _INTERACTIONS.items():
            if name.startswith("_") or name == "Interaction":
                continue
            # create instance with custom parameters if available
            interaction = interaction_cls(**parameters.get(name, {}))
            setattr(self, name.lower(), interaction)
            if name in interactions:
                self.interactions[name] = interaction

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
            Show hidden classes (usually base classes whose name starts with an
            underscore ``_``. Those are not supposed to be called directly)
        """
        if show_hidden:
            interactions = [name for name in _INTERACTIONS.keys()]
        else:
            interactions = [
                name
                for name in _INTERACTIONS.keys()
                if not (name.startswith("_") or name == "Interaction")
            ]
        return sorted(interactions)

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
            An array storing the encoded interactions between res1 and res2
        """
        bitvector = [
            interaction(res1, res2) for interaction in self.interactions.values()
        ]
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
        metadata : dict
            Metadata dictionnary indexed by interaction name. If a specific interaction
            is not present between residues, it is filtered out of the dictionary.


        .. versionchanged:: 0.3.2
            Atom indices are returned as two separate lists instead of a single
            list of tuples

        .. versionchanged:: 2.0.0
            Returns a dictionnary with all available metadata for each interaction,
            rather than just atom indices.

        """
        metadata = {
            name: metadata
            for name, interaction in self.interactions.items()
            if (metadata := interaction(res1, res2, metadata=True)) is not None
        }
        return metadata

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
        ifp : dict
            A dictionary indexed by ``(ligand, protein)`` residue pairs. The
            format for values will depend on ``metadata``:

            - A single bitvector if ``metadata=False``
            - A sparse dictionary of metadata for each interaction if ``metadata=True``

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
            dictionary of metadata indexed by interaction name instead of a tuple of
            arrays.
        """
        ifp = IFP()
        prot_residues = residues
        if residues == "all":
            prot_residues = prot.residues.keys()
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
        converter_kwargs : list or None
            List of kwargs passed to the underlying :class:`~MDAnalysis.converters.RDKit.RDKitConverter`
            from MDAnalysis: the first for the ligand, and the second for the protein
        progress : bool
            Use the `tqdm <https://tqdm.github.io/>`_ package to display a
            progressbar while running the calculation
        n_jobs : int or None
            Number of processes to run in parallel. If ``n_jobs=None``, the
            analysis will use all available CPU threads, while if ``n_jobs=1``,
            the analysis will run in serial.

        Raises
        ------
        ValueError : if ``n_jobs <= 0``

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
        lig_kwargs, prot_kwargs = converter_kwargs or ({}, {})

        iterator = tqdm(traj) if progress else traj
        if residues == "all":
            residues = Molecule.from_mda(prot, **prot_kwargs).residues.keys()
        ifp = {}
        for ts in iterator:
            prot_mol = Molecule.from_mda(prot, **prot_kwargs)
            lig_mol = Molecule.from_mda(lig, **lig_kwargs)
            ifp[ts.frame] = self.generate(
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
        lig_kwargs, prot_kwargs = converter_kwargs or ({}, {})

        if residues == "all":
            residues = Molecule.from_mda(prot, **prot_kwargs).residues.keys()
        chunks = np.array_split(frames, n_chunks)
        # setup parallel progress bar
        pcount = ProgressCounter()
        if progress:
            pbar = Progress(pcount, total=len(frames))
        else:
            pbar = lambda: None
        pbar_thread = Thread(target=pbar, daemon=True)

        # run pool of workers
        with mp.Pool(
            n_jobs,
            initializer=declare_shared_objs_for_chunk,
            initargs=(self, residues, progress, pcount, (lig_kwargs, prot_kwargs)),
        ) as pool:
            pbar_thread.start()
            args = ((traj, lig, prot, chunk) for chunk in chunks)
            ifp = {}
            for ifp_data_chunk in pool.imap_unordered(process_chunk, args):
                ifp.update(ifp_data_chunk)
        # sort
        self.ifp = {frame: ifp[frame] for frame in sorted(ifp)}
        return self

    def run_from_iterable(
        self, lig_iterable, prot_mol, residues=None, progress=True, n_jobs=None
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
            Use the `tqdm <https://tqdm.github.io/>`_ package to display a
            progressbar while running the calculation
        n_jobs : int or None
            Number of processes to run in parallel. If ``n_jobs=None``, the
            analysis will use all available CPU threads, while if ``n_jobs=1``,
            the analysis will run in serial.

        Raises
        ------
        ValueError : if ``n_jobs <= 0``

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
        if n_jobs != 1:
            return self._run_iter_parallel(
                lig_iterable=lig_iterable,
                prot_mol=prot_mol,
                residues=residues,
                progress=progress,
                n_jobs=n_jobs,
            )

        iterator = tqdm(lig_iterable) if progress else lig_iterable
        if residues == "all":
            residues = prot_mol.residues.keys()
        ifp = {}
        for i, lig_mol in enumerate(iterator):
            ifp[i] = self.generate(lig_mol, prot_mol, residues=residues, metadata=True)
        self.ifp = ifp
        return self

    def _run_iter_parallel(
        self, lig_iterable, prot_mol, residues=None, progress=True, n_jobs=None
    ):
        """Parallel implementation of :meth:`~Fingerprint.run_from_iterable`"""
        previous_pkl_props = Chem.GetDefaultPickleProperties()
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        if isinstance(lig_iterable, Chem.SDMolSupplier) or (
            isinstance(lig_iterable, Iterable) and not isgenerator(lig_iterable)
        ):
            total = len(lig_iterable)
        else:
            total = None
        suppl = enumerate(lig_iterable)
        if residues == "all":
            residues = prot_mol.residues.keys()

        with mp.Pool(
            n_jobs,
            initializer=declare_shared_objs_for_mol,
            initargs=(self, prot_mol, residues),
        ) as pool:
            results = {}
            for i, ifp_data in tqdm(
                pool.imap_unordered(process_mol, suppl),
                total=total,
                disable=not progress,
            ):
                results[i] = ifp_data
        self.ifp = {frame: results[frame] for frame in sorted(results)}
        Chem.SetDefaultPickleProperties(previous_pkl_props)
        return self

    def to_dataframe(self, **kwargs):
        """Converts fingerprints to a pandas DataFrame

        Parameters
        ----------
        dtype : object or None
            Cast the input of each bit in the bitvector to this type. If None,
            keep the data as is
        drop_empty : bool
            Drop columns with only empty values

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
            information directly through :attr:`~Fingerprint.ifp`.
        """
        if hasattr(self, "ifp"):
            return to_dataframe(self.ifp, self.interactions.keys(), **kwargs)
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

        """
        if path:
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return None
        return pickle.dumps(self)

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

        """
        if isinstance(path_or_bytes, bytes):
            return pickle.loads(path_or_bytes)
        with open(path_or_bytes, "rb") as f:
            return pickle.load(f)

    def to_ligplot(
        self, ligand_mol, kind="aggregate", frame=0, threshold=0.3, **kwargs
    ):
        """Generate a :class:`~prolif.plotting.network.LigNetwork` plot from a
        fingerprint object that has been executed.

        .. versionadded:: 2.0.0
        """
        from prolif.plotting.network import LigNetwork

        if hasattr(self, "ifp"):
            return LigNetwork.from_fingerprint(
                fp=self,
                ligand_mol=ligand_mol,
                kind=kind,
                frame=frame,
                threshold=threshold,
                **kwargs,
            )
        raise AttributeError("Please use the `run` method before")
