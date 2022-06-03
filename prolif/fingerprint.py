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
    fp.run(u.trajectory[::10], lig, prot)
    df = fp.to_dataframe()
    df
    bv = fp.to_bitvectors()
    TanimotoSimilarity(bv[0], bv[1])

"""
from functools import wraps
import numpy as np
from tqdm.auto import tqdm
from .interactions import _INTERACTIONS
from .molecule import Molecule
from .utils import get_residues_near_ligand, to_dataframe, to_bitvectors


def _return_first_element(f):
    """Modifies the return signature of a function by forcing it to return
    only the first element when multiple values are returned

    Raises
    ------
    TypeError
        If the function doesn't return three values

    Notes
    -----
    The original return signature of the decorated function is still accessible
    by calling ``function.__wrapped__(*args, **kwargs)``.

    Example
    -------
    ::

        >>> def foo():
        ...     return 1, 2, 3
        ...
        >>> bar = _return_first_element(foo)
        >>> foo()
        (1, 2, 3)
        >>> bar()
        1
        >>> bar.__wrapped__()
        (1, 2, 3)

    .. versionchanged:: 0.3.3
        The function now must return three values

    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        results = f(*args, **kwargs)
        try:
            bool_, lig_idx, prot_idx = results
        except (TypeError, ValueError):
            raise TypeError(
                "Incorrect function signature: the interaction class must "
                "return 3 values (boolean, int, int)"
            ) from None
        return bool_
    return wrapper


class Fingerprint:
    """Class that generates an interaction fingerprint between two molecules

    While in most cases the fingerprint will be generated between a ligand and
    a protein, it is also possible to use this class for protein-protein
    interactions, or host-guest systems.

    Parameters
    ----------
    interactions : list
        List of names (str) of interaction classes as found in the
        :mod:`prolif.interactions` module

    Attributes
    ----------
    interactions : dict
        Dictionnary of interaction functions indexed by class name. For more
        details, see :mod:`prolif.interactions`
    n_interactions : int
        Number of interaction functions registered by the fingerprint
    ifp : list, optionnal
        List of interactions fingerprints for the given trajectory.

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
        ifp["Frame"] = 0
        prolif.to_dataframe([ifp], fp.interactions.keys())

    - On a specific pair of residues for a specific interaction:

    .. ipython:: python

        # ligand-protein
        fp.hbdonor(lig, prot["ASP129.A"])
        # protein-protein (alpha helix)
        fp.hbacceptor(prot["ASP129.A"], prot["CYS133.A"])

    You can also obtain the indices of atoms responsible for the interaction:

    .. ipython:: python

        fp.bitvector_atoms(lig, prot["ASP129.A"])
        fp.hbdonor.__wrapped__(lig, prot["ASP129.A"])

    """

    def __init__(self, interactions=["Hydrophobic", "HBDonor", "HBAcceptor",
                 "PiStacking", "Anionic", "Cationic", "CationPi", "PiCation"]):
        self.interactions = {}
        if interactions == "all":
            interactions = self.list_available()
        # sanity check
        unsafe = set(interactions)
        unk = unsafe.symmetric_difference(_INTERACTIONS.keys()) & unsafe
        if unk:
            raise NameError(f"Unknown interaction(s): {', '.join(unk)}")
        # add interaction methods
        for name, interaction_cls in _INTERACTIONS.items():
            if name.startswith("_") or name == "Interaction":
                continue
            func = interaction_cls().detect
            func = _return_first_element(func)
            setattr(self, name.lower(), func)
            if name in interactions:
                self.interactions[name] = func

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
            interactions = [name for name in _INTERACTIONS.keys()
                            if not (name.startswith("_")
                                    or name == "Interaction")]
        return sorted(interactions)

    @property
    def n_interactions(self):
        return len(self.interactions)

    def bitvector(self, res1, res2):
        """Generates the complete bitvector for the interactions between two
        residues. To get the indices of atoms responsible for each interaction,
        see :meth:`bitvector_atoms`

        Parameters
        ----------
        res1 : prolif.residue.Residue or prolif.molecule.Molecule
            A residue, usually from a ligand
        res2 : prolif.residue.Residue or prolif.molecule.Molecule
            A residue, usually from a protein

        Returns
        -------
        bitvector : numpy.ndarray
            An array storing the encoded interactions between res1 and res2
        """
        bitvector = []
        for func in self.interactions.values():
            bit = func(res1, res2)
            bitvector.append(bit)
        return np.array(bitvector)

    def bitvector_atoms(self, res1, res2):
        """Generates the complete bitvector for the interactions between two
        residues, and returns the indices of atoms responsible for these
        interactions

        Parameters
        ----------
        res1 : prolif.residue.Residue or prolif.molecule.Molecule
            A residue, usually from a ligand
        res2 : prolif.residue.Residue or prolif.molecule.Molecule
            A residue, usually from a protein

        Returns
        -------
        bitvector : numpy.ndarray
            An array storing the encoded interactions between res1 and res2
        lig_atoms : list
            A list containing indices for the ligand atoms responsible for each
            interaction
        pro_atoms : list
            A list containing indices for the protein atoms responsible for
            each interaction


        .. versionchanged:: 0.3.2
            Atom indices are returned as two separate lists instead of a single
            list of tuples

        """
        bitvector = []
        lig_atoms = []
        prot_atoms = []
        for func in self.interactions.values():
            bit, la, pa = func.__wrapped__(res1, res2)
            bitvector.append(bit)
            lig_atoms.append(la)
            prot_atoms.append(pa)
        bitvector = np.array(bitvector)
        return bitvector, lig_atoms, prot_atoms

    def generate(self, lig, prot, residues=None, return_atoms=False):
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
            less from each ligand residue.
        return_atoms : bool
            For each residue pair and interaction, return indices of atoms
            responsible for the interaction instead of bits.

        Returns
        -------
        ifp : dict
            A dictionnary indexed by ``(ligand, protein)`` residue pairs. The
            format for values will depend on ``return_atoms``:

            - A single bitvector if ``return_atoms=False``
            - A tuple of bitvector, ligand atom indices and protein atom
              indices if ``return_atoms=True``

        Example
        -------
        ::

            >>> u = mda.Universe("complex.pdb")
            >>> lig = prolif.Molecule.from_mda(u, "resname LIG")
            >>> prot = prolif.Molecule.from_mda(u, "protein")
            >>> fp = prolif.Fingerprint()
            >>> ifp = fp.generate(lig, prot)

        .. versionadded:: 0.3.2
        """
        ifp = {}
        resids = residues
        if residues == "all":
            resids = prot.residues.keys()
        for lresid, lres in lig.residues.items():
            if residues is None:
                resids = get_residues_near_ligand(lres, prot)
            for prot_key in resids:
                pres = prot[prot_key]
                key = (lresid, pres.resid)
                if return_atoms:
                    ifp[key] = self.bitvector_atoms(lres, pres)
                else:
                    ifp[key] = self.bitvector(lres, pres)
        return ifp

    def run(self, traj, lig, prot, residues=None, progress=True):
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
        progress : bool
            Use the `tqdm <https://tqdm.github.io/>`_ package to display a
            progressbar while running the calculation

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

        """
        iterator = tqdm(traj) if progress else traj
        if residues == "all":
            residues = Molecule.from_mda(prot).residues.keys()
        ifp = []
        for ts in iterator:
            prot_mol = Molecule.from_mda(prot)
            lig_mol = Molecule.from_mda(lig)
            data = self.generate(lig_mol, prot_mol, residues=residues,
                                 return_atoms=True)
            data["Frame"] = ts.frame
            ifp.append(data)
        self.ifp = ifp
        return self

    def run_from_iterable(self, lig_iterable, prot_mol, residues=None,
                          progress=True):
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

        """
        iterator = tqdm(lig_iterable) if progress else lig_iterable
        if residues == "all":
            residues = prot_mol.residues.keys()
        ifp = []
        for i, lig_mol in enumerate(iterator):
            data = self.generate(lig_mol, prot_mol, residues=residues,
                                 return_atoms=True)
            data["Frame"] = i
            ifp.append(data)
        self.ifp = ifp
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
        return_atoms : bool
            For each residue pair and interaction, return indices of atoms
            responsible for the interaction instead of bits

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
