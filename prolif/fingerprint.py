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
    fp = prolif.Fingerprint(["HBDonor", "HBAcceptor", "PiStacking", "CationPi", "Cationic"])
    fp.run(u.trajectory[::10], lig, prot)
    df = fp.to_dataframe()
    df
    bv = fp.to_bitvectors()
    TanimotoSimilarity(bv[0], bv[1])
    
"""
from functools import wraps
from collections.abc import Iterable
import numpy as np
from tqdm.auto import tqdm
from .interactions import _INTERACTIONS
from .molecule import Molecule
from .utils import get_residues_near_ligand, to_dataframe, to_bitvectors


def _return_first_element(f):
    """Modifies the return signature of a function by forcing it to return
    only the first element if multiple values were returned.

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

    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        results = f(*args, **kwargs)
        try:
            value, *rest = results
        except TypeError:
            value = results
        return value
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
        Results are stored in the form of a list of
        ``{"Frame": frame_index, ResidueId: numpy.ndarray, ...}`` dictionaries
        that contain the interaction fingerprints

    Notes
    -----
    You can use the fingerprint generator in multiple ways:

    - On a trajectory directly from MDAnalysis objects:

    .. ipython:: python

        prot = u.select_atoms("protein")
        lig = u.select_atoms("resname LIG")
        fp = prolif.Fingerprint(["HBDonor", "HBAcceptor", "PiStacking", "Hydrophobic"])
        fp.run(u.trajectory[:5], lig, prot)
        fp.to_dataframe()

    - On a specific frame and a specific pair of residues:

    .. ipython:: python

        u.trajectory[0] # use the first frame
        prot = prolif.Molecule.from_mda(prot)
        lig = prolif.Molecule.from_mda(lig)
        fp.bitvector(lig, prot["ASP129.A"])

    - On a specific pair of residues for a specific interaction:

    .. ipython:: python

        fp.hbdonor(lig, prot["ASP129.A"]) # ligand-protein
        fp.hbacceptor(prot["ASP129.A"], prot["CYS133.A"]) # protein-protein (alpha helix)
    
    You can also obtain the indices of atoms responsible for the interaction:

    .. ipython:: python

        fp.bitvector_atoms(lig, prot["ASP129.A"])
        fp.hbdonor.__wrapped__(lig, prot["ASP129.A"])
    
    """

    def __init__(self, interactions=["Hydrophobic", "HBDonor", "HBAcceptor",
                 "PiStacking", "Anionic", "Cationic", "CationPi", "PiCation"]):
        # read interactions to compute
        self.interactions = {}
        if interactions == "all":
            interactions = [i for i in _INTERACTIONS.keys() 
                            if not (i.startswith("_") or i == "Interaction")]
        for name, interaction_cls in _INTERACTIONS.items():
            if name.startswith("_") or name == "Interaction":
                continue
            func = interaction_cls().detect
            func = _return_first_element(func)
            setattr(self, name.lower(), func)
            if name in interactions:
                self.interactions[name] = func

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_interactions} interactions: {list(self.interactions.keys())}"
        return f"<{name}: {params} at {id(self):#x}>"

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
        return np.array(bitvector, dtype=bool)

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
        bitvector : :class:`numpy.ndarray`
            An array storing the encoded interactions between res1 and res2
        atoms : :class:`list`
            A list containing tuples of (res1_atom_index, res2_atom_index) for
            each interaction
        """
        bitvector = []
        atoms_lst = []
        for func in self.interactions.values():
            bit, *atoms = func.__wrapped__(res1, res2)
            bitvector.append(bit)
            atoms_lst.append(atoms)
        bitvector = np.array(bitvector, dtype=bool)
        return bitvector, atoms_lst

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
            A list of residues (:class:`str`, :class:`int` or
            :class:`~prolif.residue.ResidueId`) to take into account for
            the fingerprint extraction.
            If ``"all"``, all residues will be used.
            If ``None``, at each frame the :func:`~prolif.utils.get_residues_near_ligand`
            function is used to automatically use protein residues that are
            distant of 6.0 Å or less from the ligand.
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
        
        Notes
        -----
        The results are stored in the ``ifp`` attribute of the fingerprint as
        a list of dictionnaries in the format ``{"Frame": frame_index,
        ResidueId: numpy.ndarray, ...}``.
        Residues for which no interaction were detected are not added to the
        IFP list.
        """
        iterator = tqdm(traj) if progress else traj
        # set which residues to use
        select_residues = False
        if residues == "all":
            resids = Molecule.from_mda(prot).residues
        elif isinstance(residues, Iterable):
            resids = residues
        else:
            select_residues = True
        ifp = []
        for ts in iterator:
            prot_mol = Molecule.from_mda(prot)
            lig_mol = Molecule.from_mda(lig)
            if select_residues:
                resids = get_residues_near_ligand(lig_mol, prot_mol)
            data = {"Frame": ts.frame}
            for res in resids:
                prot_res = prot_mol[res]
                bs = self.bitvector(lig_mol, prot_res)
                if bs.sum() > 0:
                    data[prot_res.resid] = bs
            ifp.append(data)
        self.ifp = ifp
        return self

    def to_dataframe(self):
        """Converts fingerprints to a pandas DataFrame

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

            >>> df = fp.to_dataframe()
            >>> print(df)
            Frame     ILE59                  ILE55       TYR93
                    Hydrophobic HBAcceptor Hydrophobic Hydrophobic PiStacking
            0      0           1          0           0           0          0
            ...

        """
        if hasattr(self, "ifp"):
            return to_dataframe(self.ifp, self)
        raise AttributeError("Please use the run method before")

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
            return to_bitvectors(self.ifp, self)
        raise AttributeError("Please use the run method before")
