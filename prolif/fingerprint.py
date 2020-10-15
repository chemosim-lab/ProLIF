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
    lig = u.select_atoms("resname ERM")
    fp = prolif.Fingerprint(["HBDonor", "HBAcceptor", "PiStacking", "CationPi", "Cationic"])
    fp.run(u.trajectory[::2], lig, prot)
    df = fp.to_dataframe()
    df
    bv = fp.to_bitvectors()
    TanimotoSimilarity(bv[0], bv[1])
    
"""
import logging
from functools import wraps
from collections.abc import Iterable
import numpy as np
from tqdm.auto import tqdm
from .interactions import _INTERACTIONS
from .molecule import Molecule
from .utils import get_pocket_residues, to_dataframe, to_bitvectors

logger = logging.getLogger("prolif")

_BOOL_OR_NONE = (bool, type(None))


def _only_return_bits(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        results = f(*args, **kwargs)
        return results if isinstance(results, _BOOL_OR_NONE) else results[0]
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
        Dictionnary of interaction functions index by class name. For more
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
        lig = u.select_atoms("resname ERM")
        fp = prolif.Fingerprint(["HBDonor", "HBAcceptor", "PiStacking", "Hydrophobic"])
        fp.run(u.trajectory[:5], lig, prot)
        fp.to_dataframe()
    
    - On a specific frame and a specific pair of residues:

    .. ipython:: python

        u.trajectory[0] # use the first frame
        prot = prolif.Molecule.from_mda(prot)
        lig = prolif.Molecule.from_mda(lig)
        fp.bitstring(lig, prot["ASP129.0"])

    - On a specific pair of residues for a specific interaction:

    .. ipython:: python

        fp.hbdonor(lig, prot["ASP129.0"]) # ligand-protein
        fp.hbacceptor(prot["ASP129.0"], prot["CYS133.0"]) # protein-protein (alpha helix)
    
    You can also obtain the indices of atoms responsible for the interaction:

    .. ipython:: python

        fp.bitstring_atoms(lig, prot["ASP129.0"])
        fp.hbdonor.__wrapped__(lig, prot["ASP129.0"])
    
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
            func = _only_return_bits(func)
            setattr(self, name.lower(), func)
            if name in interactions:
                self.interactions[name] = func
        logger.info('The following bitstring will be generated: {}'.format(
                    ' '.join(interactions)))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_interactions} interactions: {list(self.interactions.keys())}"
        return f"<{name}: {params} at {id(self):#x}>"

    @property
    def n_interactions(self):
        return len(self.interactions)

    def bitstring(self, res1, res2):
        """Generates the complete bitstring for the interactions between two
        residues. To get the indices of atoms responsible for each interaction,
        see :meth:`bitstring_atoms`

        Parameters
        ----------
        res1 : prolif.residue.Residue or prolif.molecule.Molecule
            A residue, usually from a ligand
        res2 : prolif.residue.Residue or prolif.molecule.Molecule
            A residue, usually from a protein

        Returns
        -------
        bitstring : numpy.ndarray
            An array storing the encoded interactions between res1 and res2

        """
        bitstring = []
        for func in self.interactions.values():
            bit = func(res1, res2)
            bitstring.append(bit)
        return np.array(bitstring, dtype=bool)

    def bitstring_atoms(self, res1, res2):
        """Generates the complete bitstring for the interactions between two
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
        bitstring : :class:`numpy.ndarray`
            An array storing the encoded interactions between res1 and res2
        atoms : :class:`list`
            A list containing tuples of (res1_atom_index, res2_atom_index) for
            each interaction
        """
        bitstring = []
        atoms_lst = []
        for func in self.interactions.values():
            bit, *atoms = func.__wrapped__(res1, res2)
            bitstring.append(bit)
            atoms_lst.append(atoms)
        bitstring = np.array(bitstring, dtype=bool)
        return bitstring, atoms_lst

    def run(self, traj, lig, prot, residues=None, progress=True):
        """Generates the fingerprint on a trajectory for two atomgroups

        Parameters
        ----------
        traj : MDAnalysis.coordinates.base.ProtoReader or MDAnalysis.coordinates.base.FrameIteratorSliced
            Iterate over this Universe trajectory or sliced trajectory object
            to extract the frames used for the fingerprint extraction
        lig : MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the ligand
        prot : MDAnalysis.core.groups.AtomGroup
            An MDAnalysis AtomGroup for the protein
        residues : list or None or `"pocket"`
            A list of residues (:class:`str`, :class:`int` or
            :class:`~prolif.residue.ResidueId`) taken into account for
            the fingerprint extraction. If ``None``, all residues will be used.
            If ``"pocket"``, the :func:`~prolif.utils.get_pocket_residues`
            function is used to automatically extract protein residues based on
            the ligand's position at each frame
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

        """
        iterator = tqdm(traj) if progress else traj
        # set residues
        run_pocket = False
        if residues == "pocket":
            run_pocket = True
        elif isinstance(residues, Iterable):
            resids = residues
        else:
            resids = Molecule.from_mda(prot).residues
        ifp = []
        for ts in iterator:
            lig_mol = Molecule.from_mda(lig)
            prot_mol = Molecule.from_mda(prot)
            if run_pocket:
                resids = get_pocket_residues(lig_mol, prot_mol)
            data = {"Frame": ts.frame}
            for res in resids:
                bs = self.bitstring(lig_mol, prot_mol[res])
                if bs.sum() > 0:
                    data[res] = bs
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
