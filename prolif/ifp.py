"""
Storing interactions --- :mod:`prolif.ifp`
==========================================
"""

from collections import UserDict

from prolif.residue import ResidueId


class IFP(UserDict):
    """Mapping between residue pairs and interaction fingerprint.

    Notes
    -----
    This class provides an easy way to access interaction data from the
    :attr:`~prolif.fingerprint.Fingerprint.ifp` dictionary. This class is a dictionary
    formatted as:

    .. code-block:: text

        {
            tuple[<residue_id>, <residue_id>]: {
                <interaction name>: tuple[{
                    "indices": {
                        "ligand": tuple[int, ...],
                        "protein": tuple[int, ...]
                    },
                    "parent_indices": {
                        "ligand": tuple[int, ...],
                        "protein": tuple[int, ...]
                    },
                    <other metadata>: <value>
                }, ...]
            }
        }

    Here ``<residue_id>`` corresponds to a :class:`~prolif.residue.ResidueId` object.
    For convenience, one can directly use strings rather than ``ResidueId`` objects when
    indexing the IFP, e.g. ``ifp[("LIG1.G", "ASP129.A")]``.
    You can also use a single ``ResidueId`` or string to return a filtered IFP that only
    contains interactions with the specified residue, e.g. ``ifp["ASP129.A"]``.
    """

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError as exc:
            if isinstance(key, tuple) and len(key) == 2:
                lig_res, prot_res = key
                if isinstance(lig_res, ResidueId):
                    raise exc from None
                key = ResidueId.from_string(lig_res), ResidueId.from_string(prot_res)
                return self.data[key]
            if isinstance(key, str):
                key = ResidueId.from_string(key)
            if isinstance(key, ResidueId):
                return IFP(
                    {
                        residue_tuple: interactions
                        for residue_tuple, interactions in self.data.items()
                        if key in residue_tuple
                    }
                )
        raise KeyError(
            f"{key} does not correspond to a valid IFP key: it must be a tuple of "
            "either ResidueId or residue string. If you need to filter the IFP, a "
            "single ResidueId or residue string can also be used."
        )
