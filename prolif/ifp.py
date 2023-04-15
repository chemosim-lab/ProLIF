from collections import UserDict

from .residue import ResidueId


class IFP(UserDict):
    """Mapping between residue pairs and interactions metadata."""

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
