import pytest

from prolif.fingerprint import Fingerprint
from prolif.residue import ResidueId


@pytest.fixture(scope="session")
def ifp(u, ligand_ag, protein_ag):
    fp = Fingerprint(["Hydrophobic", "VdWContact"])
    fp.run(u.trajectory[0:1], ligand_ag, protein_ag)
    return fp.ifp[0]


def test_ifp_indexing(ifp):
    lig_id, prot_id = "LIG1.G", "LEU126.A"
    metadata1 = ifp[(ResidueId.from_string(lig_id), ResidueId.from_string(prot_id))]
    metadata2 = ifp[(lig_id, prot_id)]
    assert metadata1 is metadata2


def test_ifp_filtering(ifp):
    lig_id, prot_id = "LIG1.G", "LEU126.A"
    assert ifp[lig_id] == ifp
    assert (
        next(iter(ifp[prot_id].values()))
        == ifp[(ResidueId.from_string(lig_id), ResidueId.from_string(prot_id))]
    )


def test_wrong_key(ifp):
    with pytest.raises(KeyError, match="does not correspond to a valid IFP key"):
        ifp[0]
