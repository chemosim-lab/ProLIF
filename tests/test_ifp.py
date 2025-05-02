from typing import TYPE_CHECKING

import pytest

from prolif.fingerprint import Fingerprint
from prolif.ifp import IFP, InteractionData
from prolif.residue import ResidueId

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup
    from MDAnalysis.core.universe import Universe


@pytest.fixture(scope="session")
def ifp(u: "Universe", ligand_ag: "AtomGroup", protein_ag: "AtomGroup") -> IFP:
    fp = Fingerprint(["Hydrophobic", "VdWContact"])
    fp.run(u.trajectory[0:1], ligand_ag, protein_ag)
    return fp.ifp[0]


def test_ifp_indexing(ifp: IFP) -> None:
    lig_id, prot_id = "LIG1.G", "LEU126.A"
    metadata1 = ifp[ResidueId.from_string(lig_id), ResidueId.from_string(prot_id)]
    metadata2 = ifp[lig_id, prot_id]
    assert metadata1 is metadata2


def test_ifp_filtering(ifp: IFP) -> None:
    lig_id, prot_id = "LIG1.G", "LEU126.A"
    assert ifp[lig_id] == ifp
    assert (
        next(iter(ifp[prot_id].values()))
        == ifp[ResidueId.from_string(lig_id), ResidueId.from_string(prot_id)]
    )


def test_wrong_key(ifp: IFP) -> None:
    with pytest.raises(KeyError, match="does not correspond to a valid IFP key"):
        ifp[0]  # type: ignore[call-overload]


def test_interaction_data_iteration(ifp: IFP) -> None:
    data = next(ifp.interactions())
    assert isinstance(data, InteractionData)
    assert data.ligand == ResidueId("LIG", 1, "G")
    assert data.protein.chain in {"A", "B"}
    assert data.interaction in {"Hydrophobic", "VdWContact"}
    assert "distance" in data.metadata
    for data in ifp.interactions():
        assert isinstance(data, InteractionData)
