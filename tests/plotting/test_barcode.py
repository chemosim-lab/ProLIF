import MDAnalysis as mda
import pytest
from matplotlib import pyplot as plt

import prolif as plf
from prolif.plotting.barcode import Barcode


class TestBarcode:
    @pytest.fixture(scope="class")
    def simple_fp(self) -> plf.Fingerprint:
        return plf.Fingerprint(count=False)

    @pytest.fixture(scope="class")
    def count_fp(self) -> plf.Fingerprint:
        return plf.Fingerprint(count=True)

    @pytest.fixture(scope="class", params=["simple_fp", "count_fp"])
    def fp(self, request: pytest.FixtureRequest) -> plf.Fingerprint:
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="class")
    def barcode(
        self, u: mda.Universe, ligand_ag, protein_ag, fp: plf.Fingerprint
    ) -> Barcode:
        fp.run(u.trajectory[0:2], ligand_ag, protein_ag)
        return Barcode.from_fingerprint(fp)

    def test_display(self, barcode: Barcode) -> None:
        ax = barcode.display()
        assert isinstance(ax, plt.Axes)

    def test_display_kwargs(self, barcode: Barcode) -> None:
        ax = barcode.display(
            figsize=(1, 2),
            dpi=200,
            interactive=True,
            n_frame_ticks=2,
            residues_tick_location="bottom",
            xlabel="foobar",
            subplots_kwargs={},
            tight_layout_kwargs={"pad": 2},
        )
        assert isinstance(ax, plt.Axes)
