class PyMOL:
    def __init__(self):
        self.viewer = MolViewer()
        self.cmd("set group_auto_mode, 2")

    def cmd(self, command: str) -> None:
        self.viewer.server.do(command)

    def show_interactions(self, ifp: IFP, group: str) -> None:
        lig_displayed_atom = {**Complex3D.LIGAND_DISPLAYED_ATOM, "HBDonor": 0}
        prot_displayed_atom = {**Complex3D.PROTEIN_DISPLAYED_ATOM, "HBAcceptor": 0}
        itypes = set()
        for (lres, pres), interactions in ifp.items():
            for itype, dtuple in interactions.items():
                itypes.add(itype)
                for i, data in enumerate(dtuple):
                    name = f"{group}.interactions.{itype}.{lres}_{pres}_{i}"
                    colour = Complex3D.COLORS.get(itype, "#dedede").upper()

                    lig_indices = data["parent_indices"]["ligand"]
                    prot_indices = data["parent_indices"]["protein"]

                    if itype in Complex3D.LIGAND_RING_INTERACTIONS:
                        ranks = " or ".join([f"rank {x}" for x in lig_indices])
                        lig_rank = f"({ranks})"
                        lig_centroid = True
                    else:
                        lig_i = lig_displayed_atom.get(itype, 0)
                        lig_rank = f"rank {lig_indices[lig_i]}"
                        lig_centroid = False
                    lig_sel = f"({group}.ligand and {lig_rank})"

                    if itype in Complex3D.PROTEIN_RING_INTERACTIONS:
                        ranks = " or ".join([f"rank {x}" for x in prot_indices])
                        prot_rank = f"({ranks})"
                        prot_centroid = True
                    else:
                        prot_i = prot_displayed_atom.get(itype, 0)
                        prot_i = 0
                        prot_rank = f"rank {prot_indices[prot_i]}"
                        prot_centroid = False
                    prot_sel = f"({group}.protein and {prot_rank})"

                    mode = 4 if (lig_centroid or prot_centroid) else 0
                    self.cmd(f"distance {name}, {lig_sel}, {prot_sel}, mode={mode}")
                    self.cmd(f"hide labels, {name}")
                    self.cmd(f"color {colour.replace('#', '0x')}, {name}")
        self.cmd(f"group {group}, {group}.*, add")
        self.cmd(f"group {group}.interactions, {group}.interactions.*, add")
        for itype in itypes:
            self.cmd(
                f"group {group}.interactions.{itype}, {group}.interactions.{itype}.*, add"
            )

    def reset(self) -> None:
        self.cmd("reinitialize")

    def show_mols(self, lmol: Molecule, pmol: Molecule, group: str) -> None:
        self.viewer.ShowMol(
            lmol, name=f"{group}.ligand", showOnly=False, showSticks=True, zoom=False
        )
        self.viewer.ShowMol(
            pmol, name=f"{group}.protein", showOnly=False, showSticks=True, zoom=False
        )
        self.cmd(f"set_bond stick_radius, 0.1, {group}.protein")

    def align(
        self, fixed: str, mobile: str, source: str | None = None, *targets: str
    ) -> None:
        self.cmd(f"super {mobile}, {fixed}")
        if not source:
            source = mobile
        for target in targets:
            self.cmd(f"matrix_copy {source}, {target}")
