from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Generic, Literal, Optional, Self, Type, TypeVar, Union

from rdkit import Chem

from prolif.fingerprint import Fingerprint
from prolif.interactions.base import Interaction
from prolif.interactions.interactions import PiStacking
from prolif.residue import Residue
from prolif.types import Angles, Geometry, Pattern, Specs, is_geometry, is_pattern

T = TypeVar("T")


@dataclass
class DebugResult(Generic[T]):
    interaction: str
    parameter: str
    value: T
    explanation: str
    target: Optional[Literal["ligand", "protein"]] = None


@dataclass
class DebugAction:
    interaction: str
    parameter: str
    attr: str

    @abstractmethod
    def debug(
        self, ligand_residue: Residue, protein_residue: Residue
    ) -> Optional[DebugResult]:
        raise NotImplementedError()


@dataclass
class DebugPattern(DebugAction):
    target: Literal["ligand", "protein"]
    qmol: Chem.Mol

    @classmethod
    def from_spec(
        cls,
        parameter: str,
        interaction: Interaction,
        spec: Pattern,
        target: Optional[Literal["ligand", "protein"]] = None,
    ) -> list[Self]:
        if target is not None:
            attr = parameter if (n := getattr(spec, target)) is None else n
            qmol = getattr(interaction, attr)
            itype = interaction.__class__.__name__
            return [
                cls(
                    interaction=itype,
                    parameter=parameter,
                    attr=attr,
                    target=target,
                    qmol=qmol,
                )
            ]
        targets = []
        if spec.ligand:
            targets.append("ligand")
        if spec.protein:
            targets.append("protein")
        return [cls.from_spec(parameter, interaction, spec, t)[0] for t in targets]

    def debug(
        self, ligand_residue: Residue, protein_residue: Residue
    ) -> Optional[DebugResult]:
        target = ligand_residue if self.target == "ligand" else protein_residue
        if target.HasSubstructMatch(self.qmol):
            return None
        smarts = Chem.MolToSmarts(self.qmol)
        return DebugResult(
            interaction=self.interaction,
            parameter=self.parameter,
            value=smarts,
            target=self.target,
            explanation=(
                f"{self.target} does not match {self.interaction} parameter"
                f" {self.parameter}={smarts!r}"
            ),
        )


@dataclass
class DebugDistance(DebugAction):
    cls: Type[Interaction]
    distance: float
    padding: float

    @classmethod
    def from_spec(
        cls, parameter: str, interaction: Interaction, spec: Geometry, padding: float
    ) -> Self:
        icls = interaction.__class__
        itype = icls.__name__
        attr = spec.attr or parameter
        distance: float = getattr(interaction, attr)
        return cls(
            interaction=itype,
            parameter=parameter,
            attr=attr,
            distance=distance,
            cls=icls,
            padding=padding,
        )

    def debug(
        self, ligand_residue: Residue, protein_residue: Residue
    ) -> Optional[DebugResult]:
        padded_distance = self.distance + self.padding
        interaction = self.cls(**{self.parameter: padded_distance})
        metadata = next(interaction.detect(ligand_residue, protein_residue), None)
        if metadata is None:
            return DebugResult(
                interaction=self.interaction,
                parameter=self.parameter,
                value=self.distance,
                explanation=(
                    f"Could not find {self.interaction} interaction for parameter"
                    f" {self.parameter}={padded_distance}"
                ),
            )
        return None


@dataclass
class DebugAngles(DebugAction):
    cls: Type[Interaction]
    angles: Angles
    padding: Angles

    @classmethod
    def from_spec(
        cls, parameter: str, interaction: Interaction, spec: Geometry, padding: Angles
    ) -> Self:
        icls = interaction.__class__
        itype = icls.__name__
        attr = spec.attr or parameter
        angles: Angles = getattr(interaction, attr)
        return cls(
            interaction=itype,
            parameter=parameter,
            attr=attr,
            angles=angles,
            cls=icls,
            padding=padding,
        )

    def debug(
        self, ligand_residue: Residue, protein_residue: Residue
    ) -> Optional[DebugResult]:
        padded_angles = (
            self.angles[0] - self.padding[0],
            self.angles[1] + self.padding[1],
        )
        interaction = self.cls(**{self.parameter: padded_angles})
        metadata = next(interaction.detect(ligand_residue, protein_residue), None)
        if metadata is None:
            return DebugResult(
                interaction=self.interaction,
                parameter=self.parameter,
                value=self.angles,
                explanation=(
                    f"Could not find {self.interaction} interaction for parameter"
                    f" {self.parameter}={padded_angles}"
                ),
            )
        return None


class InteractionDebugger:
    def __init__(
        self,
        fp: Fingerprint,
        distance_padding: float = 0.5,
        angles_padding: Union[float, Angles] = 5,
    ) -> None:
        self.fp = fp
        self.distance_padding = distance_padding
        self.angles_padding = (
            (angles_padding, angles_padding)
            if isinstance(angles_padding, float)
            else angles_padding
        )

        interactions: dict[str, Interaction] = {}
        for itype in fp.interactions:
            int_obj: Interaction = getattr(fp, itype.lower())
            if isinstance(int_obj, PiStacking):
                interactions["EdgeToFace"] = int_obj.etf
                interactions["FaceToFace"] = int_obj.ftf
            interactions[itype] = int_obj
        self.interactions = interactions

        self.signatures = {
            interaction: signature(interaction.__init__)
            for interaction in interactions.values()
        }
        self.parse_specs()

    @staticmethod
    def spec_from_parameter(param: Parameter) -> Optional[Specs]:
        return getattr(param.annotation, "__metadata__", (None,))[0]

    def parse_specs(self):
        specs: dict[str, dict[str, Specs]] = {}
        for itype, interaction in self.interactions.items():
            if param_specs := self.get_param_specs(interaction):
                specs[itype] = param_specs
        self.specs = specs

    def get_param_specs(self, interaction: Interaction) -> dict[str, Specs]:
        sig = self.signatures[interaction]
        return {
            name: spec
            for name, param in sig.parameters.items()
            if (spec := self.spec_from_parameter(param))
        }

    def debug(
        self,
        ligand_residue: Residue,
        protein_residue: Residue,
        interaction: Optional[str] = None,
    ) -> list[DebugResult]:
        results = []
        for actions in self.gather_actions(interaction).values():
            for action in actions:
                if (
                    result := action.debug(ligand_residue, protein_residue)
                ) is not None:
                    results.append(result)
                    break
        return results

    def gather_actions(
        self, interaction_name: Optional[str] = None
    ) -> dict[str, list[DebugAction]]:
        actions: dict[str, list[DebugAction]] = defaultdict(list)

        if interaction_name is None:
            for interaction_name in self.interactions:
                acts = next(iter(self.gather_actions(interaction_name).values()))
                actions[interaction_name].extend(acts)
                return dict(actions)

        interaction = self.interactions[interaction_name]
        specs = self.specs[interaction_name]
        for parameter, spec in specs.items():
            if is_pattern(spec):
                actions[interaction_name].extend(
                    DebugPattern.from_spec(parameter, interaction, spec)
                )
            elif is_geometry(spec):
                if spec.type == "distance":
                    actions[interaction_name].append(
                        DebugDistance.from_spec(
                            parameter,
                            interaction,
                            spec,
                            padding=self.distance_padding,
                        )
                    )
                else:
                    actions[interaction_name].append(
                        DebugAngles.from_spec(
                            parameter,
                            interaction,
                            spec,
                            padding=self.angles_padding,
                        )
                    )
        return dict(actions)
