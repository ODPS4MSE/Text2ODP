from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal


class Serializable:
    @classmethod
    def model_validate(cls, data: dict[str, Any]):
        return cls(**data)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


@dataclass
class PaperRecord(Serializable):
    paper_id: str
    title: str
    abstract: str
    venue: str | None = None
    year: int | None = None
    source: Literal["semantic_scholar", "arxiv"] = "semantic_scholar"


@dataclass
class ScenarioAndCQs(Serializable):
    scenario: str
    competency_questions: list[str] = field(default_factory=list)


@dataclass
class ConceptRelationGraph(Serializable):
    concepts: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    triples: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class ODPArtifact(Serializable):
    pattern_name: str
    intent: str
    classes: list[str] = field(default_factory=list)
    object_properties: list[str] = field(default_factory=list)
    axioms_manchester: list[str] = field(default_factory=list)
    ttl_fragment: str = ""


@dataclass
class EvaluationResult(Serializable):
    paper_id: str
    lexical_coverage: float
    graph_density: float
    cq_answerability_proxy: float
    self_consistency: float
    notes: str | None = None
