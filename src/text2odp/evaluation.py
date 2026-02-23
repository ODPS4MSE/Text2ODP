from __future__ import annotations

from collections import Counter
import math
import re

from .schemas import ConceptRelationGraph, EvaluationResult, ODPArtifact, PaperRecord, ScenarioAndCQs

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def lexical_coverage(abstract: str, concepts: list[str]) -> float:
    abstract_tokens = set(_tokenize(abstract))
    concept_tokens = set(_tokenize(" ".join(concepts)))
    if not concept_tokens:
        return 0.0
    return len(abstract_tokens.intersection(concept_tokens)) / len(concept_tokens)


def graph_density(graph: ConceptRelationGraph) -> float:
    nodes = set(graph.concepts)
    edges = {(s, o) for s, _, o in graph.triples}
    n = len(nodes)
    if n <= 1:
        return 0.0
    return len(edges) / (n * (n - 1))


def cq_answerability_proxy(cqs: list[str], concepts: list[str], relations: list[str]) -> float:
    concept_vocab = Counter(_tokenize(" ".join(concepts)))
    relation_vocab = Counter(_tokenize(" ".join(relations)))
    if not cqs:
        return 0.0

    covered = 0
    for cq in cqs:
        tokens = _tokenize(cq)
        if any(t in concept_vocab or t in relation_vocab for t in tokens):
            covered += 1
    return covered / len(cqs)


def self_consistency(odp: ODPArtifact, graph: ConceptRelationGraph) -> float:
    class_set = set(c.lower() for c in odp.classes)
    concept_set = set(c.lower() for c in graph.concepts)
    if not class_set:
        return 0.0
    overlap = len(class_set.intersection(concept_set))
    return overlap / len(class_set)


def evaluate(
    paper: PaperRecord,
    scenario: ScenarioAndCQs,
    graph: ConceptRelationGraph,
    odp: ODPArtifact,
) -> EvaluationResult:
    lc = lexical_coverage(paper.abstract, graph.concepts)
    gd = graph_density(graph)
    aq = cq_answerability_proxy(scenario.competency_questions, graph.concepts, graph.relations)
    sc = self_consistency(odp, graph)

    notes = (
        "Scores in [0,1]. High lexical coverage can be misleading for paraphrases; "
        "consider adding embedding-based metrics for publication-level evaluation."
    )
    return EvaluationResult(
        paper_id=paper.paper_id,
        lexical_coverage=round(lc, 4),
        graph_density=round(gd, 4),
        cq_answerability_proxy=round(aq, 4),
        self_consistency=round(sc, 4),
        notes=notes,
    )


def aggregate(results: list[EvaluationResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = ["lexical_coverage", "graph_density", "cq_answerability_proxy", "self_consistency"]
    out: dict[str, float] = {}
    for key in keys:
        mean = sum(getattr(r, key) for r in results) / len(results)
        std = math.sqrt(sum((getattr(r, key) - mean) ** 2 for r in results) / len(results))
        out[f"{key}_mean"] = round(mean, 4)
        out[f"{key}_std"] = round(std, 4)
    return out
