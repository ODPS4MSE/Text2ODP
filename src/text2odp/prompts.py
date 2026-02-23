from __future__ import annotations

from textwrap import dedent


def scenario_prompt(title: str, abstract: str) -> str:
    return dedent(
        f"""
        You are an ontology engineer.
        Given this paper title and abstract, produce JSON with keys:
        - scenario (string)
        - competency_questions (array of at least 5 questions)

        TITLE: {title}
        ABSTRACT: {abstract}

        Output JSON only.
        """
    ).strip()


def graph_prompt(scenario: str, competency_questions: list[str]) -> str:
    joined_cq = "\n".join(f"- {cq}" for cq in competency_questions)
    return dedent(
        f"""
        Extract a concept-relation graph in JSON from this domain scenario and competency questions.
        Return keys:
        - concepts (array of normalized concept labels)
        - relations (array of normalized relation labels)
        - triples (array of [subject, relation, object])

        SCENARIO:
        {scenario}

        COMPETENCY QUESTIONS:
        {joined_cq}

        Output JSON only.
        """
    ).strip()


def odp_prompt(scenario: str, triples: list[tuple[str, str, str]]) -> str:
    triple_lines = "\n".join(f"- ({s}, {r}, {o})" for s, r, o in triples)
    return dedent(
        f"""
        You are designing a reusable Ontology Design Pattern (ODP).
        Use the scenario and triples below and return JSON with keys:
        - pattern_name
        - intent
        - classes (array)
        - object_properties (array)
        - axioms_manchester (array)
        - ttl_fragment (string containing Turtle snippet)

        SCENARIO:
        {scenario}

        TRIPLES:
        {triple_lines}

        Output JSON only.
        """
    ).strip()
