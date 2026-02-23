from __future__ import annotations

import csv
import json
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from .data import SemanticScholarCollector
from .evaluation import aggregate, evaluate
from .llm import JSONConstrainedMixin, LLMBackend
from .prompts import graph_prompt, odp_prompt, scenario_prompt
from .schemas import ConceptRelationGraph, ODPArtifact, PaperRecord, ScenarioAndCQs


class Text2ODPPipeline(JSONConstrainedMixin):
    def __init__(
        self,
        llm: LLMBackend,
        output_dir: str = "outputs",
        collector: SemanticScholarCollector | None = None,
    ) -> None:
        self.llm = llm
        self.collector = collector or SemanticScholarCollector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_dataset(self, query: str, limit: int = 20) -> list[PaperRecord]:
        papers = self.collector.search(query=query, limit=limit)
        dataset_path = self.output_dir / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as fp:
            for paper in papers:
                fp.write(paper.model_dump_json() + "\n")
        return papers

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
    def _call_json(self, prompt: str) -> dict:
        text = self.llm.generate(prompt)
        return self.parse_json_or_raise(text)

    def generate_for_paper(self, paper: PaperRecord) -> tuple[ScenarioAndCQs, ConceptRelationGraph, ODPArtifact]:
        scenario_data = self._call_json(scenario_prompt(paper.title, paper.abstract))
        scenario = ScenarioAndCQs.model_validate(scenario_data)

        graph_data = self._call_json(graph_prompt(scenario.scenario, scenario.competency_questions))
        graph = ConceptRelationGraph.model_validate(graph_data)

        odp_data = self._call_json(odp_prompt(scenario.scenario, graph.triples))
        odp = ODPArtifact.model_validate(odp_data)

        return scenario, graph, odp

    def run(self, query: str, limit: int = 20) -> dict[str, float]:
        papers = self.collect_dataset(query=query, limit=limit)
        evaluations = []

        records = []
        for paper in papers:
            scenario, graph, odp = self.generate_for_paper(paper)
            eval_result = evaluate(paper, scenario, graph, odp)
            evaluations.append(eval_result)
            records.append(
                {
                    "paper": paper.model_dump(),
                    "scenario": scenario.model_dump(),
                    "graph": graph.model_dump(),
                    "odp": odp.model_dump(),
                    "evaluation": eval_result.model_dump(),
                }
            )

        with (self.output_dir / "artifacts.json").open("w", encoding="utf-8") as fp:
            json.dump(records, fp, indent=2)

        summary = aggregate(evaluations)
        with (self.output_dir / "evaluation.csv").open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "paper_id",
                    "lexical_coverage",
                    "graph_density",
                    "cq_answerability_proxy",
                    "self_consistency",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(r.model_dump() for r in evaluations)

        with (self.output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        return summary
