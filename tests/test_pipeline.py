from __future__ import annotations

import json

from text2odp.pipeline import Text2ODPPipeline
from text2odp.schemas import PaperRecord


class StubCollector:
    def search(self, query: str, limit: int = 20) -> list[PaperRecord]:
        return [
            PaperRecord(
                paper_id="p1",
                title="A study on patient trajectories",
                abstract="Patients receive treatments and produce outcomes in hospitals.",
                venue="TestConf",
                year=2025,
            )
        ]


class StubLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "scenario": "A hospital monitors patients, treatments, and outcomes.",
                    "competency_questions": [
                        "Which patient receives which treatment?",
                        "What outcomes are associated with treatments?",
                        "Which hospital records the treatment?",
                        "How are outcomes measured over time?",
                        "Which patient has a specific outcome?",
                    ],
                }
            )
        if self.calls == 2:
            return json.dumps(
                {
                    "concepts": ["Patient", "Treatment", "Outcome", "Hospital"],
                    "relations": ["receives", "hasOutcome", "recordedBy"],
                    "triples": [
                        ["Patient", "receives", "Treatment"],
                        ["Patient", "hasOutcome", "Outcome"],
                        ["Treatment", "recordedBy", "Hospital"],
                    ],
                }
            )
        return json.dumps(
            {
                "pattern_name": "PatientTreatmentOutcomePattern",
                "intent": "Represent how patients receive treatments and produce outcomes.",
                "classes": ["Patient", "Treatment", "Outcome", "Hospital"],
                "object_properties": ["receives", "hasOutcome", "recordedBy"],
                "axioms_manchester": [
                    "Patient SubClassOf receives some Treatment",
                    "Patient SubClassOf hasOutcome some Outcome",
                ],
                "ttl_fragment": "@prefix ex: <http://example.org/> .",
            }
        )


def test_pipeline_run_writes_artifacts(tmp_path) -> None:
    pipeline = Text2ODPPipeline(llm=StubLLM(), collector=StubCollector(), output_dir=str(tmp_path))

    summary = pipeline.run(query="patient treatment", limit=1)

    assert "lexical_coverage_mean" in summary
    assert (tmp_path / "dataset.jsonl").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert (tmp_path / "evaluation.csv").exists()
    assert (tmp_path / "evaluation_summary.json").exists()

    artifacts = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
    assert artifacts[0]["paper"]["paper_id"] == "p1"
    assert artifacts[0]["odp"]["pattern_name"] == "PatientTreatmentOutcomePattern"
