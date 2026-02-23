from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_eval_csv(path: Path, rows: list[dict[str, str]]) -> None:
    header = "paper_id,lexical_coverage,graph_density,cq_answerability_proxy,self_consistency,notes\n"
    lines = [
        f"{r['paper_id']},{r['lexical_coverage']},{r['graph_density']},{r['cq_answerability_proxy']},{r['self_consistency']},{r['notes']}"
        for r in rows
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def test_summarize_runs(tmp_path: Path) -> None:
    run1 = tmp_path / "run_1"
    run2 = tmp_path / "run_2"
    run1.mkdir()
    run2.mkdir()

    _write_eval_csv(
        run1 / "evaluation.csv",
        [
            {
                "paper_id": "p1",
                "lexical_coverage": "0.5",
                "graph_density": "0.2",
                "cq_answerability_proxy": "0.8",
                "self_consistency": "0.7",
                "notes": "ok",
            }
        ],
    )
    _write_eval_csv(
        run2 / "evaluation.csv",
        [
            {
                "paper_id": "p2",
                "lexical_coverage": "0.7",
                "graph_density": "0.4",
                "cq_answerability_proxy": "0.6",
                "self_consistency": "0.9",
                "notes": "ok",
            }
        ],
    )

    module = _load_module(Path("scripts/run_experiment.py"))
    summary = module.summarize_runs(tmp_path)

    assert summary["lexical_coverage_mean"] == 0.6
    assert summary["graph_density_mean"] == 0.3
    assert "self_consistency_std" in summary
