from text2odp.evaluation import aggregate, cq_answerability_proxy, lexical_coverage
from text2odp.schemas import EvaluationResult


def test_lexical_coverage_non_zero() -> None:
    score = lexical_coverage("Knowledge graph extracts patient diagnosis", ["patient", "diagnosis"])
    assert 0 < score <= 1


def test_cq_answerability_proxy() -> None:
    score = cq_answerability_proxy(
        ["Which patient has diagnosis X?"],
        ["Patient", "Diagnosis"],
        ["hasDiagnosis"],
    )
    assert score == 1.0


def test_aggregate_returns_stats() -> None:
    results = [
        EvaluationResult(
            paper_id="1",
            lexical_coverage=0.5,
            graph_density=0.2,
            cq_answerability_proxy=0.8,
            self_consistency=0.4,
        ),
        EvaluationResult(
            paper_id="2",
            lexical_coverage=0.7,
            graph_density=0.3,
            cq_answerability_proxy=0.6,
            self_consistency=0.6,
        ),
    ]
    summary = aggregate(results)
    assert "lexical_coverage_mean" in summary
    assert "self_consistency_std" in summary
