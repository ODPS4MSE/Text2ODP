from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def summarize_runs(root: Path) -> dict[str, float]:
    per_run: list[dict[str, float]] = []
    for csv_file in root.glob("run_*/evaluation.csv"):
        rows = []
        with csv_file.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items() if k not in {"paper_id", "notes"}})
        if not rows:
            continue
        per_run.append({metric: mean(r[metric] for r in rows) for metric in rows[0].keys()})

    if not per_run:
        return {}

    summary: dict[str, float] = {}
    metrics = per_run[0].keys()
    for metric in metrics:
        values = [run[metric] for run in per_run]
        summary[f"{metric}_mean"] = round(mean(values), 4)
        summary[f"{metric}_std"] = round(pstdev(values), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="outputs")
    args = parser.parse_args()

    root = Path(args.results_root)
    summary = summarize_runs(root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
