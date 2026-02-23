from __future__ import annotations

import argparse
import json

from .llm import OllamaBackend, TransformersBackend
from .pipeline import Text2ODPPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text2ODP pipeline")
    parser.add_argument("--query", type=str, required=True, help="Search query for paper abstracts")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    parser.add_argument("--model", type=str, default="llama3.1:8b")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "ollama":
        llm = OllamaBackend(model=args.model)
    else:
        llm = TransformersBackend(model=args.model)

    pipeline = Text2ODPPipeline(llm=llm, output_dir=args.output_dir)
    summary = pipeline.run(query=args.query, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
