# Text2ODP

Text2ODP is a reproducible Python pipeline that converts scientific text (e.g., paper abstracts) into **Ontology Design Patterns (ODPs)** using open-source LLMs.

## Research Goal

Given a corpus of paper abstracts, Text2ODP performs:
1. **Scenario elicitation** + **competency question generation**.
2. **Concept and relation extraction** into a semantic graph.
3. **Ontology Design Pattern synthesis** (classes, properties, axioms, Turtle fragment).
4. **Quantitative evaluation** with publication-friendly metrics.

This repository is designed to be a strong starting point for scientific studies on LLM-assisted ontology engineering.

## Architecture

```text
Paper abstracts (Semantic Scholar API)
        ↓
LLM step 1: scenario + competency questions (JSON)
        ↓
LLM step 2: concepts/relations/triples (JSON)
        ↓
LLM step 3: ODP synthesis (JSON + TTL snippet)
        ↓
Evaluation:
- lexical_coverage
- graph_density
- cq_answerability_proxy
- self_consistency
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional LLM extras:

```bash
pip install -e .[llm]
```

## Run

### Option A: Ollama backend (recommended for local open-source models)

1. Install Ollama and pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
2. Run pipeline:
   ```bash
   text2odp --query "ontology engineering healthcare" --limit 10 --backend ollama --model llama3.1:8b
   ```

### Option B: Transformers backend

```bash
text2odp --query "knowledge graph construction" --limit 5 --backend transformers --model mistralai/Mistral-7B-Instruct-v0.3
```

## Outputs

Generated under `outputs/`:
- `dataset.jsonl`: collected papers and abstracts.
- `artifacts.json`: scenario/CQ/graph/ODP/evaluation per paper.
- `evaluation.csv`: paper-level metrics.
- `evaluation_summary.json`: aggregate metrics.

## Evaluation Design (publication-oriented)

Current implemented metrics:
- **Lexical coverage**: overlap between extracted concepts and abstract lexicon.
- **Graph density**: structural richness of concept graph.
- **CQ answerability proxy**: proportion of CQs touching extracted concept/relation vocabulary.
- **Self-consistency**: overlap between generated ODP classes and extracted concepts.

For a publishable paper, extend with:
- Human expert annotation (inter-rater reliability, Cohen's/Fleiss' kappa).
- Baselines: rule-based IE, non-LLM ontology extraction, and alternative LLMs.
- Statistical significance tests (paired bootstrap / Wilcoxon signed-rank).
- Robustness checks across domains, model sizes, and prompt variants.

## Repeated experiments

Use `scripts/run_experiment.py` to aggregate multiple run folders:

```bash
python scripts/run_experiment.py --results-root outputs
```

Expected directory structure example:

```text
outputs/
  run_1/evaluation.csv
  run_2/evaluation.csv
  run_3/evaluation.csv
```

## Reproducibility checklist

- Fix random seeds (if you add sampling-heavy components).
- Log exact model name and version.
- Archive prompts and all generated JSON artifacts.
- Report hardware setup and decoding parameters.
- Provide failure analysis and representative error cases.

## Ethical and scientific considerations

- LLM-generated ontologies may contain hallucinations and domain bias.
- Never deploy ODPs in safety-critical domains without expert validation.
- Use this pipeline as a semi-automatic assistant, not a fully autonomous authority.

