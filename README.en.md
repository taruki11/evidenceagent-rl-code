# EvidenceAgent-RL

> Code-only source snapshot. No models, datasets, logs, or credentials are included.

EvidenceAgent-RL is an Agentic-RL research engineering project for evidence-grounded paper question answering. It studies auditable training and evaluation workflows that incorporate evidence boundaries, retrieval strategy memory, and tool interaction.

## Offline quickstart

Use Python 3.11+ in a fresh virtual environment:

```sh
python -m pip install -e .
python -m evidence_agent_rl.demo
python -m unittest discover -s tests -v
```

The demo uses fictional documents and scripted actions on CPU, with no model
service or credentials. It checks evidence boundaries and deterministic replay;
it is not an LLM benchmark, semantic entailment test, or RL training result.
The broader research workflows and external datasets are not validated by it.
JSON goes to stdout; `--output demo-report.json` saves to a new file without
overwriting an existing one. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Research tooling (existing snapshot)

- Retrieval and tool-use environment
- Evidence-constrained policy and memory mechanisms
- Training, frozen selection, and evaluation gates
- Failure-analysis and result-audit utilities

## Data and results

A public repository should contain source, generic configurations, and only approved examples. Exclude formal test sets, private corpora, trajectories, checkpoints, and evidence bundles. Interpret every score under its experiment protocol; do not present diagnostics as formal holdout performance.

## Language

中文：[README.zh-CN.md](README.zh-CN.md)

## License

Code is licensed under [MIT](LICENSE). External corpora, models and datasets retain their own terms and are not included.


