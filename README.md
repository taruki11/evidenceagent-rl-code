# EvidenceAgent-RL

> Code-only source snapshot. No models, datasets, logs, or credentials are included.

EvidenceAgent-RL is an Agentic-RL research engineering project for evidence-grounded paper question answering. It studies auditable training and evaluation workflows that incorporate evidence boundaries, retrieval strategy memory, and tool interaction.

## Offline quickstart

Use Python 3.11+ in a fresh virtual environment. The package has no runtime
dependencies, so nothing beyond the standard library is needed after install:

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

The demo exits 0 only when the accepted run succeeds, the cross-episode answer is
blocked, and replay reproduces the same state. It writes no file unless `--output`
is given, and it never overwrites one: an existing path exits 2 with a message and
the original bytes are untouched.

## What this demo lets you verify

Five behaviours, all deterministic and offline:

1. A supported claim is accepted once its evidence handle has been citation-checked.
2. A claim cannot be supported by an unchecked handle, a fabricated handle, or a
   handle whose DOI is outside the task's allowed set.
3. Evidence handles do not cross episode boundaries, and search stays inside the
   task's corpus scope.
4. Tool budget never goes below zero, and an action rejected by argument
   validation or by an insufficient budget is not charged. A rejection raised
   *inside* a tool handler is charged, because the budget is debited before the
   handler runs.
5. A serialised action trace replays to a state equal to the original one.

`tests/test_harness_contract.py` covers these boundaries as repository regression
tests: unchecked citation, cross-episode handle, budget floor, free invalid
arguments, corpus scope, and replay equality.

Coverage is not uniform across item 2. The demo itself exercises the unchecked
handle and the cross-episode handle. The fabricated-handle and disallowed-DOI cases
were confirmed by manual inspection of the harness, but neither has a dedicated
regression test in this repository yet, so they are weaker evidence than the
tested cases above.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'evidence_agent_rl'` | The package is not installed in the interpreter you are running. | `python -m pip install -e .` from the repository root. In a source checkout without installing, point `PYTHONPATH` at `src`. |
| `python -m unittest discover` prints `NO TESTS RAN` | Discovery starts at the working directory and does not descend into `tests/` here. | Use `python -m unittest discover -s tests -v` as shown above. |
| `test_installed_module_cli_emits_json` fails before installation | That test starts a subprocess that imports the *installed* package. | Install first. The suite is meant to run against an installed package, not a bare checkout. |
| `pip install` fails under "Installing build dependencies" | The build fetches `setuptools>=77`, which needs network access. | On an offline machine, install `setuptools>=77` first and run `python -m pip install --no-build-isolation -e .`. |
| `pip install` rejects `project.license` / "License classifiers have been superseded" | `setuptools` older than 77 does not understand the SPDX license metadata. | Upgrade `setuptools`. This only happens when build isolation is disabled. |
| `demo.py: error: Cannot create output file: ... File exists` | `--output` never overwrites by design. | Pick a new path. Exit code 2; the existing file is unchanged. |

## Research tooling (existing snapshot)

- Retrieval and tool-use environment
- Evidence-constrained policy and memory mechanisms
- Training, frozen selection, and evaluation gates
- Failure-analysis and result-audit utilities

## Core mechanisms added (Nov 2026)

- `src/evidence_agent_rl/public_contract.py` — question-derived public support contract (public task question and candidate titles only; zero gold exposure).
- `src/evidence_agent_rl/claim_support.py` — answer-time lexical and numeric claim-support guard (`claim_support_guard`).
- `src/evidence_agent_rl/harness.py` — mechanical two-document baseline (`require_distinct_support_documents`) and bounded non-terminal retry loop (`support_repair` / `support_repair_max_hints`).
- `scripts/audit_hv_failure_funnel.py` — zero-GPU read-only 6-bucket failure-funnel replay auditor over executed diagnostic `predictions.jsonl` files.

## Diagnostic findings summary (pre-registered controls, Nov 2026)

All measurements below are controlled diagnostic results on 4B BF16 LoRA (`Qwen3-4B-Instruct-2507` + frozen V8 SFT adapter) across disjoint 120-task all-2-gold HotpotQA direct-evidence subsets (`hv-rep120` and `hv-clean120`) and a 10,212-episode offline replay corpus. They are diagnostic findings, not SOTA or formal held-out capability claims.

- **Mechanical `min-2` matches or beats the question-derived contract:** On all-2-gold `hv-rep120`, mechanical `blanket min-2` achieved `16/120` (rep1) and `14/120` (rep2) cited Joint EM vs `14/120` and `14/120` for the question-derived contract (exact McNemar `p = 0.79 / 1.0`), closing the contract selectivity hypothesis on all-2-gold sets.
- **`Model × Harness` 2×2 attribution & fresh-set check:** Without the harness, `V8-off` scored `0/120` (rep1) and `1/120` (rep2) cited Joint EM despite ~47% Answer F1 because it terminated after citing a single document. Under `min-2`, V8 beat base on the design set `hv-rep120` (`15/14` vs `7/5`, `p = 0.0215 / 0.0117`), but this advantage did **not** replicate on the disjoint fresh set `hv-clean120` (`17/17` vs `12/10`, `p = 0.332 / 0.143`).
- **Bounded `support_repair` reduces abstention, not citation EM:** Adding 2 non-terminal repair hints to `V8 + min-2` lowered empty answers by **11.7 / 10.8 pp** (`39/39 → 25/26`, `p < 0.001`) without a significant gain in cited Joint EM (`14/15 → 14/18`, `p = 1.0 / 0.375`).
- **Claim-support guard (offline precision vs online A/B gap):** Offline replay across 10,212 episodes / 9,989 claims flagged 14.2% unsupported claims (including 689 missing numbers) with `0/1,238` false positives on strict successes (`95.1%` precision vs verifier), while the live multi-hop A/B primary came out null (`0/16` vs `0/16`).
- **6-bucket failure funnel across 28 arms:** Every single arm scored `0/33` (`hv-rep120`) and `0/38` (`hv-clean120`) `sp_em` on tasks with 3+ gold sentences when policies selected one span per document, isolating a structural citation ceiling alongside ~68% `M1`-reachable empty rejections.

## Data and results

A public repository should contain source, generic configurations, and only approved examples. Exclude formal test sets, private corpora, trajectories, checkpoints, and evidence bundles. Interpret every score under its experiment protocol; do not present diagnostics as formal holdout performance.

## License

Code is licensed under [MIT](LICENSE). External corpora, models and datasets retain their own terms and are not included.


