# Contributing

Python 3.11 or newer is required. Start with an isolated environment:

```sh
python -m venv .venv
# Activate the environment using your shell's normal activation command.
python -m pip install -e .
python -m unittest discover -s tests -v
python -m evidence_agent_rl.demo
```

If one of these commands fails, see the troubleshooting table in
[README.en.md](README.en.md) before opening an issue.

Use synthetic fixtures for regression tests. Report the command, Python version,
expected behavior and observed behavior when opening an issue. Keep fixes focused;
search existing issues and pull requests before starting overlapping work.

Do not submit credentials, private corpora, personal or company records, training
outputs or checkpoints. A change to an evaluation protocol must explain how it
affects comparability with earlier results. Synthetic checks are engineering
regressions, not benchmark scores or evidence of general factual correctness.

Code is licensed under [MIT](LICENSE). Submit only contributions you have the
right to distribute under those terms. External models and datasets retain their
own licenses.
