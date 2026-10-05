# Contributing

Use a branch for each change.
Add a regression test for changed behavior.
Use synthetic input records.
Do not run tests against personal databases, browser cookies, or live write endpoints.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check --select E9,F63,F7,F82 declip tests
```

Keep README claims tied to source code or tests.
Describe test coverage and limits in the pull request.
