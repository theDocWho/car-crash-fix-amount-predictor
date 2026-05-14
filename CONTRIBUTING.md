# Contributing

This is a capstone project; the workflow is deliberately strict so `main` stays
clean and every change is reviewable.

## Branch model

* **`main`** is protected by convention — never push to it directly. It only
  moves via reviewed merges from a feature branch.
* **Feature branches** are named `checkpoint-<N>-<short-desc>` where:
  * `<N>` is a monotonically increasing integer (so the history reads in order).
  * `<short-desc>` is a 2–4 word kebab-case summary, e.g.
    `checkpoint-2-refactor-and-docs`, `checkpoint-3-fastapi-demo`.
* The first checkpoint (`checkpoint-1-baseline`) snapshots the state of `main`
  at v0.1.0 so we have an immutable reference point before any new work.

## Workflow

```bash
# 1. Find the next N
git fetch origin
git branch -r | grep -E 'origin/checkpoint-[0-9]+-' | sort
# 2. Branch off main
git checkout main && git pull
git checkout -b checkpoint-<N>-<short-desc>
# 3. Make changes, run tests, commit
pytest -q
git commit -m "..."
# 4. Push and open a PR
git push -u origin checkpoint-<N>-<short-desc>
gh pr create --base main --title "..." --body "..."
# 5. Reviewer merges via the GitHub UI / `gh pr merge`
```

## Commit hygiene

* Run `pytest -q` before each commit. We require all tests to pass on `main`.
* Keep commits scoped — one concern per commit; bigger changes go in multiple
  commits on the same branch.
* No secrets, no large artifacts. Model weights belong in GitHub releases; raw
  data lives in `data/raw/` which is gitignored.

## Code style

The project optimises for **readable, small files** over clever abstractions.
Concrete preferences:

* **KISS** — prefer two clear lines over one clever expression. Each module
  should be readable end-to-end in under 5 minutes.
* **Single responsibility** — files do one job. `ccdp.utils.device` only picks
  devices; `ccdp.utils.transforms` only builds torchvision pipelines.
* **Shared base classes only when there's real duplication.**
  `BaseVariantPipeline` exists because Variant A and B genuinely share the
  XGBoost + FX + provenance flow; we don't manufacture base classes for things
  that happen to look similar.
* **Docstrings on every module** explain *what* the module is for and *why* it
  exists separately from its neighbours, not how every line works.
* **Type hints on public surfaces.** Internal helpers can skip annotations when
  the type is obvious from context.
* **No emojis in source code** (PRs and chat are fine).

## Testing

* `pytest -q` runs the full suite — currently 64 tests.
* Tests live in `tests/` and use only `pytest` (no plugins beyond
  `pytest-cov` which is optional).
* Network / GPU / large-dataset tests are skipped when their prerequisites are
  absent (see the `@pytest.mark.skipif` guards on Stanford Cars / iaai tests).
* New behaviour needs at least one test; complex new logic needs a few.

## Trained models

* **Don't commit weights to git.** Use GitHub Releases.
* Each release version corresponds to a tag on `main`. Use `gh release create
  vX.Y.Z` with weights attached as assets.
* Promotion via `ccdp registry promote <run_id> <variant>` is the only thing
  that should move the `checkpoints/production/<variant>.pt` symlink.

## Project structure

See [README.md](README.md) for the source-tree layout and execution flow.
For the rationale behind each phase, see [PLAN.md](PLAN.md) and the per-phase
status docs under [progress/](progress/).
