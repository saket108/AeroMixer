# Release Checklist

## Versioning
- Update `CHANGELOG.md` with release notes.
- Bump release version tag (`vX.Y.Z`).
- Confirm `git status` is clean before tagging.
- Run: `python scripts/release_tools.py check --version X.Y.Z`

## Reproducibility
- Run `scripts/pipeline.py --mode run --dry-run` on a tiny dataset.
- Confirm `pipeline_manifest.json` includes:
  - resolved command
  - git commit hash
  - dataset fingerprint
  - resolved dataset plan and hyperparameters
- Confirm benchmark row append works (`benchmarks/summary.csv`).
- If threshold sweep is enabled, confirm `threshold_tuning.json` is produced.

## Quality Gates
- CI passes on `main`:
  - lint (`ruff`)
  - format check (`black --check`)
  - smoke tests (`python -m unittest`)
- Run one end-to-end tiny run locally.

## Packaging
- Build Docker image successfully.
- Verify one documented inference command works in container.

## Tagging
- Create annotated tag:
  - `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- Push tag:
  - `git push origin vX.Y.Z`
- Or use helper:
  - `python scripts/release_tools.py tag --version X.Y.Z`
