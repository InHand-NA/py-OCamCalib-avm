# Repository Guidelines

## Project Structure & Module Organization
- `src/pyocamcalib/core/` houses the numerical calibration routines; keep reusable math there and only surface it publicly when required.
- `src/pyocamcalib/modelling/` contains camera abstractions, while `src/pyocamcalib/script/` hosts the Typer CLIs (`calibration_script.py`, `projection_conversion_script.py`) that should stay as thin wrappers.
- Generated checkpoints live under `src/pyocamcalib/checkpoints/`, `test_images/` supplies regression chessboards, and `docs/` stores explanatory figures.

## Build, Test, and Development Commands
```bash
conda env create --file environment.yml && conda activate py-OCamCalib
pip install -e .
python src/pyocamcalib/script/calibration_script.py test_images/fish_1 8 6 --camera-name demo --check
python src/pyocamcalib/script/projection_conversion_script.py test_images/fish_1/Fisheye1_1.jpg src/pyocamcalib/checkpoints/calibration/example.json 80 700 700
pytest
```
Prefer Typer flags so `--help` stays authoritative; avoid environment-variable-only toggles.

## Coding Style & Naming Conventions
- Follow PEP 8 (4-space indent, 88–100 char lines) and favor vectorized NumPy. Use lower_snake_case for functions/variables, UpperCamelCase for classes, and prefix internal helpers with `_`.
- Keep type hints and docstrings current. Call out array shapes and units (pixels, mm, radians) whenever ambiguity exists.
- Calibration outputs should follow `calibration_<camera>_<YYYYMMDD_HHMMSS>.json` to match the artifacts already in checkpoints.

## Testing Guidelines
- Prefer `pytest` and mirror the module layout (`tests/core/test_intrinsec.py`, `tests/script/test_cli.py`). Name tests after the behavior protected (`test_partial_extrinsics_handles_manual_center`).
- Use fixtures that reference `test_images/` via relative paths, downsampling or mocking IO when large assets slow CI. Target ≥90 % coverage on new modules and explain any unavoidable gaps in the PR.

## Commit & Pull Request Guidelines
- Favor Conventional-style messages seen in history (`fix(calibration): …`) using `<type>(<scope>): <summary>` where scope maps to folders (`core`, `script`, `docs`).
- Separate behavior changes from refactors, describe numerical method updates in the commit body, and cite papers or equations when altering calibration math.
- PRs should list validation commands, produced artifacts (e.g., checkpoint filenames), linked issues, and visuals when projection outputs change. Request review whenever public APIs or checkpoint schemas move.

## Calibration Assets & Configuration
- Store generated JSON/pickle files under `src/pyocamcalib/checkpoints/`, omit sensitive customer captures, and document sensor plus chessboard specs in the PR for reproducibility.
- Surface every new configuration flag both in the Typer CLI help text and in `README.md` to keep tooling and docs synchronized.
