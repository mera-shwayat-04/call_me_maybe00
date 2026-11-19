.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]; [shutil.rmtree(p, ignore_errors=True) for p in (pathlib.Path('.mypy_cache'), pathlib.Path('.pytest_cache'))]"

lint:
	uv run python -m flake8 . --exclude=.git,.venv,__pycache__,.mypy_cache,.pytest_cache,llm_sdk,.tmp_docs
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python -m flake8 .
	uv run python -m mypy . --strict