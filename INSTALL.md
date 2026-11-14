# Installation and verification

This is local release candidate V2. Original nbpearson code uses the MIT notice `Copyright (c) 2026 Mohamed Kone`, following the investigator's explicit ownership and authority assertion. Approval of the exact files for a first private push remains a separate decision. Public PyPI availability is not assumed.

## Environment

Candidate runtime checks use Python 3.12.3, NumPy 2.5.4 and SciPy 1.18.1. Build tools are setuptools 68.1.2, wheel 0.42.0 and pip 24.0. Runtime/build interpreters may be separate. Python >=3.10 is inherited metadata, not a completed compatibility matrix.

A maintainer preparing an isolated environment can run these commands from this directory using an authorized package source:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-tested.txt
python3 -m pip install -r build-requirements-tested.txt
python3 -m pip install --no-index --no-deps --no-build-isolation .
```

Dependency installation needs cached or accessible packages. The candidate mission uses existing dependencies and does not establish fresh-download availability. For an existing compatible environment, only the local install is needed.

## Build

With the recorded backend available, from this directory:

```bash
SOURCE_DATE_EPOCH=1794700800 PYTHONHASHSEED=0 TZ=UTC PYTHONDONTWRITEBYTECODE=1 python3 -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir --wheel-dir dist .
SOURCE_DATE_EPOCH=1794700800 PYTHONHASHSEED=0 TZ=UTC PYTHONDONTWRITEBYTECODE=1 python3 -c 'from setuptools.build_meta import build_sdist; build_sdist("dist")'
```

These produce new artifacts. Record hashes and compare packaged source with the evaluated wheel; equal version strings do not imply equal archive bytes. The candidate receipts record actual commands and output locations. Deterministic timestamps and backend versions do not by themselves establish reproducibility of every archive format or platform.

## Isolated wheel installation

Use a new empty target directory per artifact. From here, substitute the actual newly built wheel path:

```bash
python3 -m pip install --no-index --no-deps --no-compile --no-cache-dir --target /tmp/nbpearson-review-site dist/nbpearson-0.1.0.dev1-py3-none-any.whl
PYTHONPATH=/tmp/nbpearson-review-site python3 -B -P -c 'import nbpearson; print(nbpearson.__version__); print(nbpearson.__file__)'
PYTHONPATH=/tmp/nbpearson-review-site python3 -B -P examples/operator_products.py
PYTHONPATH=/tmp/nbpearson-review-site python3 -B -P examples/operator_products.py --pca
NBPEARSON_HISTORY_ROOT=/tmp/nbpearson-no-history PYTHONPATH=/tmp/nbpearson-review-site python3 -B -P -m unittest discover -s tests -v
```

The printed import path must belong to the selected target, not a historical source tree. NumPy/SciPy are provided by the selected runtime; the temporary target contains only the candidate package. No dependency installation is triggered by `--no-deps`.

The existing suite has 17 discovered cases, four requiring optional historical development archives. Without archives those four explicitly skip. Actual pass/fail/skip counts belong in execution receipts and are not inferred from this expectation. Autonomous cases and synthetic examples require no biological data. Expected example structural outputs are in `examples/expected_operator_products.json`; numerical tolerances/assertions are in the exact example source. No timing or biological claim is attached to these functional checks.
