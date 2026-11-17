# nbpearson

**Public source repository for nbpearson 0.1.0.dev1.** This repository contains the MIT-licensed software source, with `Copyright (c) 2026 Mohamed Kone`. The numerical implementation evaluated for the manuscript is preserved at commit [`c0f41107f888d69713c8e22e488bf1e7386f8c4d`](https://github.com/mohkone/nbpearson/commit/c0f41107f888d69713c8e22e488bf1e7386f8c4d). This source repository does not contain biological datasets, benchmark archives, private author records, or experiment runtimes.

nbpearson 0.1.0.dev1 supplies three explicitly selected CPU/binary64 linear operators for fixed negative-binomial Pearson residuals of sparse integer counts. The five Python modules preserve evaluated dev1 source. Packaging and supporting materials are new; a rebuilt wheel has its own hash.

## Target and representations

For observations × genes counts `X`, positive depths `n`, positive proportions `p` and one finite scalar `theta > 0`:

```text
mu[i,j] = n[i] * p[j]
R[i,j] = (X[i,j] - mu[i,j]) / sqrt(mu[i,j] + mu[i,j]**2 / theta)
A = H clip(R, -c, c)
```

`H` centers columns after clipping, without scaling genes by standard deviation. `clip=None` selects the unclipped target. Finite clipping requires `c >= sqrt(theta)`.

| Required `representation` | Stored background | Meaning |
|---|---|---|
| `interpolated` | N×r and G×r factors | Interpolated background |
| `interpolated_grouped` | U×r and G×r factors, group metadata | Same interpolant, sharing coefficients at exactly equal binary64 depths |
| `exact_grouped` | U×G table, group metadata | No interpolation error; still floating-point arithmetic |

U counts distinct depths; r counts interpolation nodes, separately from PCA dimension. Counts/observations are not aggregated. Grouped centering uses multiplicities; transpose products sum within groups. There is no dispatcher. Background dimensions exclude sparse correction, input, temporary and solver memory.

## Install and run

Python >=3.10 is inherited metadata. Candidate checks use Python 3.12.3, NumPy 2.5.4 and SciPy 1.18.1; other combinations are untested here. NumPy and SciPy are the only runtime dependencies. `requirements-tested.txt` records tested runtime versions and `build-requirements-tested.txt` records build versions.

With compatible dependencies/build tools available, run here:

```bash
python3 -m pip install --no-index --no-deps --no-build-isolation .
python3 examples/operator_products.py
python3 examples/operator_products.py --pca
python3 -m unittest discover -s tests -v
```

[INSTALL.md](INSTALL.md) provides environment, wheel and build commands. These use local software without assuming public PyPI availability. Candidate reports and execution receipts accompany this software tree separately.

## Minimal API

```python
import numpy as np
from scipy import sparse
from nbpearson import build_operator

X = sparse.csr_matrix(np.tile(np.array([[2, 0, 1], [1, 4, 1]], dtype=np.int64), (8, 1)))
result = build_operator(X, representation="exact_grouped", theta=100.0,
                        clip=10.0, memory_budget_bytes=512 * 2**20)
A, diagnostics = result.operator, result.diagnostics
y = A.matvec(np.array([1.0, -1.0, 0.5]))
z = A.rmatvec(np.ones(X.shape[0]))
```

`clip=10.0` is valid at theta=100 for this 16-row fixture; `clip="sqrt_n"` would be outside the supported domain. For `exact_grouped`, omit `matrix_rtol`/`max_nodes` or set them to `None`. For interpolants they default to 1e-6/128. Selection tolerance and external solver tolerance differ.

The real SciPy `LinearOperator` exposes `matvec`, `rmatvec`, `matmat`, `rmatmat`. Complex operands raise `TypeError`. nbpearson has no PCA solver; optional examples use SciPy `svds`. The tiny reproducibility example checks all four products against the direct formula and optionally runs an external solver. Its outputs are functional checks, not a scientific benchmark.

## Input, selection and resource contracts

Inputs are two-dimensional SciPy CSR matrices/arrays with dimensions >=2 and finite, real, nonnegative, exactly integer-valued counts. Negative entries are rejected before duplicate summation. Canonicalization uses a private binary64 copy; input is preserved. Zero margins are rejected. External axis removal must retain mappings and use the final universe.

Default margins are calculated once in binary64, without promising exact integer sums. `FixedMargins` and `counts_fingerprint` bind supplied positive binary64 margins to canonical counts without silently recalculating them.

Diagnostics distinguish representation, margins/clipping, nodes/degree, analytic selection, owned-array storage and estimated resources. Constructor `floating_validation="NOT_EVALUATED"` is unchanged by products or a converged SVD. Analytic selection concerns exact arithmetic; computed L is not a certified lower bound and there is no end-to-end floating-point certificate. Near-null cancellation and spectral gaps remain relevant.

`SelectionError`/`ResourceLimitError` retain diagnostics. There is no hidden fallback or denominator epsilon. The budget, at most 512 MiB, estimates admission including the current process baseline. It is not an RSS guarantee or watchdog; external solver work and caller-selected block widths may exceed its assumptions. Do not replace the production baseline with zero to claim a memory guarantee.

## Tests, examples and layout

All five original dev1 test modules remain. Autonomous cases use synthetic inputs, small Decimal references and `unittest`. Four historical-fidelity cases need optional development archives, which are excluded; they explicitly skip without them. Skips are not passes. To isolate the suite from adjacent archives:

```bash
NBPEARSON_HISTORY_ROOT=/tmp/nbpearson-no-history python3 -m unittest discover -s tests -v
```

`examples/operator_products.py` demonstrates four products, diagnostics, formula agreement and optional `--pca`. `examples/expected_operator_products.json` records the checked structural outputs with and without PCA; numerical checks use tolerances in the example. The original `examples/minimal_counts.py` is preserved and uses separate synthetic counts for 21 computed/20 retained triplets. Biological data is unnecessary.

```text
src/nbpearson/        unchanged five-module dev1 implementation
tests/               autonomous and optional historical tests
examples/            synthetic examples and expected outputs
pyproject.toml       build/package metadata
requirements-tested.txt  tested runtime versions
build-requirements-tested.txt  tested backend versions
LICENSE              MIT licence for original code; 2026 Mohamed Kone
THIRD_PARTY_NOTICES.md    provenance/dependency terms
CITATION.cff         software citation metadata
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The implementation uses established interpolation, sparse-plus-low-rank, grouping and SciPy interfaces. No new normalization, first implicit NB-PCA method, universal time/RSS advantage or biological validation is claimed. Citation metadata contains no invented DOI or publication identifier.
