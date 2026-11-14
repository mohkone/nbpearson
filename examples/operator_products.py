"""Tiny synthetic dev1 API smoke example; no timing; optional external PCA."""
import argparse
import json

import numpy as np
from scipy import sparse

from nbpearson import __version__, build_operator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", action="store_true")
    args = parser.parse_args()
    assert __version__ == "0.1.0.dev1", "Use the evaluated dev1 artifact"
    X = sparse.csr_matrix(
        np.tile(np.array([[2, 0, 1], [1, 4, 1]], dtype=np.int64), (8, 1))
    )
    before = tuple(a.copy() for a in (X.data, X.indices, X.indptr))
    result = build_operator(
        X,
        representation="exact_grouped",
        theta=100.0,
        clip=10.0,
        memory_budget_bytes=512 * 2**20,
    )
    A, info = result.operator, result.diagnostics

    v = np.array([1.0, -1.0, 0.5])
    w = np.arange(X.shape[0], dtype=np.float64)
    V = np.column_stack((v, -v))
    W = np.column_stack((w, np.ones(X.shape[0])))
    y, z = A.matvec(v), A.rmatvec(w)
    Y, Z = A.matmat(V), A.rmatmat(W)

    assert A.shape == (16, 3)
    assert A.dtype == np.dtype("float64")
    assert y.shape == (16,) and z.shape == (3,)
    assert Y.shape == (16, 2) and Z.shape == (3, 2)
    assert all(np.isfinite(a).all() for a in (y, z, Y, Z))
    assert info["constructed_representation"] == "exact_grouped"
    assert info["U"] == 2
    assert info["analytic_selection"]["interpolation_nodes"] is None
    assert info["floating_validation"] == "NOT_EVALUATED"
    for original, current in zip(before, (X.data, X.indices, X.indptr)):
        np.testing.assert_array_equal(original, current)

    # Tiny direct binary64 formula, clipped before gene centering.
    dense = X.toarray().astype(np.float64)
    depths, totals = dense.sum(axis=1), dense.sum(axis=0)
    mu = depths[:, None] * (totals / totals.sum())
    direct = np.clip((dense - mu) / np.sqrt(mu + mu * mu / 100.0), -10.0, 10.0)
    direct -= direct.mean(axis=0)
    for actual, expected in ((y, direct @ v), (z, direct.T @ w),
                             (Y, direct @ V), (Z, direct.T @ W)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-10)
    np.testing.assert_allclose(A.rmatvec(np.ones(16)), 0.0, atol=1e-11)
    pca_status = "NOT_REQUESTED"
    if args.pca:
        from scipy.sparse.linalg import svds
        u, singular, vt = svds(A, k=1, solver="arpack", tol=1e-10,
                              maxiter=1000, v0=np.random.default_rng(6619).normal(size=3))
        # Only the nonzero component: this fixture has two row profiles.
        np.testing.assert_allclose(singular[0], np.linalg.svd(direct, compute_uv=False)[0],
                                   rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(direct @ vt.T, u * singular, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(direct.T @ u, vt.T * singular, rtol=1e-10, atol=1e-11)
        pca_status = "PASS_ONE_TRIPLET"

    print(json.dumps({
        "direct_formula_checks": "PASS_FOUR_PRODUCTS",
        "external_pca": pca_status,
        "input_unchanged": True,
        "version": __version__,
        "shape": list(A.shape),
        "representation": info["constructed_representation"],
        "distinct_depths": info["U"],
        "product_shapes": [list(a.shape) for a in (y, z, Y, Z)],
        "interpolation_nodes": info["analytic_selection"]["interpolation_nodes"],
        "floating_validation": info["floating_validation"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
