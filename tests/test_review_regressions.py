"""Regression targets for a corrected nbpearson release.

Run against an explicitly installed package, not by importing src/ from the archive:
    python -m unittest -v test_review_regressions

The two grouped complex tests and compressed-admission test expose defects in
0.1.0.dev0. Real-only rejection is accepted only when consistent across all four
products; correct complex arithmetic is the other accepted contract.
"""
import unittest
import warnings
from unittest.mock import patch
import numpy as np
from scipy import sparse
from nbpearson import build_operator
import nbpearson._api as api


class ReviewRegressions(unittest.TestCase):
    def test_compressed_admission_uses_compressed_storage(self):
        n = 100_000
        x = sparse.csr_matrix(
            (np.where(np.arange(n) % 2 == 0, 1., 4.),
             (np.arange(n), np.arange(n) % 2)), shape=(n, 2))
        # This isolates resource-estimation arithmetic. It is NOT a measured RSS
        # claim and does not disable input validation or mathematical selection.
        with patch.object(api, 'process_baseline_bytes', return_value=0):
            large = build_operator(x, representation='interpolated_grouped', clip=None)
            estimate = large.diagnostics['resources']['estimated_peak_bytes']
            self.assertLessEqual(estimate, 128 * 2**20)
            small = build_operator(x, representation='interpolated_grouped', clip=None,
                                   memory_budget_bytes=128 * 2**20)
        self.assertEqual(large.diagnostics['analytic_selection']['interpolation_nodes'],
                         small.diagnostics['analytic_selection']['interpolation_nodes'])
        v = np.array([1., -2.])
        np.testing.assert_allclose(large.operator @ v, small.operator @ v,
                                   rtol=1e-13, atol=1e-13)

    def _complex_contract(self, representation):
        x = sparse.csr_matrix([[1.,2.,0.],[0.,5.,1.],[3.,0.,2.],[1.,4.,1.]])
        op = build_operator(x, representation=representation, clip=None).operator
        a = x.toarray(); d = a.sum(1); p = a.sum(0)/a.sum()
        mu = d[:, None] * p[None, :]
        oracle = (a-mu)/np.sqrt(mu+mu*mu/100.)
        oracle -= oracle.mean(0)
        contracts = []
        for transpose, block in [(False,False),(True,False),(False,True),(True,True)]:
            n = x.shape[0] if transpose else x.shape[1]
            values = np.arange(n)*(1+2j)+3j
            if block: values = np.column_stack((values, 2*values))
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                try:
                    actual = op.T @ values if transpose else op @ values
                except (TypeError, ValueError):
                    contracts.append('explicit_rejection')
                    continue
            expected = oracle.T @ values if transpose else oracle @ values
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
            self.assertFalse(any('discards the imaginary' in str(w.message) for w in recorded))
            contracts.append('correct_complex')
        self.assertEqual(len(set(contracts)), 1, 'Contract must be consistent across four products')

    def test_complex_interpolated(self):
        self._complex_contract('interpolated')

    def test_complex_interpolated_grouped(self):
        self._complex_contract('interpolated_grouped')

    def test_complex_exact_grouped(self):
        self._complex_contract('exact_grouped')


if __name__ == '__main__':
    unittest.main(verbosity=2)
