"""Standalone F1/F2 regressions; no archive or research-data dependency."""
import math,unittest,warnings
from unittest.mock import patch
import numpy as np
from scipy import sparse
from nbpearson import build_operator,ResourceLimitError
import nbpearson._api as api
from nbpearson._numerics import compressed_peak_estimate


def repeated_depth_counts():
    n=100000;rows=np.arange(n)
    return sparse.csr_matrix((np.where(rows%2==0,1.,4.),(rows,rows%2)),shape=(n,2))


class MaintenanceTests(unittest.TestCase):
    def test_F1_documented_cost_and_legitimate_refusal(self):
        x=repeated_depth_counts();csr=sum(a.nbytes for a in (x.data,x.indices,x.indptr))
        self.assertEqual(compressed_peak_estimate(x.shape,csr,2,14,0),125542700)
        # Isolation of admission arithmetic only, never a process-RSS measurement.
        with patch.object(api,'process_baseline_bytes',return_value=0):
            result=build_operator(x,representation='interpolated_grouped',clip=None,memory_budget_bytes=128*2**20)
            selection=result.diagnostics['analytic_selection'];r=selection['interpolation_nodes']
            self.assertEqual(r,14)
            self.assertEqual(result.diagnostics['resources']['estimated_peak_bytes'],125542700)
            self.assertEqual(selection['estimated_peak_bytes'],125542700)
            self.assertEqual(selection['resource_model'],'interpolated_grouped')
            # Explicitly check the mathematical N (not U) in Delta_r.
            logrho=math.asinh(math.pi/math.log(4.))
            pref=math.log(4)+.5*(math.log(100.)+math.log(100000)+math.log(2))-math.log(math.expm1(logrho))
            self.assertAlmostEqual(selection['log_Delta_r'],pref-(r-1)*logrho,places=12)
            with patch.object(api,'_InterpolatedGrouped') as ctor:
                with self.assertRaises(ResourceLimitError):
                    build_operator(x,representation='interpolated_grouped',clip=None,memory_budget_bytes=110*2**20)
                ctor.assert_not_called()

    def test_F1_public_true_memory_controls_and_same_products(self):
        # No mocks, no override of the production baseline or memory checks.
        x=repeated_depth_counts()
        large=build_operator(x,representation='interpolated_grouped',clip=None)
        csr=sum(a.nbytes for a in (x.data,x.indices,x.indptr))
        expected=compressed_peak_estimate(x.shape,csr,2,14,api.process_baseline_bytes())
        budget=expected+16*2**20
        self.assertLessEqual(budget,512*2**20)
        limited=build_operator(x,representation='interpolated_grouped',clip=None,memory_budget_bytes=budget)
        self.assertGreater(limited.diagnostics['resources']['baseline_high_water_bytes'],0)
        self.assertLessEqual(limited.diagnostics['resources']['estimated_peak_bytes'],budget)
        for key in ('interpolation_nodes','polynomial_degree','pairs','L','Delta_r','relative_upper_bound'):
            self.assertEqual(large.diagnostics['analytic_selection'][key],limited.diagnostics['analytic_selection'][key])
        v=np.array([1.,-2.]);V=np.column_stack((v,2*v));w=np.linspace(-1.,1.,x.shape[0]);W=np.column_stack((w,2*w))
        for a,b in ((large.operator@v,limited.operator@v),(large.operator@V,limited.operator@V),
                    (large.operator.T@w,limited.operator.T@w),(large.operator.T@W,limited.operator.T@W)):
            np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)
        self.assertEqual(limited.diagnostics['floating_validation'],'NOT_EVALUATED')
        with self.assertRaises(ResourceLimitError):build_operator(x,representation='interpolated_grouped',clip=None,memory_budget_bytes=1)

    def test_F2_explicit_real_only_contract_all_entry_points(self):
        x=sparse.csr_matrix([[1.,2.,0.],[0.,5.,1.],[3.,0.,2.],[1.,4.,1.]])
        for rep in ('interpolated','interpolated_grouped','exact_grouped'):
            op=build_operator(x,representation=rep,clip=None).operator
            for factor in (0j,2j):
                v=np.arange(3,dtype=complex)*(1+factor);w=np.arange(4,dtype=complex)*(1+factor)
                V=np.column_stack((v,2*v));W=np.column_stack((w,2*w))
                actions=(lambda:op@v,lambda:op.T@w,lambda:op@V,lambda:op.T@W,
                         lambda:op.matvec(v),lambda:op.rmatvec(w),lambda:op.matmat(V),lambda:op.rmatmat(W),
                         lambda:op.H@w,lambda:op.T@W[:,:1])
                for action in actions:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        with self.assertRaisesRegex(TypeError,'real numeric operands only'):action()


if __name__=='__main__':unittest.main()
