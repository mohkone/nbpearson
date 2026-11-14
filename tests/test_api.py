"""Standalone tests: these require only the installed package, NumPy and SciPy."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from nbpearson import build_operator,FixedMargins,counts_fingerprint,SelectionError,ResourceLimitError
import nbpearson._api as api

REPRESENTATIONS=('interpolated','interpolated_grouped','exact_grouped')


def dense(x,theta,c):
    a=x.toarray();d=a.sum(1);p=a.sum(0)/a.sum();mu=d[:,None]*p
    r=(a-mu)/np.sqrt(mu+mu*mu/theta)
    if c is not None:r=np.clip(r,-c,c)
    return r-r.mean(0)


class ApiTests(unittest.TestCase):
    def test_products_centers_and_immutability(self):
        cases=[[[2,0,0],[0,2,0],[0,0,2],[1,1,0]],
               np.tile([[2,0,1],[1,4,1]],(8,1)),
               np.vstack([np.tile([2,0,1],(31,1)),[0,10,0],[0,5,5],[20,0,0]]),
               np.column_stack([np.arange(1,151),np.full(150,2),np.full(150,3)]),np.eye(15)*100000+1]
        rng=np.random.default_rng(9900)
        for values in cases:
            x=sparse.csr_matrix(values,dtype=float);before=[a.copy() for a in (x.data,x.indices,x.indptr)]
            for theta in (1.,10.,100.):
                for clip in (np.sqrt(theta),None):
                    truth=dense(x,theta,clip);V=rng.normal(size=(x.shape[1],8));W=rng.normal(size=(x.shape[0],8))
                    for rep in REPRESENTATIONS:
                        result=build_operator(x,representation=rep,theta=theta,clip=clip,**({} if rep=='exact_grouped' else {'matrix_rtol':1e-12}));op=result.operator
                        norm=np.linalg.norm(truth);error=np.linalg.norm(op@np.eye(x.shape[1])-truth)
                        if norm:self.assertLess(error/norm,1e-12)
                        else:self.assertEqual(error,0.)
                        for a,b in ((op@V,truth@V),(op.T@W,truth.T@W),(op@V[:,0],truth@V[:,0]),(op.T@W[:,0],truth.T@W[:,0])):
                            np.testing.assert_allclose(a,b,rtol=1e-12,atol=2e-10)
                        np.testing.assert_allclose(W.T@(op@V),(op.T@W).T@V,atol=2e-10,rtol=1e-12)
                        np.testing.assert_allclose(op.T@np.ones(x.shape[0]),0,atol=1e-11)
                        self.assertEqual(op.dtype,np.dtype('float64'))
                        self.assertEqual(result.diagnostics['floating_validation'],'NOT_EVALUATED')
                        self.assertEqual(result.diagnostics['constructed_representation'],rep)
                        self.assertFalse(result.diagnostics['floating_point_certificate'])
                        if rep=='exact_grouped':self.assertIsNone(result.diagnostics['analytic_selection']['interpolation_nodes'])
            for a,b in zip(before,(x.data,x.indices,x.indptr)):np.testing.assert_array_equal(a,b)

    def test_canonical_csr_duplicates_unsorted_and_fixed_margins(self):
        x=sparse.csr_matrix((np.array([1.,2.,1.,3.,2.,1.]),np.array([1,0,1,0,1,0]),np.array([0,3,6])),shape=(2,2))
        buffers=[a.copy() for a in (x.data,x.indices,x.indptr)]
        canonical=x.copy();canonical.sum_duplicates();canonical.sort_indices()
        d=np.asarray(canonical.sum(1)).ravel();p=np.asarray(canonical.sum(0)).ravel()/canonical.sum()
        fixed=FixedMargins(d,p,counts_fingerprint(x),'unit fixture fixed binary64 margins')
        for rep in REPRESENTATIONS:
            a=build_operator(x,representation=rep,clip=10.,margins=fixed)
            b=build_operator(canonical,representation=rep,clip=10.)
            np.testing.assert_allclose(a.operator@np.eye(2),b.operator@np.eye(2),atol=1e-12)
            self.assertEqual(a.diagnostics['margins']['contract'],'fixed_binary64_margins')
            self.assertEqual(b.diagnostics['margins']['contract'],'derived_binary64_count_margins')
        for a,b in zip(buffers,(x.data,x.indices,x.indptr)):np.testing.assert_array_equal(a,b)
        result=build_operator(sparse.csr_array(canonical),representation='exact_grouped',clip=None)
        self.assertEqual(result.operator.shape,(2,2))
        wrong=FixedMargins(d,p,'wrong','source')
        with self.assertRaises(ValueError):build_operator(x,representation='exact_grouped',clip=10.,margins=wrong)
        # The fixed contract is not silently replaced by recalculated depths.
        shifted=FixedMargins(np.array([10.,11.]),p,counts_fingerprint(x),'explicit fixed test inputs')
        a=build_operator(x,representation='exact_grouped',clip=10.,margins=shifted)
        self.assertEqual(a.diagnostics['margins']['source'],'explicit fixed test inputs')
        mu=np.array([10.,11.])[:,None]*p
        reference=(canonical.toarray()-mu)/np.sqrt(mu+mu*mu/100.)
        reference-=reference.mean(0)
        np.testing.assert_allclose(a.operator@np.eye(2),reference,atol=1e-12)

    def test_invalid_inputs_and_explicit_choices(self):
        x=sparse.csr_matrix([[1.,2.],[2.,1.]])
        with self.assertRaises(TypeError):build_operator(x)
        for rep in ('auto','other',None):
            with self.assertRaises(ValueError):build_operator(x,representation=rep)
        for theta in (0.,-1.,np.inf,np.nan,[100.],True):
            with self.assertRaises(ValueError):build_operator(x,representation='exact_grouped',theta=theta,clip=None)
        for clip in (1.,np.inf,np.nan,0.,-1.,True,'auto'):
            with self.assertRaises(ValueError):build_operator(x,representation='exact_grouped',clip=clip)
        for bad in (sparse.csr_matrix([[-1.,2.],[2.,1.]]),sparse.csr_matrix([[np.nan,2.],[2.,1.]]),
                    sparse.csr_matrix([[.5,2.],[2.,1.]]),sparse.csr_matrix([[1.,0.],[2.,0.]]),
                    sparse.csr_matrix([[0.,0.],[2.,1.]]),sparse.csc_matrix(x),x.toarray(),x.astype(complex)):
            with self.assertRaises(ValueError):build_operator(bad,representation='exact_grouped',clip=None)
        for kw in ({'matrix_rtol':1e-6},{'max_nodes':128}):
            with self.assertRaises(ValueError):build_operator(x,representation='exact_grouped',clip=None,**kw)
        for budget in (0,-1,True,512*2**20+1):
            with self.assertRaises(ValueError):build_operator(x,representation='exact_grouped',clip=None,memory_budget_bytes=budget)
        # Unsupported clipping really affects zeros in this independent example.
        truth=sparse.eye(2,format='csr')*100;mu=50.
        self.assertLess(-np.sqrt(mu/(1+mu/100.)),-1.)
        with self.assertRaises(ValueError):build_operator(truth,representation='exact_grouped',clip=1.)

    def test_selection_failures_resources_and_no_large_materializations(self):
        x=sparse.csr_matrix([[1.,2.],[2.,1.],[5.,1.],[1.,8.]])
        with self.assertRaises(SelectionError) as caught:build_operator(x,representation='interpolated',clip=10.,max_nodes=1)
        self.assertEqual(caught.exception.diagnostics['analytic_selection']['status'],'node_limit')
        with self.assertRaises(SelectionError) as caught:build_operator(x,representation='interpolated',clip=10.,n_norm_pairs=0)
        self.assertEqual(caught.exception.diagnostics['analytic_selection']['status'],'insufficient_norm_bound')
        # A deterministic estimate test: enough for inputs, insufficient for the table.
        y=sparse.csr_matrix((np.arange(1,201,dtype=float),(np.arange(200),np.arange(200)%80)),shape=(200,80))
        with patch.object(api,'process_baseline_bytes',return_value=0),patch.object(api,'_ExactGrouped') as ctor:
            with self.assertRaises(ResourceLimitError):build_operator(y,representation='exact_grouped',memory_budget_bytes=64*2**20+100000)
            ctor.assert_not_called()
        with self.assertRaises(ResourceLimitError):build_operator(x,representation='exact_grouped',clip=10.,memory_budget_bytes=1)
        original=sparse.csr_matrix.toarray
        def bounded(a,*args,**kwargs):
            if a.shape[0]>32 or a.shape[1]>128:raise AssertionError('unbounded dense conversion')
            return original(a,*args,**kwargs)
        # Only lower-bound rows may densify; no whole residual or left factor path.
        with patch.object(sparse.csr_matrix,'toarray',bounded),patch('nbpearson._operators.anchored_factors',side_effect=AssertionError('N*r path forbidden')):
            b=build_operator(y,representation='interpolated_grouped')
            self.assertEqual(b.operator._left.shape[0],200)
            self.assertFalse(hasattr(b.operator,'_table'))
        with patch.object(sparse.csr_matrix,'toarray',side_effect=AssertionError('exact constructor must not densify')):
            build_operator(y,representation='exact_grouped')


if __name__=='__main__':unittest.main()
