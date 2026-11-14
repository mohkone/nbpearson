"""Optional archive comparisons only; never shipped as a runtime dependency."""
import os,sys,unittest,importlib.util,json
from pathlib import Path
from decimal import localcontext
import numpy as np
from scipy import sparse
from nbpearson import build_operator,FixedMargins,counts_fingerprint,SelectionError
from nbpearson._operators import _Interpolated,_InterpolatedGrouped,_ExactGrouped

HISTORY=Path(os.environ.get('NBPEARSON_HISTORY_ROOT',Path(__file__).resolve().parents[2]/'nb_pca_dev'))
AVAILABLE=(HISTORY/'v9/env9.py').is_file()
if AVAILABLE:
    sys.path.insert(0,str(HISTORY/'v9'))
    import env9
    import compressed
    from reference import oracle,errors,CONTRACTS,norm
    core=env9.core


def margins(x):
    d=np.asarray(x.sum(1)).ravel();p=np.asarray(x.sum(0)).ravel();p=p/p.sum()
    core.bind_margins(x,d,p)
    return d,p,FixedMargins(d,p,counts_fingerprint(x),'historical fixture frozen binary64')


@unittest.skipUnless(AVAILABLE,'historical sources explicitly optional; standalone API tests remain runnable')
class HistoricalFidelity(unittest.TestCase):
    def test_unclipped_and_historical_selection_limits(self):
        x=sparse.csr_matrix([[1.,2.,0],[5.,1.,3],[0.,4.,2],[10.,0.,1]])
        d,p,fixed=margins(x)
        oldA,selection=core.v5.build_adaptive(x,100.,1e100,generic=True)
        oldB,_=compressed.build_adaptive(x,d,p,100.,1e100)
        oldC=env9.ExactGrouped(x,d,p,100.,1e100)
        for rep,old in zip(('interpolated','interpolated_grouped','exact_grouped'),(oldA,oldB,oldC)):
            result=build_operator(x,representation=rep,clip=None,margins=fixed)
            np.testing.assert_allclose(result.operator@np.eye(3),old@np.eye(3),atol=2e-10,rtol=1e-12)
        for name in ('zero_L_nonzero_target','very_unequal'):
            x=sparse.load_npz(HISTORY/'v5/artifacts/fixtures'/(name+'.npz'));d,p,fixed=margins(x)
            _,selection=core.v5.build_adaptive(x,100.,max(10.,np.sqrt(x.shape[0])),generic=True)
            if name=='zero_L_nonzero_target':self.assertEqual(selection['status'],'insufficient_norm_bound')
            for rep in ('interpolated','interpolated_grouped'):
                if selection['status'] not in ('selected','exact_equal_depths'):
                    with self.assertRaises(SelectionError) as caught:build_operator(x,representation=rep,clip=max(10.,np.sqrt(x.shape[0])),margins=fixed)
                    self.assertEqual(caught.exception.diagnostics['analytic_selection']['status'],selection['status'])
                else:self.assertEqual(build_operator(x,representation=rep,clip=max(10.,np.sqrt(x.shape[0])),margins=fixed).diagnostics['analytic_selection']['interpolation_nodes'],selection['interpolation_nodes'])

    def test_adaptive_historical_coefficients_and_products(self):
        cases=[sparse.csr_matrix([[2.,0,1],[1,4,1],[2,0,1],[1,4,1]]),
            sparse.csr_matrix(np.vstack([np.tile([2.,0,1],(31,1)),[0,10,0],[0,5,5],[20,0,0]])),
            sparse.csr_matrix(np.column_stack([np.arange(1,151),np.full(150,2),np.full(150,3)])),
            sparse.csr_matrix(np.eye(15)*100000+1)]
        rng=np.random.default_rng(9900)
        for x in cases:
            d,p,fixed=margins(x)
            for theta in (1.,10.,100.):
                c=np.sqrt(theta)
                oldA,sel=core.v5.build_adaptive(x,theta,c,generic=True)
                oldB,bsel=compressed.build_adaptive(x,d,p,theta,c)
                oldC=env9.ExactGrouped(x,d,p,theta,c)
                for rep,old in zip(('interpolated','interpolated_grouped','exact_grouped'),(oldA,oldB,oldC)):
                    result=build_operator(x,representation=rep,theta=theta,clip=c,margins=fixed);op=result.operator
                    if rep!='exact_grouped':
                        for key in ('interpolation_nodes','polynomial_degree','L','Delta_r','pairs'):
                            self.assertEqual(result.diagnostics['analytic_selection'].get(key),sel.get(key))
                        np.testing.assert_array_equal(op._right,old.v)
                        np.testing.assert_allclose(op._left,old.u if rep=='interpolated' else old.W,atol=1e-15,rtol=1e-13)
                    else:np.testing.assert_array_equal(op._table,old.table)
                    np.testing.assert_array_equal(op._correction.data,old.correction.data)
                    np.testing.assert_array_equal(op._mean,old.mean)
                    V=rng.normal(size=(x.shape[1],8));W=rng.normal(size=(x.shape[0],8))
                    for actual,expected in ((op@V,old@V),(op.T@W,old.T@W),(op@V[:,0],old@V[:,0]),(op.T@W[:,0],old.T@W[:,0])):
                        np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=1e-12)

    def test_decimal_nearly_null_and_equal_depths(self):
        for name in ('nearly_zero_centered','near_depth_adjacent','nearly_equal_depth','exact_zero_proven','equal_depth_nonzero'):
            x=sparse.load_npz(HISTORY/'v5/artifacts/fixtures'/(name+'.npz'));d,p,fixed=margins(x)
            for theta in (1.,10.,100.):
                c=max(np.sqrt(theta),np.sqrt(x.shape[0]));a=oracle(x,theta,c,CONTRACTS[1],80);b=oracle(x,theta,c,CONTRACTS[1],160)
                with localcontext() as ctx:
                    ctx.prec=160;agreement=norm([[aa-bb for aa,bb in zip(ar,br)] for ar,br in zip(a['A'],b['A'])])
                self.assertLess(float(agreement),1e-60)
                # Fixed node constructors isolate extraction even when L is unusable.
                grouping=np.unique(d,return_inverse=True,return_counts=True)
                operators=[_Interpolated(x,d,p,theta,c,32),_InterpolatedGrouped(x,d,p,theta,c,32,grouping),_ExactGrouped(x,d,p,theta,c,grouping)]
                for op in operators:
                    self.assertEqual(op._correction.dtype,np.dtype('float64'))
                    error=errors(op@np.eye(x.shape[1]),b['A'],b['zero_proven'])
                    if b['zero_proven']:self.assertIsNone(error['relative_error']);self.assertEqual(error['absolute_error'],0.)
                    else:self.assertLess(error['relative_error'],1e-6)
                old,sel=core.v5.build_adaptive(x,theta,c,generic=True)
                for rep in ('interpolated','interpolated_grouped'):
                    if sel['status'] in ('selected','exact_equal_depths'):
                        result=build_operator(x,representation=rep,theta=theta,clip=c,margins=fixed)
                        self.assertEqual(result.diagnostics['analytic_selection']['interpolation_nodes'],sel['interpolation_nodes'])
                    else:
                        with self.assertRaises(SelectionError) as caught:build_operator(x,representation=rep,theta=theta,clip=c,margins=fixed)
                        self.assertEqual(caught.exception.diagnostics['analytic_selection']['status'],sel['status'])

    def test_permutation_and_group_weighting(self):
        rng=np.random.default_rng(9902);x=sparse.csr_matrix(np.vstack([np.tile([2.,0,1],(19,1)),[0,10,0],[20,1,0]]))
        d,p,fixed=margins(x);perm=rng.permutation(x.shape[0]);y=x[perm];dy,py,fy=margins(y)
        for rep in ('interpolated','interpolated_grouped','exact_grouped'):
            a=build_operator(x,representation=rep,clip=10.,margins=fixed).operator
            b=build_operator(y,representation=rep,clip=10.,margins=fy).operator
            expected=a@np.eye(x.shape[1]);np.testing.assert_allclose(b@np.eye(x.shape[1]),expected[perm],atol=1e-12)
            if rep!='interpolated':
                gp=rng.permutation(len(a._depth_values));inverse=np.argsort(gp)
                field='_left' if rep=='interpolated_grouped' else '_table'
                setattr(a,field,getattr(a,field)[gp].copy());a._groups=inverse[a._groups]
                a._depth_values=a._depth_values[gp];a._multiplicity=a._multiplicity[gp]
                np.testing.assert_allclose(a@np.eye(3),expected,atol=1e-12)
                w=rng.normal(size=x.shape[0]);np.testing.assert_allclose(a.T@w,expected.T@w,atol=1e-12)


if __name__=='__main__':unittest.main()
