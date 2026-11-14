"""Small standalone Decimal oracles; fixed binary64 margins, never archives."""
from decimal import Decimal,localcontext
import unittest
import numpy as np
from scipy import sparse
from nbpearson import build_operator,FixedMargins,counts_fingerprint,SelectionError


def margins(x):
    d=np.asarray(x.sum(1),dtype=float).ravel();cols=np.asarray(x.sum(0),dtype=float).ravel();p=cols/cols.sum()
    return d,p,FixedMargins(d,p,counts_fingerprint(x),'autonomous synthetic fixed-binary64 fixture')


def decnorm(values):
    return sum((value*value for row in values for value in row),Decimal(0)).sqrt()


def oracle(x,d,p,theta,c,precision):
    if x.shape[0]*x.shape[1]>100:raise ValueError('tiny oracle only')
    values=x.toarray()
    with localcontext() as context:
        context.prec=precision;th=Decimal.from_float(float(theta));clip=Decimal.from_float(float(c))
        depths=[Decimal.from_float(float(value)) for value in d]
        proportions=[Decimal.from_float(float(value)) for value in p]
        raw=[]
        for i in range(x.shape[0]):
            row=[]
            for j in range(x.shape[1]):
                mu=depths[i]*proportions[j]
                r=(Decimal(int(values[i,j]))-mu)/(mu+mu*mu/th).sqrt()
                row.append(min(clip,max(-clip,r)))
            raw.append(row)
        # Contrast centering avoids discarding small differences in the oracle.
        contrast=[[raw[i][j]-raw[0][j] for j in range(x.shape[1])] for i in range(x.shape[0])]
        mean=[sum((row[j] for row in contrast),Decimal(0))/x.shape[0] for j in range(x.shape[1])]
        centered=[[value-mean[j] for j,value in enumerate(row)] for row in contrast]
        return centered,decnorm(centered)


def error(actual,reference):
    with localcontext() as context:
        context.prec=160
        diff=[[Decimal.from_float(float(actual[i,j]))-reference[i][j] for j in range(actual.shape[1])] for i in range(actual.shape[0])]
        absolute=decnorm(diff);norm=decnorm(reference)
        return float(absolute),float(norm),None if norm==0 else float(absolute/norm)


class AutonomousStabilityTests(unittest.TestCase):
    def test_nearly_null_80_160_fixed_margins(self):
        b=10**12
        fixtures=[[[b+1,b-1],[b-1,b+1]],
                  [[b+1,b],[b,b+1],[2*b+1,2*b-1],[2*b-1,2*b+1]]]
        for values in fixtures:
            x=sparse.csr_matrix(values,dtype=float);d,p,fixed=margins(x)
            for theta in (1.,100.):
                c=max(np.sqrt(theta),np.sqrt(x.shape[0]))
                ref80,_=oracle(x,d,p,theta,c,80);ref160,norm=oracle(x,d,p,theta,c,160)
                with localcontext() as context:
                    context.prec=160
                    agreement=decnorm([[a-b for a,b in zip(ar,br)] for ar,br in zip(ref80,ref160)])
                self.assertLess(float(agreement),1e-60)
                self.assertGreater(norm,0);self.assertLess(float(norm),1e-8)
                for rep in ('interpolated','interpolated_grouped','exact_grouped'):
                    result=build_operator(x,representation=rep,theta=theta,clip=c,margins=fixed)
                    absolute,reference_norm,relative=error(result.operator@np.eye(x.shape[1]),ref160)
                    self.assertLess(relative,1e-6)
                    self.assertGreater(reference_norm,0)
                    self.assertEqual(result.diagnostics['floating_validation'],'NOT_EVALUATED')

    def test_L_zero_nonzero_target_and_exact_zero(self):
        cases=[(sparse.csr_matrix([[1.,3.],[1.,3.],[4.,2.],[4.,2.]]),False),
               (sparse.csr_matrix([[1.,1.],[2.,2.],[3.,3.],[4.,4.]]),True)]
        for x,is_zero in cases:
            d,p,fixed=margins(x);ref80,_=oracle(x,d,p,100.,10.,80);ref,norm=oracle(x,d,p,100.,10.,160)
            self.assertEqual(norm==0,is_zero)
            if is_zero:self.assertTrue(np.all(x.toarray()[:,0]==x.toarray()[:,1])) # p=1/2 and mu=X exactly.
            for rep in ('interpolated','interpolated_grouped'):
                with self.assertRaises(SelectionError) as caught:build_operator(x,representation=rep,theta=100.,clip=10.,margins=fixed)
                self.assertEqual(caught.exception.diagnostics['analytic_selection']['status'],'insufficient_norm_bound')
                self.assertEqual(caught.exception.diagnostics['analytic_selection']['L'],0.)
            result=build_operator(x,representation='exact_grouped',theta=100.,clip=10.,margins=fixed)
            absolute,reference_norm,relative=error(result.operator@np.eye(2),ref)
            if is_zero:
                self.assertEqual(reference_norm,0.);self.assertIsNone(relative);self.assertLess(absolute,2e-10)
            else:self.assertLess(relative,1e-12)


if __name__=='__main__':unittest.main()
