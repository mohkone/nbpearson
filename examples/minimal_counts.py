"""Entirely synthetic example; run only after an explicit local installation."""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from nbpearson import build_operator


def main():
    n,g=180,80;rows=np.repeat(np.arange(n),4)
    cols=np.array([(i+j*13)%g for i in range(n) for j in range(4)])
    values=np.random.default_rng(11).integers(1,30,size=len(rows))
    x=sparse.csr_matrix((values,(rows,cols)),shape=(n,g))
    result=build_operator(x,representation='interpolated_grouped',theta=100.,clip='sqrt_n',matrix_rtol=1e-6,max_nodes=128)
    u,s,vt=svds(result.operator,k=21,ncv=64,tol=1e-10,maxiter=10000,solver='arpack',v0=np.random.default_rng(6619).normal(size=min(x.shape)))
    order=np.argsort(s)[::-1];u,s,vt=u[:,order],s[order],vt[order]
    print(json.dumps(dict(shape=x.shape,nodes=result.diagnostics['analytic_selection']['interpolation_nodes'],
        retained_components=20,computed_triplets=21,gap_20_21=float(s[19]-s[20]),
        matrix_floating_validation=result.diagnostics['floating_validation'],
        total_explained_variance='NOT_ESTIMATED_FROM_RETAINED_COMPONENTS')))


if __name__=='__main__':main()
