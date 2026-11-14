"""Three manual representations of the same stabilized column-centered NB target."""
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator
from ._numerics import stable_residual,background_delta,anchored_factors,grouped_factors


def correction(x,d,p,theta,c):
    """v3 support correction plus v5 full-column stabilization, on a private CSR."""
    work=x.astype(np.float64,copy=True);positive=negative=0;n0=float(d.min())
    for first in range(0,x.shape[0],128):
        last=min(first+128,x.shape[0]);start,stop=work.indptr[first],work.indptr[last]
        rr=np.repeat(np.arange(first,last),np.diff(work.indptr[first:last+1]))
        mu=d[rr]*p[work.indices[start:stop]];sd=np.sqrt(mu+mu**2/theta)
        exact_r=(work.data[start:stop]-mu)/sd
        delta=np.clip(exact_r,-c,c)-exact_r
        positive+=int(np.count_nonzero(exact_r>c));negative+=int(np.count_nonzero(exact_r<-c))
        work.data[start:stop]=work.data[start:stop]/sd+delta
    occurrences=np.bincount(work.indices,minlength=x.shape[1]);full=occurrences==x.shape[0]
    if full.any():
        for a in range(0,x.shape[0],128):
            b=min(a+128,x.shape[0]);lo,hi=x.indptr[a],x.indptr[b]
            rows=np.repeat(np.arange(a,b),np.diff(x.indptr[a:b+1]));cols=x.indices[lo:hi];mask=full[cols]
            if not mask.any():continue
            rr,cc=rows[mask],cols[mask]
            residual=stable_residual(x.data[lo:hi][mask],d[rr],p[cc],theta)
            delta=(d[rr]-n0)/n0;bg=background_delta(delta,n0,p[cc],theta)
            work.data[lo:hi][mask]=np.clip(residual,-c,c)-bg
    csc=work.tocsc()
    mean=np.array([math.fsum(float(z) for z in csc.data[csc.indptr[j]:csc.indptr[j+1]])/x.shape[0] for j in range(x.shape[1])])
    clips=dict(clipped_positive=positive,clipped_negative=negative,clipped_entries=positive+negative,clipped_zero_entries=0)
    return work,mean,clips,int(full.sum())


class _NBOperator(LinearOperator):
    def __init__(self,x,d,p,theta,c):
        self._correction,self._mean,self._clips,self._full_columns=correction(x,d,p,theta,c)
        self._n0=float(d.min())
        super().__init__(dtype=np.dtype(np.float64),shape=x.shape)

    def _matvec(self,v):return self._matmat(np.asarray(v).reshape(-1,1)).ravel()
    def _rmatvec(self,w):return self._rmatmat(np.asarray(w).reshape(-1,1)).ravel()

    @staticmethod
    def _real_operand(values):
        values=np.asarray(values)
        if np.iscomplexobj(values) or values.dtype.kind not in 'biuf':
            raise TypeError('nbpearson supports real numeric operands only; complex operands are rejected')
        return values

    def storage_detail(self):
        result=dict(C=sum(a.nbytes for a in (self._correction.data,self._correction.indices,self._correction.indptr)),C_mean=self._mean.nbytes)
        for field in ('left','right','groups','multiplicity','depth_values','basis_mean','table','background_mean'):
            a=getattr(self,'_'+field,None)
            if a is not None:result[field]=a.nbytes
        return result


class _Interpolated(_NBOperator):
    def __init__(self,x,d,p,theta,c,nodes):
        super().__init__(x,d,p,theta,c)
        self._left,self._right=anchored_factors(d,p,theta,nodes)

    def _matmat(self,v):
        v=self._real_operand(v)
        return (self._correction@v-(self._mean@v)[None,:])+self._left@(self._right.T@v)

    def _rmatmat(self,w):
        w=self._real_operand(w)
        return (self._correction.T@w-self._mean[:,None]*w.sum(0)[None,:])+self._right@(self._left.T@w)


class _GroupedBase(_NBOperator):
    def _aggregate(self,w):
        if w.shape[1]==1:return np.bincount(self._groups,weights=w[:,0],minlength=len(self._depth_values))[:,None]
        grouped=np.zeros((len(self._depth_values),w.shape[1]));np.add.at(grouped,self._groups,w)
        return grouped


class _InterpolatedGrouped(_GroupedBase):
    def __init__(self,x,d,p,theta,c,nodes,grouping):
        super().__init__(x,d,p,theta,c)
        self._depth_values,self._groups,self._multiplicity=grouping
        self._left,self._right,self._basis_mean=grouped_factors(self._depth_values,self._multiplicity,p,theta,nodes)

    def _matmat(self,v):
        v=self._real_operand(v)
        grouped=self._left@(self._right.T@v)
        return (self._correction@v-(self._mean@v)[None,:])+grouped[self._groups]

    def _rmatmat(self,w):
        w=self._real_operand(w)
        return (self._correction.T@w-self._mean[:,None]*w.sum(0)[None,:])+self._right@(self._left.T@self._aggregate(w))


class _ExactGrouped(_GroupedBase):
    def __init__(self,x,d,p,theta,c,grouping):
        super().__init__(x,d,p,theta,c)
        self._depth_values,self._groups,self._multiplicity=grouping
        U=len(self._depth_values);self._table=np.empty((U,x.shape[1]),dtype=float)
        total=np.zeros(x.shape[1]);compensation=np.zeros_like(total)
        for a in range(0,U,128):
            b=min(a+128,U);delta=(self._depth_values[a:b,None]-self._n0)/self._n0
            pp=np.broadcast_to(p,(b-a,len(p)));self._table[a:b]=background_delta(delta,self._n0,pp,theta)
            term=np.sum(self._table[a:b]*self._multiplicity[a:b,None],axis=0)/x.shape[0]
            y=term-compensation;updated=total+y;compensation=(updated-total)-y;total=updated
        self._background_mean=total;self._table-=total

    def _matmat(self,v):
        v=self._real_operand(v)
        background=self._table@v
        return (self._correction@v-(self._mean@v)[None,:])+background[self._groups]

    def _rmatmat(self,w):
        w=self._real_operand(w)
        return (self._correction.T@w-self._mean[:,None]*w.sum(0)[None,:])+self._table.T@self._aggregate(w)
