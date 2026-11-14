"""Explicit binary64 count and margin contracts; no exclusions or in-place edits."""
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import sys
import numpy as np
from scipy import sparse


def process_baseline_bytes():
    """Portable admission input where available: process high-water mark, not RSS guarantee."""
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform=='darwin' else value*1024)
    except (ImportError,AttributeError):
        return 0


def array_digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def canonical_counts(X):
    if not sparse.issparse(X) or X.format!='csr' or len(X.shape)!=2:
        raise ValueError('X must be a two-dimensional SciPy CSR count matrix or array')
    if min(X.shape)<2:raise ValueError('both dimensions must be >= 2')
    if X.dtype.kind not in 'iuf' or np.iscomplexobj(X.data):raise ValueError('real numeric counts required')
    if not np.isfinite(X.data).all() or (X.data<0).any():raise ValueError('finite nonnegative counts required before canonicalization')
    if X.dtype.kind=='f' and not np.array_equal(X.data,np.rint(X.data)):
        raise ValueError('integer-valued counts required; normalized data are not accepted')
    work=X.copy()
    try:work.check_format(full_check=True)
    except (ValueError,IndexError,TypeError) as error:raise ValueError('invalid CSR structure') from error
    work=sparse.csr_matrix(work,dtype=np.float64,copy=True)
    work.sum_duplicates();work.eliminate_zeros();work.sort_indices()
    if not np.isfinite(work.data).all():raise ValueError('nonfinite canonicalized counts')
    return work


def counts_fingerprint(X):
    """SHA-256 of canonical float64 CSR shape and buffers, for fixed-margin binding."""
    x=canonical_counts(X)
    return array_digest(np.asarray(x.shape,dtype=np.int64),x.data,x.indices.astype(np.int64),x.indptr.astype(np.int64))


def canonical_fingerprint(x):
    return array_digest(np.asarray(x.shape,dtype=np.int64),x.data,x.indices.astype(np.int64),x.indptr.astype(np.int64))


@dataclass(frozen=True)
class FixedMargins:
    """Supplied binary64 inputs, bound to canonical counts and a caller source label.

    Their equality to newly computed margins is deliberately not asserted.
    Array shape/positivity/finiteness and count fingerprint are validated at build.
    """
    depths: object
    proportions: object
    counts_fingerprint: str
    source: str


def margins_for(x,provided,fingerprint):
    if provided is None:
        d=np.asarray(x.sum(1)).ravel()
        totals=np.asarray(x.sum(0)).ravel();total=float(totals.sum())
        if not math.isfinite(total) or total<=0:raise ValueError('finite positive grand total required')
        p=totals/total;contract='derived_binary64_count_margins';source='computed once from canonical full count universe'
    else:
        if not isinstance(provided,FixedMargins):raise ValueError('margins must be FixedMargins or None')
        if not isinstance(provided.source,str) or not provided.source.strip():raise ValueError('fixed margins require a nonempty source label')
        if provided.counts_fingerprint!=fingerprint:raise ValueError('fixed-margin count fingerprint mismatch')
        if np.iscomplexobj(provided.depths) or np.iscomplexobj(provided.proportions):raise ValueError('real fixed margins required')
        try:d=np.array(provided.depths,dtype=np.float64,copy=True);p=np.array(provided.proportions,dtype=np.float64,copy=True)
        except (TypeError,ValueError) as error:raise ValueError('numeric fixed margins required') from error
        contract='fixed_binary64_margins';source=provided.source
    if d.shape!=(x.shape[0],) or p.shape!=(x.shape[1],):raise ValueError('margin dimensions must match the full count universe')
    if not np.isfinite(d).all() or not np.isfinite(p).all() or (d<=0).any() or (p<=0).any() or (p>1).any():
        raise ValueError('finite positive margins required; explicitly remove zero margins outside this API')
    if not math.isfinite(float(d.sum())):raise ValueError('nonfinite sum of depths')
    # This validates probability input, not an operator error or a norm denominator.
    if abs(float(p.sum())-1.)>8*np.finfo(float).eps:raise ValueError('proportions must sum to one within 8 binary64 eps')
    d.flags.writeable=False;p.flags.writeable=False
    return d,p,dict(contract=contract,source=source,sha256=array_digest(d,p),count_fingerprint=fingerprint,
        zero_margin_policy='reject; no rows or genes removed',integer_total_collision_warning=bool(float(d.max())>=2**53))


def resolve_theta_clip(theta,clip,n):
    if isinstance(theta,(bool,np.bool_)) or not isinstance(theta,Real) or not np.isfinite(theta) or theta<=0:
        raise ValueError('theta must be a finite scalar > 0')
    if clip is None:return float(theta),math.inf,'unclipped'
    if isinstance(clip,str):
        if clip!='sqrt_n':raise ValueError('clip must be sqrt_n, None, or a finite supported threshold')
        c=float(np.sqrt(n));contract='sqrt_n'
    elif isinstance(clip,(bool,np.bool_)) or not isinstance(clip,Real) or not np.isfinite(clip) or clip<=0:
        raise ValueError('finite positive scalar clipping threshold required; use None for unclipped')
    else:c=float(clip);contract='explicit_symmetric'
    if c<np.sqrt(theta):raise ValueError('support-only clipping requires c >= sqrt(theta); zero-count corrections cannot be omitted')
    return float(theta),c,contract
