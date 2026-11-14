"""Classical historical NB identities and node selector, with explicit validated inputs."""
import math,time
from numbers import Real
import numpy as np
from scipy.interpolate import BarycentricInterpolator
MAX_PROCESS_BYTES=512*2**20
SOLVER_RESERVE_BYTES=64*2**20
BOUND_SCOPE='borne analytique en arithmétique exacte'

def two_product(a,b):
    """Dekker error-free product where split/products do not overflow or underflow."""
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    splitter=134217729.  # 2**27+1, binary64 Dekker splitting constant, not a tolerance.
    if np.any(abs(a)>np.finfo(float).max/splitter) or np.any(abs(b)>np.finfo(float).max/splitter):
        raise FloatingPointError('compensated product outside supported splitting range')
    ca=splitter*a;ah=ca-(ca-a);al=a-ah
    cb=splitter*b;bh=cb-(cb-b);bl=b-bh
    product=a*b
    if np.any(abs(product)<np.finfo(float).tiny*2**54):
        raise FloatingPointError('compensated product outside normal EFT range')
    error=((ah*bh-product)+ah*bl+al*bh)+al*bl
    return product,error


def stable_residual(values,depths,proportions,theta):
    mu,low=two_product(depths,proportions)
    if np.any(mu<=0) or np.any(~np.isfinite(mu)):
        raise FloatingPointError('nonpositive/nonfinite product in compensated residual')
    # TwoSum(values, -mu), followed by subtraction of the exact product tail.
    high=values-mu;back=high-values
    subtraction_error=(values-(high-back))+(-mu-back)
    numerator=high+(subtraction_error-low)
    return numerator/np.sqrt(mu*(1+mu/theta))


def background_delta(delta,n0,p,theta):
    """B(n0*(1+delta),p)-B(n0,p), rationalized; no subtraction of close B values."""
    mu0=n0*p
    mu=mu0*(1+delta)
    if np.any(mu0<=0) or np.any(~np.isfinite(mu)):
        raise FloatingPointError('background difference outside supported numerical range')
    scale=np.maximum(theta,mu)
    fraction=(theta/scale)/(theta/scale+mu/scale)
    z=delta*fraction
    # Branches avoid forming theta/mu0 when that ratio would overflow.
    b0=np.empty_like(mu0)
    small=mu0<=theta
    b0[small]=-np.sqrt(mu0[small])/np.sqrt(1+mu0[small]/theta)
    b0[~small]=-np.sqrt(theta)/np.sqrt(1+theta/mu0[~small])
    return b0*z/(np.sqrt(1+z)+1)


def column_fsum(a):
    return np.array([math.fsum(float(z) for z in a[:,j])/a.shape[0] for j in range(a.shape[1])])


def anchored_factors(d,p,theta,r):
    n0=float(d.min());span=math.log1p(float((d.max()-n0)/n0))
    if span==0:return np.zeros((len(d),1)),np.zeros((len(p),1))
    z=np.cos(np.pi*np.arange(r)/(r-1));t=2*np.log1p((d-n0)/n0)/span-1
    w=(-1.)**np.arange(r);w[[0,-1]]*=.5
    u=BarycentricInterpolator(z,np.eye(r),wi=w)(t)
    delta=np.expm1(.5*span*(z+1))[:,None]
    pp=np.broadcast_to(p,(r,len(p)))
    v=background_delta(delta,n0,pp,theta).T
    # Translate before summation, and center basis independently of sparse C.
    u=u-u[int(np.argmin(d))]
    u-=column_fsum(u)
    return u,v


def grouped_factors(depth_values, multiplicity, p, theta, nodes):
    """Build U*r directly. Neither N*r nor U*G is evaluated here."""
    n0=float(depth_values[0]);span=math.log1p(float((depth_values[-1]-n0)/n0))
    if span==0:
        return np.zeros((len(depth_values),1)), np.zeros((len(p),1)), np.zeros(1)
    z=np.cos(np.pi*np.arange(nodes)/(nodes-1))
    weights=(-1.)**np.arange(nodes);weights[[0,-1]]*=.5
    t=2*np.log1p((depth_values-n0)/n0)/span-1
    W=BarycentricInterpolator(z,np.eye(nodes),wi=weights)(t)
    # Same stable translation as v5, then multiplicity-weighted centering.
    W-=W[0].copy()
    N=int(multiplicity.sum())
    basis_mean=np.array([math.fsum(float(w)*int(m) for w,m in zip(W[:,j],multiplicity))/N
                         for j in range(nodes)])
    W-=basis_mean
    delta=np.expm1(.5*span*(z+1))[:,None]
    pp=np.broadcast_to(p,(nodes,len(p)))
    V=background_delta(delta,n0,pp,theta).T
    return W,V,basis_mean


def exp_value(log_value):
    if log_value is None: return None
    if log_value == -math.inf: return 0.
    if log_value > math.log(np.finfo(float).max): return None
    return math.exp(log_value)  # Underflow is documented; decisions never use this value.


def log_squared_norm(a):
    scale = float(np.max(np.abs(a), initial=0.))
    if scale == 0: return -math.inf
    return 2*math.log(scale)+math.log(float(np.sum((a/scale)**2)))


def lower_bound(x, d, p, theta, c, n_pairs):
    """At most 32 exact rows, blocked by genes; no whole N x G residual scan."""
    q = min(n_pairs, x.shape[0]//2)
    pairs = [(2*i,2*i+1) for i in range(q)]
    if q == 0: return -math.inf, pairs
    indices = np.arange(2*q)
    total = -math.inf
    for first in range(0, x.shape[1], 128):
        last = min(first+128, x.shape[1])
        mu = d[indices,None]*p[None,first:last]
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            rr = (x[indices,first:last].toarray()-mu)/np.sqrt(mu+mu**2/theta)
        rr = np.clip(rr,-c,c)
        total = float(np.logaddexp(total,log_squared_norm(rr[::2]-rr[1::2])))
    return .5*(total-math.log(2)), pairs


def peak_estimate(shape, nnz, nodes, baseline_bytes):
    n,g=map(int,shape);r=int(nodes)
    csr = nnz*12+(n+1)*4
    # Private sparse copies, interpolation bases/temporaries, generic r*r basis,
    # blocked residual rows and 64 MiB SVD reserve. Baseline already includes input.
    added = 2*csr+8*(6*(n+g)*r+3*r*r+12*g+6*n+8*32*min(g,128))+SOLVER_RESERVE_BYTES
    return int(baseline_bytes+added)


def compressed_peak_estimate(shape, csr_bytes, unique_depths, nodes, baseline_bytes):
    """Admission for U*r construction, including copies and transient arrays.

    Four U*r bases/evaluation arrays, sixteen G*r slots for the right-factor
    rationalized background (including masked work), three r*r SciPy basis
    slots, grouping/row buffers, historical k=21 workspace, 256-row blocks,
    and the same 64 MiB solver reserve. This is not a physical RSS guarantee.
    """
    n,g=map(int,shape);u=int(unique_depths);r=int(nodes)
    elements=4*u*r+16*g*r+3*r*r+4*n+4*u+3*n*21+6*256*g
    return int(baseline_bytes+3*csr_bytes+8*elements+SOLVER_RESERVE_BYTES)


def select_nodes(x, d, p, theta, c, rtol=1e-6, max_nodes=128,
                 max_memory_bytes=MAX_PROCESS_BYTES, n_norm_pairs=16, baseline_bytes=0,
                 peak_estimator=None):
    """Historical mathematical criterion with an injectable admission cost only."""
    start=time.perf_counter()
    estimate_peak=peak_estimate if peak_estimator is None else peak_estimator
    if not isinstance(rtol,Real) or not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('0 < finite rtol < 1 required; separate from SVD tolerance')
    for name,value,minimum,maximum in [('max_nodes',max_nodes,1,1000000),
             ('max_memory_bytes',max_memory_bytes,1,MAX_PROCESS_BYTES),('n_norm_pairs',n_norm_pairs,0,16)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not minimum <= value <= maximum:
            raise ValueError(f'invalid {name}')
    out=dict(status=None,interpolation_nodes=None,polynomial_degree=None,rtol=float(rtol),
             max_nodes=int(max_nodes),max_memory_bytes=int(max_memory_bytes),n_norm_pairs_requested=int(n_norm_pairs),
             bound_scope=BOUND_SCOPE,floating_point_certificate=False,pair_rule='first disjoint adjacent pairs (0,1),(2,3),...',
             pairs=[],L=None,log_L=None,Delta_r=None,log_Delta_r=None,relative_upper_bound=None,log_relative_upper_bound=None)
    def done(status):
        out['status']=status;out['selection_seconds']=time.perf_counter()-start;return out
    baseline=baseline_bytes;out['baseline_live_bytes']=baseline
    if baseline > max_memory_bytes:return done('memory_limit')
    try:
        baseline=baseline_bytes
        minimum=estimate_peak(x.shape,x.nnz,1,baseline)
        out['estimated_peak_bytes']=minimum
        if minimum > max_memory_bytes:return done('memory_limit')
        t=time.perf_counter()
        log_l,pairs=lower_bound(x,d,p,theta,c,n_norm_pairs)
        out.update(norm_rows_seconds=time.perf_counter()-t,pairs=pairs,n_norm_pairs=len(pairs),
                   log_L=log_l,L=exp_value(log_l),norm_rows_evaluated=2*len(pairs))
    except FloatingPointError as e:
        out['reason']=str(e);return done('numerical_range_limit')
    span=math.log1p(float((d.max()-d.min())/d.min()))
    out['log_depth_span']=span
    if span == 0:
        out.update(interpolation_nodes=1,polynomial_degree=0,Delta_r=0.,log_Delta_r=-math.inf,
                   relative_upper_bound=0. if log_l != -math.inf else None,
                   log_relative_upper_bound=-math.inf if log_l != -math.inf else None,
                   exact_equal_depths=True)
        return done('exact_equal_depths')
    # Stable even for nearly equal positive depths: no subtraction of close logs.
    log_d=math.log(math.pi)-math.log(span)
    log_rho=math.asinh(math.exp(log_d)) if log_d < 700 else log_d+math.log(2)
    log_rho_minus_1 = (log_rho+math.log1p(-math.exp(-log_rho)) if log_rho > math.log(2)
                       else math.log(math.expm1(log_rho)))
    pref=(math.log(4)+.5*(math.log(theta)+math.log(x.shape[0])+math.log(x.shape[1]))-log_rho_minus_1)
    out.update(log_rho=log_rho,log_prefactor=pref,exact_equal_depths=False)
    if log_l == -math.inf:
        out['reason']='Selected contrasts vanish; this does not prove the centered matrix is zero.'
        return done('insufficient_norm_bound')
    if max_nodes < 2:return done('node_limit')
    for r in range(2,max_nodes+1):
        estimate=estimate_peak(x.shape,x.nnz,r,baseline)
        delta=pref-(r-1)*log_rho
        out.update(last_evaluated_nodes=r,estimated_peak_bytes=estimate,Delta_r=exp_value(delta),
                   log_Delta_r=delta,relative_upper_bound=exp_value(delta-log_l),log_relative_upper_bound=delta-log_l)
        if estimate > max_memory_bytes:return done('memory_limit')
        if delta <= math.log(rtol)+log_l:
            out.update(interpolation_nodes=r,polynomial_degree=r-1)
            return done('selected')
    return done('node_limit')

