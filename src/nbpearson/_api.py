"""Explicit constructor and honest selection/resource diagnostics; no solver."""
from dataclasses import dataclass
import math
import numpy as np
from ._validation import canonical_counts,canonical_fingerprint,margins_for,process_baseline_bytes,resolve_theta_clip
from ._numerics import select_nodes,compressed_peak_estimate
from ._operators import _Interpolated,_InterpolatedGrouped,_ExactGrouped

VERSION='0.1.0.dev1'


class SelectionError(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics


class ResourceLimitError(MemoryError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics


@dataclass(frozen=True)
class BuildResult:
    operator: object
    diagnostics: dict


def build_operator(X,*,representation,theta=100.0,clip='sqrt_n',matrix_rtol=None,max_nodes=None,
                   memory_budget_bytes=512*2**20,margins=None,n_norm_pairs=16):
    """Build one requested representation. Never chooses a representation or falls back.

    See README for fixed-margin binding, scalar NB clipping domain, admission
    estimates, selection exceptions and the distinction from SVD tolerance.
    """
    names=('interpolated','interpolated_grouped','exact_grouped')
    if representation not in names:raise ValueError('representation must explicitly name one of '+', '.join(names))
    if isinstance(memory_budget_bytes,(bool,np.bool_)) or not isinstance(memory_budget_bytes,(int,np.integer)) or not 0<memory_budget_bytes<=512*2**20:
        raise ValueError('memory_budget_bytes must be an integer in (0, 512 MiB]')
    if isinstance(n_norm_pairs,(bool,np.bool_)) or not isinstance(n_norm_pairs,(int,np.integer)) or not 0<=n_norm_pairs<=16:
        raise ValueError('n_norm_pairs must be an integer in [0,16]')
    if representation=='exact_grouped' and (matrix_rtol is not None or max_nodes is not None):
        raise ValueError('matrix_rtol and max_nodes must be None for exact_grouped; no interpolation bound is claimed')
    baseline=process_baseline_bytes()
    info=dict(version=VERSION,requested_representation=representation,constructed_representation=None,
        floating_validation='NOT_EVALUATED',floating_point_certificate=False,L_certified_lower_bound=False,
        operand_contract='real numeric vectors/blocks only; complex operands raise TypeError before arithmetic',
        resources=dict(memory_budget_bytes=int(memory_budget_bytes),baseline_high_water_bytes=baseline,
                       policy='conservative admission estimate, not RSS guarantee; no internal watchdog or solver'),
        numerical_limits='CPU binary64; scalar finite theta; cancellation diagnostics are not formal roundoff bounds')
    # Early admission before private canonicalization copies. Full shape validation follows.
    raw_bytes=sum(getattr(getattr(X,name,None),'nbytes',0) for name in ('data','indices','indptr'))
    if baseline+3*raw_bytes+64*2**20>memory_budget_bytes:
        info['resources']['estimated_peak_bytes']=baseline+3*raw_bytes+64*2**20
        raise ResourceLimitError('input/copy admission exceeds declared budget',info)
    x=canonical_counts(X);theta,c,clip_contract=resolve_theta_clip(theta,clip,x.shape[0])
    fingerprint=canonical_fingerprint(x);d,p,margin_info=margins_for(x,margins,fingerprint)
    info.update(shape=list(x.shape),nnz=int(x.nnz),theta=theta,clipping=dict(contract=clip_contract,threshold=None if math.isinf(c) else c,before_centering=True),margins=margin_info)
    # Conversion warnings are diagnostic only; this API does not claim exact integer margins.
    margin_info['integer_total_collision_warning'] |= bool(float(x.data.max(initial=0))*max(np.diff(x.indptr),default=0)>=2**53)
    grouping=None
    if representation!='interpolated':
        grouping=np.unique(d,return_inverse=True,return_counts=True);U=len(grouping[0]);info['U']=U
    else:U=None;info['U']=None
    csr=sum(a.nbytes for a in (x.data,x.indices,x.indptr));baseline=max(baseline,process_baseline_bytes())
    if representation=='exact_grouped':
        n,g=x.shape
        estimate=int(baseline+3*csr+2*8*U*g+8*(4*n+5*U+4*g+6*128*g+3*n*21)+64*2**20)
        selection=dict(status='NOT_APPLICABLE_NO_INTERPOLATION',interpolation_nodes=None,polynomial_degree=None,
                       L=None,Delta_r=None,relative_upper_bound=None,bound_scope=None)
    else:
        rtol=1e-6 if matrix_rtol is None else matrix_rtol;limit=128 if max_nodes is None else max_nodes
        if isinstance(limit,(bool,np.bool_)) or not isinstance(limit,(int,np.integer)) or not 1<=limit<=128:
            raise ValueError('max_nodes must be an integer in [1,128]')
        estimator=None
        if representation=='interpolated_grouped':
            def estimator(shape,nnz,nodes,baseline_bytes):
                return compressed_peak_estimate(shape,csr,U,nodes,baseline_bytes)
        selection=select_nodes(x,d,p,theta,c,rtol,int(limit),int(memory_budget_bytes),int(n_norm_pairs),baseline,
                               peak_estimator=estimator)
        selection['resource_model']=representation
        info['analytic_selection']=selection
        status=selection['status']
        if status=='memory_limit':raise ResourceLimitError('selector memory admission failed',info)
        if status not in ('selected','exact_equal_depths'):
            raise SelectionError('analytical node selection did not succeed: '+status,info)
        r=selection['interpolation_nodes'];n,g=x.shape
        if representation=='interpolated':estimate=int(selection['estimated_peak_bytes']+3*csr)
        else:estimate=compressed_peak_estimate(x.shape,csr,U,r,baseline)
    info['analytic_selection']=selection;info['resources']['estimated_peak_bytes']=estimate
    if estimate>memory_budget_bytes:raise ResourceLimitError('requested representation exceeds declared admission budget; no fallback',info)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        if representation=='interpolated':op=_Interpolated(x,d,p,theta,c,selection['interpolation_nodes'])
        elif representation=='interpolated_grouped':op=_InterpolatedGrouped(x,d,p,theta,c,selection['interpolation_nodes'],grouping)
        else:op=_ExactGrouped(x,d,p,theta,c,grouping)
    detail=op.storage_detail()
    for field in ('_correction','_mean','_left','_right','_table'):
        a=getattr(op,field,None)
        if a is not None and not np.isfinite(a.data if field=='_correction' else a).all():raise FloatingPointError('nonfinite stabilized coefficients')
    info.update(constructed_representation=representation,arrays_storage_bytes=detail,operator_storage_bytes=sum(detail.values()),
                clipping_counts=op._clips,full_columns=op._full_columns)
    return BuildResult(op,info)
