"""Exact training-time scaler and lossless float32 comparison encoding."""
import numpy as np
def scale_features(scaler,values):
    x=np.asarray(values,dtype=np.float64).copy()
    if scaler.with_centering:x-=scaler.center_
    if scaler.with_scaling:x/=scaler.scale_
    return x.astype(np.float32)
def encode_features(values):
    x=np.asarray(values,dtype=np.float32)
    if x.shape!=(27,) or not np.isfinite(x).all():
        raise ValueError('Expected 27 finite scaled features')
    x=x.copy();x[x==0]=0
    bits=x.view(np.uint32)
    return np.where((bits>>31)!=0,~(bits&0x7fffffff),bits).astype(np.uint32).view(np.int32)
