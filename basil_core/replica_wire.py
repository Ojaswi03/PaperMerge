"""Bounded JSON/float32 wire representation; never deserialize peer pickle."""
from __future__ import annotations

import base64
import json
import math

MAX_FRAME = 4 * 1024 * 1024
MAX_COORDINATES = 117706


def encode(value):
    raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if len(raw)>MAX_FRAME:raise ValueError('Replica frame too large.')
    return raw


def decode(raw):
    if len(raw)>MAX_FRAME:raise ValueError('Replica frame too large.')
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw,parse_constant=invalid)


def pack_arrays(arrays):
    import numpy as np
    arrays=tuple(np.asarray(a) for a in arrays)
    if sum(a.size for a in arrays)>MAX_COORDINATES:raise ValueError('Model exceeds approved wire size.')
    if any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Finite float32 tensors required.')
    return [dict(shape=list(a.shape),data=base64.b64encode(a.astype('<f4',copy=False).tobytes()).decode()) for a in arrays]


def unpack_arrays(payload):
    import numpy as np
    if not isinstance(payload,list) or len(payload)>32:raise ValueError('Invalid tensor list.')
    arrays=[];total=0
    for item in payload:
        shape=item['shape']
        if (not isinstance(shape,list) or len(shape)>4 or
                any(type(i) is not int or i<=0 for i in shape)):
            raise ValueError('Invalid tensor shape.')
        total+=math.prod(shape)
        if total>MAX_COORDINATES:raise ValueError('Too many tensor coordinates.')
        raw=base64.b64decode(item['data'],validate=True)
        if len(raw)!=math.prod(shape)*4:raise ValueError('Tensor byte count mismatch.')
        a=np.frombuffer(raw,dtype='<f4').reshape(shape)
        if not np.isfinite(a).all():raise ValueError('Nonfinite tensor payload.')
        arrays.append(a)
    return tuple(arrays)


def pack_vote(vote):
    return dict(signer=vote.signer,request_hash=vote.request_hash,
        ce=vote.result.cross_entropy,tensors=pack_arrays(vote.result.arrays()),
        signature=vote.signature.hex())


def unpack_vote(payload):
    from .replicated_verification import ComputeResult,SignedVote
    if not isinstance(payload,dict):raise ValueError('Vote object required.')
    return SignedVote(payload['signer'],payload['request_hash'],
        ComputeResult.from_values(payload['ce'],unpack_arrays(payload['tensors'])),
        bytes.fromhex(payload['signature']))
