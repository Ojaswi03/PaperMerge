"""Authenticated deterministic-computation certificates, not BFT consensus.

Nine replicas other than the data owner independently compute a fixed request.
With at most four faulty identities, five identical signed results include an
honest computation. This relies on committed inputs, deterministic execution,
trusted membership and isolated keys; it does not certify arbitrary SGD or
provide distributed request ordering. No queue or research execution here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

REVISION = 'cart_replicated_train_verification_v1'
FAULT_BOUND = 4
CERTIFICATE_SIZE = 5


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class ComputeRequest:
    run_id: str
    receiver: int
    owner: int
    round_id: int
    step: int
    kind: str
    model_hash: str
    data_root: str
    indices: tuple[int,...]
    augmentation_key: tuple[int,...] = ()

    def __post_init__(self):
        if not isinstance(self.run_id,str) or not self.run_id:
            raise ValueError('Run identity required.')
        if any(type(i) is not int or not 0<=i<10 for i in (self.receiver,self.owner)):
            raise ValueError('Logical node identities 0..9 required.')
        if type(self.round_id) is not int or self.round_id<0 or type(self.step) is not int or not 0<=self.step<50:
            raise ValueError('Canonical nonnegative round and activation step 0..49 required.')
        if self.round_id==0 and self.owner>self.receiver:
            raise ValueError('Future owner evidence is unavailable in round zero.')
        if self.kind not in ('gradient','selection_ce') or not _hash(self.model_hash) or not _hash(self.data_root):
            raise ValueError('Explicit computation kind and committed model/data hashes required.')
        if (type(self.indices) is not tuple or not 1<=len(self.indices)<=512 or
                len(set(self.indices))!=len(self.indices) or
                any(type(i) is not int or not 0<=i<50000 for i in self.indices)):
            raise ValueError('Distinct bounded TRAIN indices required.')
        if type(self.augmentation_key) is not tuple or (self.kind=='selection_ce' and self.augmentation_key):
            raise ValueError('Selection must use unaugmented TRAIN images.')
        if self.kind=='gradient' and (len(self.augmentation_key)!=2 or any(
                type(i) is not int or not 0<=i<2**30 for i in self.augmentation_key)):
            raise ValueError('Explicit keyed gradient augmentation required.')

    @property
    def digest(self):
        return digest_json(dict(revision=REVISION,**self.__dict__))

    @property
    def committee(self):
        return tuple(i for i in range(10) if i!=self.owner)


@dataclass(frozen=True)
class ComputeResult:
    cross_entropy: float
    shapes: tuple[tuple[int,...],...]
    tensors: tuple[bytes,...]

    def __post_init__(self):
        import numpy as np
        if not math.isfinite(self.cross_entropy) or self.cross_entropy<0:
            raise ValueError('Finite nonnegative CE required.')
        if type(self.shapes) is not tuple or type(self.tensors) is not tuple or len(self.shapes)!=len(self.tensors):
            raise ValueError('Immutable tensor layout required.')
        for shape,data in zip(self.shapes,self.tensors):
            if (type(shape) is not tuple or any(type(d) is not int or d<=0 for d in shape) or
                    type(data) is not bytes or len(data)!=math.prod(shape)*4 or
                    not np.isfinite(np.frombuffer(data,dtype='<f4')).all()):
                raise ValueError('Finite immutable float32 tensor payload required.')

    @classmethod
    def from_values(cls, ce, gradients=()):
        import numpy as np
        arrays=[np.asarray(g) for g in gradients]
        if any(a.dtype!=np.float32 for a in arrays):
            raise ValueError('Float32 gradients required; no silent precision conversion.')
        return cls(float(ce),tuple(a.shape for a in arrays),tuple(a.astype('<f4',copy=False).tobytes() for a in arrays))

    def arrays(self):
        import numpy as np
        return tuple(np.frombuffer(data,dtype='<f4').reshape(shape) for shape,data in zip(self.shapes,self.tensors))

    @property
    def digest(self):
        h=hashlib.sha256(digest_json(dict(ce=float(self.cross_entropy).hex(),shapes=self.shapes)).encode())
        for data in self.tensors:h.update(data)
        return h.hexdigest()


@dataclass(frozen=True)
class SignedVote:
    signer: int
    request_hash: str
    result: ComputeResult
    signature: bytes


def vote_message(signer,request_hash,result_hash):
    return json.dumps([REVISION,signer,request_hash,result_hash],separators=(',',':')).encode()


def generate_identity():
    # OS-generated keys are NOT derived from the public experiment seed.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private=Ed25519PrivateKey.generate()
    return private,private.public_key().public_bytes_raw()


def sign_result(private_key,signer,request,result):
    return SignedVote(signer,request.digest,result,
        private_key.sign(vote_message(signer,request.digest,result.digest)))


class CertificateCollector:
    """One fixed request and immutable membership; count each signer once."""
    def __init__(self,request,public_keys):
        if not isinstance(request,ComputeRequest) or set(public_keys)!=set(range(10)):
            raise ValueError('Complete pinned ten-member public-key roster required.')
        keys={i:bytes(k) for i,k in public_keys.items()}
        if any(len(k)!=32 for k in keys.values()) or len(set(keys.values()))!=10:
            raise ValueError('Ten distinct Ed25519 public keys required.')
        self.request=request
        self._keys=keys
        self._votes={}
        self.rejected=[]

    def add(self,vote):
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if not isinstance(vote,SignedVote) or type(vote.signer) is not int or vote.signer not in self.request.committee:
            self.rejected.append('nonmember_or_owner');return False
        if vote.request_hash!=self.request.digest:
            self.rejected.append('wrong_request');return False
        if not isinstance(vote.result,ComputeResult) or type(vote.signature) is not bytes:
            self.rejected.append('malformed_payload');return False
        if (self.request.kind=='gradient') != bool(vote.result.tensors):
            self.rejected.append('wrong_result_kind');return False
        try:
            Ed25519PublicKey.from_public_bytes(self._keys[vote.signer]).verify(vote.signature,
                vote_message(vote.signer,vote.request_hash,vote.result.digest))
        except (InvalidSignature,ValueError):
            self.rejected.append('invalid_signature');return False
        if vote.signer in self._votes:
            self.rejected.append('duplicate_or_equivocation');return False
        self._votes[vote.signer]=vote
        return True

    def certificate(self):
        groups={}
        for vote in self._votes.values():groups.setdefault(vote.result.digest,[]).append(vote)
        winners=[v for v in groups.values() if len(v)>=CERTIFICATE_SIZE]
        if len(winners)!=1:
            raise ValueError('No unique five-signature result certificate; do not lower the quorum or invent a reply.')
        winner=sorted(winners[0],key=lambda v:v.signer)
        result=winner[0].result
        metadata=dict(revision=REVISION,requestHash=self.request.digest,resultHash=result.digest,
            committee=list(self.request.committee),matchingSigners=[v.signer for v in winner],
            validReplies=len(self._votes),missingReplicas=sorted(set(self.request.committee)-self._votes.keys()),
            dissentingSigners=sorted(v.signer for v in self._votes.values() if v.result.digest!=result.digest),
            rejected=list(self.rejected),faultBound=FAULT_BOUND,requiredMatches=CERTIFICATE_SIZE,
            signatures={str(v.signer):v.signature.hex() for v in winner},
            membershipHash=digest_json({str(i):k.hex() for i,k in self._keys.items()}))
        return result,metadata
