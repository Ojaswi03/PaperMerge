"""Scheduling must preserve independently signed exact results and quorum rules."""
import threading

import pytest


def endpoints(params, *, faults=(), barrier=None):
    import numpy as np
    from basil_core.replicated_verification import generate_identity, ComputeResult, sign_result, SignedVote
    result=ComputeResult.from_values(2.,[np.ones_like(p) for p in params])
    peers={}
    for identity in range(10):
        private,public=generate_identity()
        class Peer:
            def __init__(self,i,key,pub):
                self.identity=i;self.private=key;self.public_key=pub
            def reply(self,request,sealed):
                if barrier is not None:barrier.wait(timeout=5)
                if self.identity in faults:raise TimeoutError('injected missing reply')
                assert all(not a.flags.writeable for a in sealed)
                return sign_result(self.private,self.identity,request,result)
        peers[identity]=Peer(identity,private,public)
    return peers,result


@pytest.mark.parametrize('faults',[(),(0,1,5,7)])
def test_parallel_proofs_equal_serial_and_every_endpoint_computes(faults):
    import numpy as np
    from basil_core.research_protocol import params_hash
    from basil_core.replicated_verification import ComputeRequest
    from basil_core.replicated_training_service import ReplicatedTrainingService
    from basil_core.parallel_verifier_service import ParallelVerifierService
    params=[np.asarray([1.,2.],np.float32)]
    request=ComputeRequest('unit',0,4,20,0,'gradient',params_hash(params),'a'*64,(4,),(7,8))
    peers,result=endpoints(params,faults=faults)
    serial,proof=ReplicatedTrainingService(peers).evaluate(request,params)
    # A barrier would fail under sequential scheduling: all nine must enter.
    barrier=threading.Barrier(9)
    original={i:p.reply for i,p in peers.items()}
    for i,p in peers.items():
        def reply(request,sealed,identity=i):
            barrier.wait(timeout=5)
            return original[identity](request,sealed)
        p.reply=reply
    with ParallelVerifierService(peers) as service:
        parallel,other=service.evaluate(request,params)
        assert service.calls==9
    assert parallel==serial==result
    assert other['signatures']==proof['signatures']
    assert other['matchingSigners']==proof['matchingSigners']
    assert other['requiredMatches']==5
    np.testing.assert_array_equal(params[0],[1.,2.])


def test_parallel_service_cannot_certify_fewer_than_five_matching_replies():
    import numpy as np
    from basil_core.research_protocol import params_hash
    from basil_core.replicated_verification import ComputeRequest
    from basil_core.parallel_verifier_service import ParallelVerifierService
    params=[np.asarray([1.],np.float32)]
    request=ComputeRequest('unit',0,4,20,0,'gradient',params_hash(params),'a'*64,(4,),(7,8))
    peers,_=endpoints(params,faults=(0,1,2,5,7))
    with ParallelVerifierService(peers) as service:
        with pytest.raises(ValueError):service.evaluate(request,params)

