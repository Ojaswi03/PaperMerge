"""Concurrent independent verifier RPCs with deterministic certificate assembly.

One coordinator performs SGD. Each CPU endpoint recomputes its own result.
Parallelism changes request latency, not vote order or the accepted gradient.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .replicated_training_service import ReplicatedTrainingService
from .replicated_verification import CertificateCollector


class ParallelVerifierService(ReplicatedTrainingService):
    def __init__(self, endpoints):
        super().__init__(endpoints)
        self.executor = ThreadPoolExecutor(max_workers=9, thread_name_prefix='cart-verifier')

    def evaluate(self, request, params):
        import numpy as np
        from .research_protocol import params_hash
        arrays = [np.asarray(p) for p in params]
        if not arrays or any(p.dtype != np.float32 or not np.isfinite(p).all() for p in arrays):
            raise ValueError('Finite float32 model required.')
        sealed = tuple(np.frombuffer(p.tobytes(), dtype=np.float32).reshape(p.shape) for p in arrays)
        if params_hash(sealed) != request.model_hash:
            raise ValueError('Requested model hash mismatch.')
        self.calls += len(request.committee)
        futures = {i:self.executor.submit(self.replicas[i].reply, request, sealed) for i in request.committee}
        collector = CertificateCollector(request, self.public_keys)
        unavailable = []
        # Always consume in canonical committee order, never completion order.
        for identity in request.committee:
            try:
                vote = futures[identity].result()
            except (TimeoutError, ValueError, FloatingPointError) as error:
                unavailable.append(dict(replica=identity,errorType=type(error).__name__,reason=str(error)))
                continue
            if getattr(vote, 'signer', None) != identity:
                collector.rejected.append('endpoint_identity_mismatch')
                continue
            collector.add(vote)
        result, proof = collector.certificate()
        if request.kind == 'gradient' and result.shapes != tuple(p.shape for p in sealed):
            raise ValueError('Certified gradient layout differs from requested model.')
        proof.update(endpointCalls=len(request.committee), unavailableReplicas=unavailable,
            physicalRPC=True, logicalSimulation=False, privateDataReplication=True,
            fullResearchReady=False, isolation='Linux namespaces; read-only code/TRAIN; no network',
            transport='local pipes / bounded JSON; concurrent independent endpoints',
            replicaMemoryCapBytes=2048*1024**2)
        return result, proof

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

