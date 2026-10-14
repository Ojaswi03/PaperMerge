"""CPU-only, namespace-isolated endpoint. Entry point is not a research runner."""
from __future__ import annotations

import os
import sys


def main(identity,fixture,fault):
    import numpy as np
    import tensorflow as tf
    from .replica_wire import MAX_FRAME,decode,encode,unpack_arrays,pack_vote
    from .replicated_training_service import TrainingReference,TrainingReplica
    from .replicated_verification import ComputeRequest,ComputeResult,sign_result
    from .research_protocol import build_model
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    with np.load(fixture,allow_pickle=False) as data:
        reference=TrainingReference(data['images'],data['labels'],source_split='train',
            synthetic=bool(data['synthetic'].item()))
    replica=TrainingReplica(identity,build_model('basil_paper_cnn',2025),reference)
    def send(value):
        sys.stdout.buffer.write(encode(value));sys.stdout.buffer.flush()
    # No private key crosses this boundary. The bootstrap channel is coordinator-owned.
    send(dict(identity=identity,publicKey=replica.public_key.hex(),dataRoot=reference.root,
        pid=os.getpid(),netNamespace=os.readlink('/proc/self/ns/net'),
        pidNamespace=os.readlink('/proc/self/ns/pid'),gpuVisible=bool(tf.config.list_physical_devices('GPU'))))
    while True:
        raw=sys.stdin.buffer.readline(MAX_FRAME+1)
        if not raw:return
        if len(raw)>MAX_FRAME:raise ValueError('Oversized request.')
        message=decode(raw)
        if message.get('op')=='close':return
        r=dict(message['request']);r['indices']=tuple(r['indices']);r['augmentation_key']=tuple(r['augmentation_key'])
        request=ComputeRequest(**r)
        params=unpack_arrays(message['params'])
        if fault=='exit':return
        if fault=='withhold':
            import time
            time.sleep(3600)  # Fault injection; coordinator must kill on deadline.
        try:
            honest=replica.compute(request,params)
            if fault=='poison':
                honest=ComputeResult.from_values(999.,[np.full_like(a,-100.) for a in honest.arrays()])
            vote=sign_result(replica._private,identity,request,honest)
            payload=pack_vote(vote)
            if fault=='forged':payload['signature']='00'*64
            send(dict(vote=payload))
        except (ValueError,tf.errors.InvalidArgumentError) as error:
            send(dict(error=type(error).__name__,message=str(error)[:500]))
