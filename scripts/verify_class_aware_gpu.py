#!/usr/bin/env python3
"""Synthetic CPU/GPU class-aware gradients/updates; no CIFAR research training."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basil_core.iid_runtime import prepare_environment,configure_tensorflow,GPU_MEMORY_LIMIT_MB
from basil_core.class_aware_retention import ClassAwareRetentionWorker
from basil_core.research_protocol import build_model,params_hash


def main():
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    with tf.device('/CPU:0'):
        initial=[w.numpy() for w in build_model('basil_paper_cnn',2025).trainable_variables]
    rng=np.random.default_rng(17)
    images=rng.normal(0,.2,(512,32,32,3)).astype(np.float32)
    labels=np.zeros(512,np.int32)
    outcomes={}
    for device in ('CPU','GPU'):
        with tf.device(f'/{device}:0'):
            worker=ClassAwareRetentionWorker(build_model('basil_paper_cnn',2025),mu=.01,device=device)
            worker.load(initial)
            for ref,w in zip(worker.reference,initial):ref.assign(w+.001)
            for mask in worker.diagonal:mask.assign(tf.ones_like(mask)*3)
            refhash=params_hash([r.numpy() for r in worker.reference])
            repeats=[]
            for legacy in (1.,999999.):
                grads,values=worker._gradients(tf.constant(images),tf.constant(labels),tf.constant([7,8]),
                    'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(legacy))
                repeats.append(([g.numpy() for g in grads],np.asarray([float(v) for v in values])))
            for a,b in zip(repeats[0][0],repeats[1][0]):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(repeats[0][1],repeats[1][1])
            partial,_=worker._gradients(tf.constant(images[:392]),tf.constant(labels[:392]),tf.constant([7,8]),
                'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(0.))
            assert all(np.isfinite(g.numpy()).all() for g in partial)
            assert all(device in g.device for g in grads)
            tf.keras.optimizers.SGD(.05,momentum=0.).apply_gradients(zip(grads,worker.model.trainable_variables))
            assert params_hash([r.numpy() for r in worker.reference])==refhash
            outcomes[device]=(repeats[0],worker.export())
    (a,av),au=outcomes['CPU'];(b,bv),bu=outcomes['GPU']
    diff=np.concatenate([(x-y).reshape(-1) for x,y in zip(a,b)])
    norm=np.linalg.norm(np.concatenate([x.reshape(-1) for x in a]))
    relative=float(np.linalg.norm(diff)/norm)
    assert relative<.01,relative
    np.testing.assert_allclose(av[:4],bv[:4],atol=1e-6,rtol=1e-4)
    update_max=max(float(np.max(np.abs(x-y))) for x,y in zip(au,bu))
    assert update_max<1e-4
    memory=tf.config.experimental.get_memory_info('GPU:0')
    assert memory['peak']<.75*GPU_MEMORY_LIMIT_MB*1024**2
    result=dict(status='passed',researchRunsStarted=0,gradientRelativeL2Difference=relative,
        maxParameterUpdateDifference=update_max,gradientTolerance=.01,updateTolerance=1e-4,
        legacyLambdaIgnored=True,referenceUnchanged=True,batchSizes=[512,392],gpuMemory=memory,
        objectiveSha256=hashlib.sha256((ROOT/'basil_core/class_aware_retention.py').read_bytes()).hexdigest())
    (ROOT/'gui/worker_state/CLASS_AWARE_GPU_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
