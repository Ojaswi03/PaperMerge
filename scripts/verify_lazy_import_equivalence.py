#!/usr/bin/env python3
"""Before/after deterministic synthetic loss, gradient and CART state check.

Two examples and five tiny unit-test updates; no CIFAR research run or W&B.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT/'newResults/nonIID/strict_cart/clean_full_data_three_round_v1'


def measurements():
    import numpy as np
    import tensorflow as tf
    from basil_core.strict_cart import StrictCartWorker
    from basil_core.class_registry import ClassRegistry
    from basil_core.research_protocol import params_hash, select_snapshot, Snapshot
    model = tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10,
        kernel_initializer='zeros', bias_initializer='zeros')])
    worker = StrictCartWorker(model, mu_max=.01)
    rng = np.random.default_rng(17)
    initial = [rng.normal(0,.03,w.shape).astype(np.float32) for w in model.trainable_variables]
    worker.load(initial)
    x = rng.uniform(0,1,(2,32,32,3)).astype(np.float32); y = np.zeros(2,np.int32)
    loss, accuracy = worker.probe(initial,x,y)
    candidates = [Snapshot(i,-1,[a.copy() for a in initial]) for i in range(5,10)]
    chosen, _ = select_snapshot(0,candidates,lambda p: worker.probe(p,x,y))
    objectives = {}
    for mode in ('none','gradient_norm_objective'):
        worker.load(initial)
        for ref,a in zip(worker.reference,initial):ref.assign(a+.001)
        worker.anchor_mu.assign(.007)
        gradients,terms=worker._gradients(tf.constant(x),tf.constant(y),tf.constant([7,8]),
            mode,'paper_absolute_gaussian',tf.constant(.010),tf.constant(999999.))
        tf.keras.optimizers.SGD(.05).apply_gradients(zip(gradients,model.trainable_variables))
        objectives[mode]=dict(terms=[float(v) for v in terms],
            gradients=[g.numpy().tolist() for g in gradients],update=[a.tolist() for a in worker.export()])
    trained,record=worker.train(initial,x,y,local_registry=ClassRegistry(),received_registry=ClassRegistry(),
        selection_x=x,selection_y=y,selected_loss=loss,seed=2025,round_id=0,node_id=0,
        epochs=5,batch_size=512,learning_rate=.05,ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    return dict(objectives=objectives,initialHash=params_hash(initial),trainedHash=params_hash(trained),
        selectedSender=chosen.sender_id,selectionLoss=loss,selectionAccuracy=accuracy,
        cartControl=record['cartControl'],registry=record['outputRegistry'],
        predictions=worker.predict(tf.constant(x)).numpy().tolist(),steps=record['steps'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-before',action='store_true')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',
        TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/papermerge-import-equivalence-mpl')
    actual=measurements(); before=OUTPUT/'lazy_import_before.json'
    if args.record_before:
        if before.exists():raise ValueError('Refusing overwrite of before evidence.')
        before.write_text(json.dumps(actual,sort_keys=True,indent=2)+'\n')
        print(json.dumps(dict(status='recorded',path=str(before),researchRunsStarted=0)))
    else:
        expected=json.loads(before.read_text())
        if actual!=expected:raise AssertionError('Import refactor changed deterministic mathematical results.')
        proof=dict(status='passed',comparison='exact equality on deterministic CPU synthetic inputs',
            checks=list(actual),researchRunsStarted=0,lossesIdentical=True,gradientsIdentical=True,
            parameterUpdatesIdentical=True,selectionIdentical=True,predictionsIdentical=True)
        (OUTPUT/'lazy_import_equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps(proof))


if __name__=='__main__':main()
