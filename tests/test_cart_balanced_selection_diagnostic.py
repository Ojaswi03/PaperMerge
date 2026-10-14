"""No-dataset-learning contracts for the bounded selection comparison."""
import json
import subprocess
import sys

import pytest


def test_dry_plan_is_lazy_paired_clean_fixed_anchor():
    command = ('from scripts.run_cart_balanced_selection_diagnostic import main; main(); '
               "import sys; assert not ({'tensorflow','numpy','wandb','matplotlib'} & sys.modules.keys())")
    p = subprocess.run([sys.executable, '-c', command], text=True, capture_output=True, check=True)
    plan = json.loads(p.stdout)
    assert plan['trainingRunsStarted'] == 0
    assert plan['differences'] == ['condition','snapshotSelectionRule','startingModelRule']
    for c in plan['configs']:
        assert c['nRounds'] == 3 and c['localEpochs'] == 5 and c['batchSize'] == 512
        assert c['snapshotAnchorMu'] == .01 and c['snapshotSelection']
        assert not c['actualAttackers'] and not c['ebm'] and not c['channelNoise']
        assert not c['liveTrackingEnabled'] and not c['researchValid']
        assert c['snapshotAnchorRule'] == 'fixed_frozen_selected_snapshot'


def test_same_owner_batches_but_different_selection_scores():
    import numpy as np
    from basil_core.research_protocol import Snapshot
    from scripts.run_cart_balanced_selection_diagnostic import choose_start
    labels = np.repeat(np.arange(10,dtype=np.int32),5000)
    images = np.arange(50000).reshape(-1,1)
    selections = [np.arange(c*5000,c*5000+512) for c in range(10)]
    candidates = [Snapshot(s,0,[np.array([s],np.float32)]) for s in (6,7,8,9,0)]
    class Worker:
        def __init__(self): self.seen=[]
        def probe(self,params,x,y):
            owner=int(y[0]); sender=int(params[0][0]);self.seen.append((sender,owner,tuple(x[:,0])))
            # Sender 9 wins the receiver-only CE; sender 0 retains both classes.
            loss = (0. if owner==1 else 10.) if sender==9 else (1. if sender==0 else 3.)
            return loss, .2
    signatures=[]
    for control,expected in [('receiver_local_ss',9),('balanced_owner_ss',0)]:
        worker=Worker()
        selected,records,loss,decision=choose_start(1,candidates,worker=worker,images=images,
            labels=labels,selections=selections,available_owners=[0,1],control=control)
        assert selected.sender_id==expected and len(worker.seen)==10
        assert decision['sourceSplit']=='train' and decision['availableOwners']==[0,1]
        assert loss==(0 if expected==9 else 1)
        assert records[0]['ownerLosses'].keys()=={0,1}
        signatures.append(worker.seen)
    assert signatures[0]==signatures[1]


def test_fixed_anchor_is_unchanged_for_all_five_epochs(monkeypatch):
    import numpy as np
    import tensorflow as tf
    from basil_core.class_registry import ClassRegistry
    from basil_core.strict_cart import StrictCartWorker
    from basil_core.research_protocol import params_hash
    from scripts.run_cart_balanced_selection_diagnostic import create_worker
    def model():
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
            tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,kernel_initializer='zeros')])
    worker=create_worker(model(),model(),device='CPU')
    # Exercise the real five-epoch controller on two synthetic examples, with
    # the independently tested ordinary/anchor batch kernel (no helper data).
    worker._gradients=tf.function(worker._batch_gradients,reduce_retracing=True)
    initial=worker.export();x=np.ones((2,32,32,3),np.float32)*.3;y=np.zeros(2,np.int32)
    outputs=[]
    for local_loss in (0.,100.):
        params,record=StrictCartWorker.train(worker,initial,x,y,local_registry=ClassRegistry(),
            received_registry=ClassRegistry(),selection_x=x,selection_y=y,selected_loss=local_loss,
            seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,
            ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
        assert record['steps']==5 and len(record['cartControl'])==6
        assert all(c['mu']==.01 for c in record['cartControl'])
        assert record['snapshotAnchorReferenceHash']==params_hash(initial)
        assert all(s['snapshotAnchorMu']==pytest.approx(.01) for s in record['optimizerTelemetry'])
        outputs.append(params_hash(params))
    assert outputs[0]==outputs[1]


def test_new_report_rejects_partial_and_unknown_controls():
    from scripts.report_cart_balanced_selection_diagnostic import validate_records
    with pytest.raises(ValueError,match='Unknown'):validate_records([], 'not_a_control')
    with pytest.raises(ValueError,match='Thirty'):validate_records([], 'balanced_owner_ss')
