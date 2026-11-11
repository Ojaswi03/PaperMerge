"""Demonstration selection, honest reporting, strict read-only model inference."""
import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import test_fedsharp_image as demo


def test_cross_entropy_is_stable_and_not_a_training_objective():
    logits=np.array([10000.,9999.,-10000.,0.,0.,0.,0.,0.,0.,0.])
    row=demo.prediction(logits,1)
    assert row['predictedClass']==0 and row['correct'] is False
    assert row['crossEntropyLoss']==pytest.approx(np.log1p(np.exp(-1))+1)
    assert sum(row['probabilities'])==pytest.approx(1)
    assert row['predictionConfidence']==pytest.approx(1/(1+np.exp(-1)))
    assert demo.prediction(np.zeros(10),0)['crossEntropyLoss']==pytest.approx(np.log(10))
    with pytest.raises(ValueError):demo.prediction(np.full(10,np.nan),0)


def public_cases():
    return json.loads((demo.ROOT/'docs/results/themachine_fedsharp-fleet-500r-20261103/plan.json').read_text())['configs']


def queue_fixture(tmp_path):
    cases=public_cases();queue=tmp_path/'queue';(queue/'fleet').mkdir(parents=True)
    (queue/'continuation_plan.json').write_text(json.dumps(dict(items=[dict(config=c) for c in cases])))
    runs={c['runId']:dict(state='DONE',owner='themachine',location=dict(output='/host-local/final')) for c in cases}
    (queue/'fleet/state.json').write_text(json.dumps(dict(runs=runs)))
    return queue


def test_selects_exact_four_matching_500_round_cases_without_tensorflow_or_test_loading(tmp_path,monkeypatch):
    queue=queue_fixture(tmp_path)
    monkeypatch.setattr(demo,'official_image',lambda *a:pytest.fail('Description must not open TEST.'))
    got=demo.select_runs(queue,['mnist','cifar10'],'hidden',2025)
    assert len(got)==4
    assert [r['method'] for r in got]==['FedSharp','FedSharp+']*2
    assert {r['owner'] for r in got}=={'themachine'}
    assert {r['view']['runId'] for r in got}=={
        'cifar10-dirichlet-2025-sharp_joint-ef8b6adcfc43',
        'cifar10-dirichlet-2025-sharp_cart-6aea2e473335',
        'mnist-dirichlet-2025-sharp_joint-905dcf7b651b',
        'mnist-dirichlet-2025-sharp_cart-46236275f264'}
    registry=json.loads((queue/'fleet/state.json').read_text())
    registry['runs'][got[0]['view']['runId']]['state']='RUNNING'
    (queue/'fleet/state.json').write_text(json.dumps(registry))
    with pytest.raises(ValueError,match='not DONE'):demo.select_runs(queue,['mnist'],'hidden',2025)


def test_remote_host_refused_before_checkpoint_or_test_reads():
    item=dict(view=dict(runId='run'),owner='different-'+socket.gethostname().split('.')[0],output=Path('/not-local'))
    with pytest.raises(ValueError,match='host-local'):demo.preflight([item])


def test_local_mode_uses_published_contracts_and_requires_four_actual_checkpoint_files(tmp_path):
    chosen=demo.select_local_runs(tmp_path,['mnist','cifar10'],'hidden',2025)
    assert len(chosen)==4 and all(c['allowRuntimeDifference'] for c in chosen)
    assert all(c['checkpoint']==tmp_path/c['view']['runId']/'checkpoint.npz' for c in chosen)
    with pytest.raises(FileNotFoundError,match='checkpoint.npz'):demo.preflight(chosen)
    # Preflight validates exported rows/sources separately from the strict
    # checkpoint loader, which is exercised by real-model inference below.
    for c in chosen:
        c['checkpoint'].parent.mkdir();c['checkpoint'].write_bytes(b'preflight fixture, not a trained model')
    got=demo.preflight(chosen)
    assert len(got)==4
    assert all(c['checkpoint']==c['checkpoint'].resolve() for c in got)
    assert all(str(c['checkpoint']) in c['before'] for c in got)


def test_official_image_uses_test_not_training_and_checks_full_shape(monkeypatch):
    import scripts.verify_cart_comparison_results as source
    x=np.zeros((10000,28,28,1),np.float32);y=np.arange(10000,dtype=np.int32)%10
    x[7]=.5
    monkeypatch.setattr(source,'test_data',lambda ds:(x,y,dict(path='cached_official_test',sha256='fixture')))
    image,label,identity=demo.official_image('mnist',7)
    assert label==7 and image.shape==(1,28,28,1) and np.all(image==.5)
    assert identity['split']=='official_test' and identity['distinctTestImagesEvaluated']==1
    with pytest.raises(ValueError):demo.official_image('mnist',10000)
    monkeypatch.setattr(source,'test_data',lambda ds:(x[:100],y[:100],{}))
    with pytest.raises(ValueError,match='10000'):demo.official_image('mnist',0)


@pytest.mark.parametrize('addon',[None,'cart'])
def test_real_model_inference_uses_retained_params_and_never_trains_or_changes_files(tmp_path,monkeypatch,addon):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    from basil_core.basil_sharp_thesis import effective_config
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import save_state,initial_state,vector_hash
    view=next(c for c in public_cases() if c['dataset']=='mnist' and c['seed']==2025 and c['partition']=='dirichlet'
              and c['attackType']=='hidden' and c['comparisonCase']['family']=='sharp'
              and c['comparisonCase']['environment']=='joint' and c['comparisonCase']['addon']==addon)
    cfg=effective_config(view);worker=SharpWorker(cfg,'CPU');initial=worker.export();n=cfg['nodes']
    state=initial_state(cfg,initial);state['cursor']=n*cfg['rounds'];state['senderRounds'][:]=cfg['rounds']-1
    # Distinct outgoing copies detect accidental evaluation of attacked outbound models.
    state['outbound'][:]=np.float32(.123)
    state['roundMetrics']=[dict(nodeAccuracy=[.1]*n) for _ in range(cfg['rounds'])]
    state['telemetry']=[dict(node=i%n,trainedHash=vector_hash(initial)) for i in range(state['cursor'])]
    contract=dict(config=cfg);folder=tmp_path/view['runId'];folder.mkdir()
    checkpoint=folder/'checkpoint.npz';save_state(checkpoint,state,contract)
    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((cfg['rounds'],n),.1))
    before={p:demo.sha(p) for p in folder.iterdir()}
    row=dict(runtime=dict(tensorflow=tf.__version__,numpy=np.__version__),initialHash=vector_hash(initial),contract=contract)
    item=dict(config=cfg,row=row,view=view,output=tmp_path,method='FedSharp+' if addon else 'FedSharp',
        before={str(p):v for p,v in before.items()})
    image=np.zeros((1,28,28,1),np.float32);expected=demo.prediction(worker.outputs(initial,image)[0],0)
    monkeypatch.setattr(SharpWorker,'train',lambda *a,**k:pytest.fail('Inference must not train.'))
    result=demo.infer_case(item,image,0)
    assert len(result['perClient'])==100
    assert result['allClients']['meanCrossEntropyLoss']==pytest.approx(expected['crossEntropyLoss'])
    assert all(r['probabilities']==expected['probabilities'] and r['modelSha256']==vector_hash(initial) for r in result['perClient'])
    assert result['honestClients']['clients']==67
    assert {p:demo.sha(p) for p in folder.iterdir()}==before
    item['row']['runtime']['tensorflow']='wrong-version'
    with pytest.raises(ValueError,match='versions differ'):demo.infer_case(item,image,0)
    item['allowRuntimeDifference']=True
    local=demo.infer_case(item,image,0)
    assert local['evaluationRuntime']['matchesTrainingVersions'] is False
    assert local['allClients']==result['allClients']


def test_existing_destination_refused_before_loading_test_data(tmp_path,monkeypatch):
    monkeypatch.setattr(demo,'preflight',lambda *a:pytest.fail('Existing receipt must be preserved.'))
    with pytest.raises(FileExistsError):demo.evaluate([],0,tmp_path)
