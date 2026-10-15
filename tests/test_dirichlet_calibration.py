"""Predeclared diagnostic math and isolation; no research-run launch in tests."""
import copy
import subprocess
import sys

import pytest

from basil_core.dirichlet_calibration import (calibration_offsets, calibrated_anchor_gradients,
    diagnostic_configs, validate_diagnostic_config, DirichletCalibrationWorker)


def model():
    import tensorflow as tf
    net=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(10,use_bias=False)])
    net.trainable_variables[0].assign([[.1*i for i in range(10)]])
    return net


def test_offsets_finite_absent_and_rare_classes_not_invented():
    import numpy as np
    counts=np.array([0,1,16,81,256,625,1296,2401,4096,6561])
    value=calibration_offsets(counts)
    np.testing.assert_allclose(value,[1,1,.5,1/3,.25,.2,1/6,1/7,.125,1/9],rtol=1e-6)
    assert counts[0]==0 and np.isfinite(value).all()
    np.testing.assert_array_equal(calibration_offsets(counts,tau=0),np.zeros(10))
    with pytest.raises(ValueError):calibration_offsets(np.zeros(10,dtype=int))
    with pytest.raises(ValueError):calibration_offsets(np.ones(10),tau=1)


@pytest.mark.parametrize('coefficient',[0.,.0001])
@pytest.mark.parametrize('method',['fedlc','prior_adjustment'])
def test_calibrated_objective_gradient_matches_hessian_plus_anchor(coefficient,method):
    import numpy as np
    import tensorflow as tf
    net=model(); w=net.trainable_variables[0]
    x=tf.constant([[1.],[2.]]);y=tf.constant([0,8])
    offsets=calibration_offsets(np.array([1,2,3,4,5,6,7,8,9,10]),method=method)
    reference=[tf.ones_like(w)*.2]
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=net(x)-offsets))
        base=inner.gradient(ce,w)
    h=outer.jacobian(base,w).numpy().reshape(10,10)
    gradients,values=calibrated_anchor_gradients(net,x,y,reference,offsets,mu=.01,coefficient=coefficient)
    expected=base.numpy().ravel()+2*coefficient*h@base.numpy().ravel()+.01*(w.numpy()-reference[0].numpy()).ravel()
    np.testing.assert_allclose(gradients[0].numpy().ravel(),expected,rtol=2e-5,atol=1e-7)
    loss=float(ce)+coefficient*float(tf.reduce_sum(base**2))+.01/2*float(tf.reduce_sum((w-reference[0])**2))
    assert float(values[2])==pytest.approx(loss,rel=1e-6)


def test_uniform_offsets_cancel_and_zero_tau_preserves_original_gradient():
    import numpy as np
    import tensorflow as tf
    from basil_core.noniid_study import anchored_gradients
    net=model();x=tf.constant([[1.],[2.]]);y=tf.constant([0,8]);reference=[tf.identity(w) for w in net.trainable_variables]
    ordinary,values=anchored_gradients(net,x,y,reference,mu=.01,sigma_e=0,ebm=False)
    for offset in [np.zeros(10,np.float32),calibration_offsets(np.full(10,500))]:
        calibrated,result=calibrated_anchor_gradients(net,x,y,reference,offset,mu=.01)
        np.testing.assert_allclose(calibrated[0].numpy(),ordinary[0].numpy(),atol=1e-7,rtol=1e-6)
        assert float(result[0])==pytest.approx(float(values[0]),rel=1e-6)


def test_prior_adjustment_removes_constant_logit_empirical_prior_gradient():
    import numpy as np
    import tensorflow as tf
    counts=np.arange(1,11)
    labels=tf.constant(np.repeat(np.arange(10,dtype=np.int32),counts))
    logits=tf.Variable(tf.zeros(10))
    offsets=calibration_offsets(counts,method='prior_adjustment')
    with tf.GradientTape() as tape:
        adjusted=tf.broadcast_to(logits-offsets,(55,10))
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=adjusted))
    gradient=tape.gradient(loss,logits).numpy()
    # Float32 log/softmax and the 55-example reduction accumulate roundoff;
    # 1e-7 is below one float32 epsilon and far below the unadjusted bias.
    np.testing.assert_allclose(gradient,0.,atol=1e-7)
    reference=tf.Variable(tf.zeros(10,dtype=tf.float64))
    exact_prior=tf.constant(counts/counts.sum(),tf.float64)
    with tf.GradientTape() as tape:
        z=tf.broadcast_to(reference+tf.math.log(exact_prior),(55,10))
        exact_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=z))
    np.testing.assert_allclose(tape.gradient(exact_loss,reference).numpy(),0.,atol=1e-14)
    assert np.linalg.norm(np.full(10,.1)-counts/counts.sum())>.1
    prior=np.exp(-calibration_offsets(np.array([0,1,2,3,4,5,6,7,8,9]),method='prior_adjustment'))
    assert np.isfinite(prior).all() and prior[0]>0
    assert prior.sum()==pytest.approx(1.,rel=1e-6)


def test_prior_variant_is_explicit_and_cannot_enter_original_queue():
    from basil_core.dirichlet_calibration import prior_diagnostic_config
    from basil_core.dirichlet_study import gui_approved
    cfg=prior_diagnostic_config('a'*64,'b'*64)
    validate_diagnostic_config(cfg)
    assert cfg['localObjective']=='logit_prior_snapshot_anchor' and not gui_approved(cfg)
    assert cfg['logitCalibrationTau']==1. and cfg['snapshotAnchorMu']==.01


def test_explicit_diagnostic_configs_keep_baseline_queue_unapproved():
    from basil_core.research_protocol import validate_config
    from basil_core.dirichlet_study import gui_approved
    configs=diagnostic_configs('a'*64,'b'*64)
    for cfg in configs:
        validate_config(cfg)
        assert not gui_approved(cfg)
        assert cfg['localEpochs']==5 and cfg['batchSize']==512
        assert cfg['nRounds']==3 and cfg['snapshotAnchorMu']==.01
        assert not cfg['useChannelNoise'] and cfg['resolvedAttackerIds']==[]
        assert cfg['snapshotSelectionRule']=='lowest_receiver_local_loss'
    differences={k for k in configs[0] if configs[0][k]!=configs[1][k]}
    assert differences=={'localObjective','logitCalibrationEnabled','displayName','experimentName','runId','resultDirectory','plotDirectory'}
    for key,value in [('nRounds',30),('snapshotAnchorMu',.02),('useChannelNoise',True),('localEpochs',1),('researchValid',True)]:
        with pytest.raises(ValueError):validate_diagnostic_config(dict(configs[1],**{key:value}))


def test_small_five_epoch_activation_keeps_reference_and_full_local_accuracy():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    net=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer='ones',bias_initializer='zeros')])
    worker=DirichletCalibrationWorker(net,calibration=True)
    start=worker.export();count=515
    x=np.zeros((count,32,32,3),np.float32);y=np.concatenate([np.zeros(512,np.int32),np.ones(3,np.int32)])
    trained,metrics=worker.train(start,x,y,seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,
        learning_rate=.05,ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    assert metrics['steps']==10 and all(e['batchSizes']==[512,3] for e in metrics['epochs'])
    assert metrics['accuracyScope']=='full_local_train_unaugmented'
    assert metrics['accuracy']==pytest.approx(float(np.mean(net(x).numpy().argmax(axis=1)==y)))
    assert params_hash([v.numpy() for v in worker.reference])==params_hash(start)


def test_calibration_selection_still_ordinary_ce_and_lambda_ignored():
    import numpy as np
    import tensorflow as tf
    from basil_core.trainer import lossFn
    net=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer='ones',bias_initializer='zeros')])
    worker=DirichletCalibrationWorker(net,calibration=True);params=worker.export()
    worker.offsets.assign(calibration_offsets(np.array([0,1,2,3,4,5,6,7,8,9])))
    x=tf.zeros((4,32,32,3));y=tf.constant([0,1,1,2]);key=tf.constant([7,8])
    assert worker.probe(params,x,y)[0]==float(lossFn(y,net(x)))
    for ref,p in zip(worker.reference,params):ref.assign(p)
    outputs=[worker._gradients(x,y,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(lam)) for lam in [1.,999999.]]
    for gradients,metrics in zip(outputs[0],outputs[1]):
        for first,second in zip(gradients,metrics):np.testing.assert_array_equal(np.asarray(first),np.asarray(second))


@pytest.mark.parametrize('module',['basil_core.dirichlet_calibration','scripts.run_dirichlet_calibration',
                                  'reporting.dirichlet_calibration'])
def test_new_imports_lazy(module):
    subprocess.run([sys.executable,'-c',f'import importlib,sys;importlib.import_module({module!r});'
        'assert not ({"numpy","tensorflow","wandb","matplotlib"} & sys.modules.keys())'],check=True)


def test_cli_default_starts_no_training():
    import json
    result=subprocess.run([sys.executable,'scripts/run_dirichlet_calibration.py'],check=True,capture_output=True,text=True)
    payload=json.loads(result.stdout)
    assert not payload['trainingStarted'] and payload['localEpochs']==5 and payload['rounds']==3


def test_cli_refuses_other_output_trees():
    result=subprocess.run([sys.executable,'scripts/run_dirichlet_calibration.py',
        '--output-root','newResults/IID/not_allowed'],capture_output=True,text=True)
    assert result.returncode==2 and 'must be inside newResults/nonIID' in result.stderr


def test_reporting_refuses_partial_run_conclusions(tmp_path):
    import json
    from reporting.dirichlet_calibration import review
    path=tmp_path/'baseline';path.mkdir()
    (path/'run.json').write_text(json.dumps(dict(status='running',completedRounds=2,completedActivations=24)))
    with pytest.raises(ValueError,match='completed'):review(tmp_path)


@pytest.mark.parametrize('stage,rounds',[('clean',10),('noise',3),('attack_gate',23),('threat',30)])
def test_prospective_readiness_configs_are_bounded_not_production(stage,rounds):
    from basil_core.dirichlet_calibration import readiness_configs
    from basil_core.dirichlet_study import gui_approved
    from basil_core.research_protocol import validate_config
    configs=readiness_configs('a'*64,'b'*64,stage)
    for cfg in configs:
        validate_config(cfg)
        assert cfg['nRounds']==rounds and cfg['localEpochs']==5 and cfg['batchSize']==512
        assert not cfg['researchValid'] and not gui_approved(cfg)
        assert cfg['snapshotPlausibilityGuard']=='none' and cfg['basilMemorySize']==5
        assert cfg['attackStartRound']==20 and cfg['logitCalibrationTau']==1.
        if stage not in {'threat','attack_gate'}:assert cfg['resolvedAttackerIds']==[]
        for key,value in [('nRounds',500),('snapshotAnchorMu',.1),('channelNoiseSigmaAbsolute',.02),('researchValid',True)]:
            with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
    if stage=='threat':
        from basil_core.dirichlet_study import verify_pairing
        assert verify_pairing(configs)['C/D']==['ebmMode','localObjective']


def test_prior_ebm_worker_lambda_loss_gradients_and_update_identical():
    import numpy as np
    import tensorflow as tf
    from basil_core.dirichlet_calibration import GATE_REVISION
    net=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer='ones')])
    worker=DirichletCalibrationWorker(net,calibration=True,method='prior_adjustment',revision=GATE_REVISION)
    worker.offsets.assign(calibration_offsets(np.arange(1,11),method='prior_adjustment'))
    initial=worker.export()
    for ref,w in zip(worker.reference,initial):ref.assign(w)
    outcomes=[]
    for lam in (1.,999999.):
        worker.load(initial)
        grads,values=worker._gradients(tf.ones((4,32,32,3)),tf.constant([0,1,2,3]),tf.constant([1,2]),
            'gradient_norm_objective','paper_absolute_gaussian',tf.constant(.01),tf.constant(lam))
        tf.keras.optimizers.SGD(.05,momentum=0.).apply_gradients(zip(grads,net.trainable_variables))
        outcomes.append(([g.numpy() for g in grads],[float(v) for v in values],worker.export()))
    for first,second in zip(outcomes[0],outcomes[1]):
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    assert outcomes[0][1][3]==pytest.approx(.0001,rel=1e-6)
    assert outcomes[0][1][6]>0


def test_readiness_cli_no_training_and_fixed_horizon():
    import json
    result=subprocess.run([sys.executable,'scripts/run_dirichlet_calibration.py','--stage','threat'],
        check=True,capture_output=True,text=True)
    payload=json.loads(result.stdout)
    assert not payload['trainingStarted'] and payload['rounds']==30 and payload['variants']==list('ABCD')
