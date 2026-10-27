"""Measured nonlinear class bounds and rollback in the isolated candidate."""
import pytest


def make_worker():
    import tensorflow as tf
    from basil_core.cart_feasible_worker import FeasibleCartWorker
    model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
        tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=31))])
    return FeasibleCartWorker(model)


def examples():
    import numpy as np
    return np.random.default_rng(19).uniform(0,1,(19,32,32,3)).astype(np.float32),np.arange(19,dtype=np.int32)%3


def options(round_id=0):
    return dict(seed=2025,round_id=round_id,node_id=0,epochs=5,batch_size=128,
        learning_rate=.01,ebm_mode='gradient_norm_objective',semantics='paper_absolute_gaussian',sigma=.01)


def test_first_reference_is_coherent_and_later_committed_steps_satisfy_actual_bounds():
    import numpy as np
    from basil_core.cart_feasible_worker import validate_memory
    from basil_core.cart_feasible_memory import within_envelope
    worker=make_worker();x,y=examples()
    first,record=worker.train(worker.export(),x,y,**options())
    assert not record['cartMemory']['guardEnabled']
    assert record['cartMemory']['calibrationDraws']==19
    assert record['steps']==5 and record['localClassCounts'][3:]==[0]*7
    state=worker.export_local_memory();validate_memory(state,seed=2025,cursor=1)
    ref=state['nodes']['0']
    assert ref['identity']['classes']==[0,1,2]
    # A damaged receiver input is repaired based on local memory, never on
    # configured malicious identity or evaluation data.
    damaged=[p.copy() for p in first];damaged[-1][0]+=20.
    _,second=worker.train(damaged,x,y,**options(1))
    memory=second['cartMemory']
    assert memory['guardEnabled'] and memory['inputBridge']['scale']<1.
    assert len(memory['stepGuards'])==5
    assert all(within_envelope(s['classLosses'],memory['activeBounds']) for s in memory['stepGuards'])
    assert all(0<=s['scale']<=1 for s in memory['stepGuards'])
    assert all(np.isfinite(v).all() for v in worker.export())


def test_interrupted_activation_does_not_admit_future_reference():
    import numpy as np
    from basil_core.research_protocol import params_hash
    worker=make_worker();x,y=examples()
    first,_=worker.train(worker.export(),x,y,**options())
    state=worker.export_local_memory()
    def metadata(value):
        return {k:(params_hash(v['referenceParams']),v['activation'],v['envelope']) for k,v in value['nodes'].items()}
    with pytest.raises(InterruptedError):
        worker.train(first,x,y,should_stop=lambda:True,**options(1))
    assert metadata(worker.export_local_memory())==metadata(state)
    worker.restore_local_memory(state)
    changed=x.copy();changed[0]=0
    with pytest.raises(ValueError,match='identity'):
        worker.train(first,changed,y,**options(1))


def test_guard_scales_momentum_consistently_with_accepted_step():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_feasible_worker import GuardedOptimizer
    class Receiver:
        model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1,use_bias=False)])
        active_bounds=[0.]
        step_guards=[]
        def export(self):return [v.numpy() for v in self.model.trainable_variables]
        def load(self,params):
            for v,p in zip(self.model.trainable_variables,params):v.assign(p)
        def measure_current(self):return np.array([float(self.model.trainable_variables[0][0,0])**2])
        def measure_params(self,params):self.load(params);return self.measure_current()
    receiver=Receiver();receiver.load([np.array([[0.]],np.float32)])
    opt=GuardedOptimizer(tf.keras.optimizers.SGD(.1,momentum=.9),receiver)
    opt.apply_gradients([(tf.ones((1,1)),receiver.model.trainable_variables[0])])
    assert receiver.step_guards[0]['scale']==0.
    np.testing.assert_array_equal(receiver.export()[0],[[0.]])
    np.testing.assert_array_equal(opt.inner.momentums[0].numpy(),[[0.]])


def test_checkpoint_codec_binds_reference_parameters_and_completed_cursor():
    import copy
    import json
    import numpy as np
    from basil_core.cart_feasible_worker import pack_memory,unpack_memory
    from basil_core.research_protocol import params_hash
    worker=make_worker();x,y=examples()
    worker.train(worker.export(),x,y,**options())
    state=worker.export_local_memory();params={}
    def pack(values):
        digest=params_hash(values);params[digest]=[v.copy() for v in values];return digest
    encoded,digest=pack_memory(state,seed=2025,cursor=1,pack_params=pack)
    assert len(json.dumps(encoded))<10000
    restored=unpack_memory(encoded,digest,seed=2025,cursor=1,unpack_params=lambda key:params[key])
    for a,b in zip(restored['nodes']['0']['referenceParams'],state['nodes']['0']['referenceParams']):
        np.testing.assert_array_equal(a,b)
    tampered=copy.deepcopy(encoded);tampered['nodes']['0']['sigma']=.02
    with pytest.raises(ValueError,match='hash'):unpack_memory(tampered,digest,seed=2025,cursor=1,unpack_params=lambda key:params[key])
    with pytest.raises(ValueError,match='completed'):pack_memory(state,seed=2025,cursor=2,pack_params=pack)
