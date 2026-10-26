"""Reject unusable experiments before loading TensorFlow or claiming a GPU."""
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('kind',['float','negative','out_of_range','imbalanced'])
def test_split_rejects_invalid_original_train_labels(kind):
    import numpy as np
    from basil_core.cart_validation import split_training
    labels=np.repeat(np.arange(10,dtype=np.int32),5000)
    if kind=='float':labels=labels.astype(np.float32)
    elif kind=='negative':labels[-1]=-1
    elif kind=='out_of_range':labels[-1]=10
    else:labels[-1]=0
    with pytest.raises(ValueError,match='TRAIN label'):
        split_training(labels)


@pytest.mark.parametrize('value',['nan','inf',-.001,1.001])
def test_training_cache_checks_pixels_through_the_final_block(value):
    import numpy as np
    from basil_core.cart_validation import validate_training_arrays
    class Images:
        shape=(50000,32,32,3)
        dtype=np.dtype('float32')
        def __len__(self):return 50000
        def __getitem__(self,section):
            block=np.zeros((min(section.stop,50000)-section.start,32,32,3),np.float32)
            if section.stop>=50000:block[-1,-1,-1,-1]=float(value)
            return block
    with pytest.raises(ValueError,match='finite normalized'):
        validate_training_arrays(Images(),np.repeat(np.arange(10,dtype=np.int32),5000))


@pytest.mark.parametrize('shape,dtype', [((50000,32,32),'float32'),((50000,32,32,3),'uint8')])
def test_cache_image_format_is_explicit(shape,dtype):
    import numpy as np
    from basil_core.cart_validation import validate_training_arrays
    with pytest.raises(ValueError,match='TRAIN images'):
        validate_training_arrays(SimpleNamespace(shape=shape,dtype=np.dtype(dtype)),np.empty(50000,np.int32))


def test_missing_cache_blocks_execution_before_environment_or_output_changes(tmp_path,monkeypatch):
    import scripts.run_cart_train_validation as runner
    import basil_core.iid_runtime as runtime
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    def must_not_initialize(*_):raise AssertionError('GPU environment initialized before input validation')
    monkeypatch.setattr(runtime,'prepare_environment',must_not_initialize)
    output=tmp_path/'.papermerge/local-tests/fresh'
    with pytest.raises(ValueError,match='Missing TRAIN cache'):
        runner.execute(SimpleNamespace(output=output,recover=False,device='GPU',stage='budgeted_clean'))
    assert not output.exists()


def test_preflight_catches_source_mismatch_without_replacing_recovery_evidence(tmp_path,monkeypatch):
    import numpy as np
    import scripts.run_cart_train_validation as runner
    import basil_core.cart_validation as study
    import basil_core.protocol_checkpoint as checkpoints
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    cache=tmp_path/'experiments/cache/cifar10';cache.mkdir(parents=True)
    # Exercise preflight ordering and identity with tiny stand-ins; independent
    # tests above cover the complete 50000-image schema and pixel validation.
    np.save(cache/'x_train_float32.npy',np.zeros((20,2),np.float32))
    np.save(cache/'y_train_int32.npy',np.repeat(np.arange(10,dtype=np.int32),2))
    monkeypatch.setattr(study,'validate_training_arrays',lambda *_:None)
    monkeypatch.setattr(study,'split_training',lambda _: (np.arange(10),np.arange(10,20)))
    monkeypatch.setattr(study,'partition',lambda _,**kwargs: ('a'*64,[np.array([i]) for i in range(10)]))
    monkeypatch.setattr(checkpoints,'checkpoint_implementation',lambda _: {'worker.py':'b'*64})
    output=tmp_path/'.papermerge/local-tests/fresh';output.mkdir(parents=True)
    identity=output/'identity.json';identity.write_text(json.dumps(dict(implementationSha256={'worker.py':'c'*64})))
    previous=identity.read_bytes()
    with pytest.raises(ValueError,match='identity'):
        runner.preflight(SimpleNamespace(output=output,recover=True,device='GPU',stage='budgeted_clean'))
    assert identity.read_bytes()==previous and list(output.iterdir())==[identity]


def test_preflight_cli_is_mutually_exclusive_with_execution():
    import subprocess
    result=subprocess.run([sys.executable,'scripts/run_cart_train_validation.py','--execute','--preflight'],
        capture_output=True,text=True)
    assert result.returncode==2 and 'not allowed with argument' in result.stderr


@pytest.mark.parametrize('condition',['B','C','D'])
def test_stress_pair_changes_only_predeclared_retention_and_run_identity(condition):
    from basil_core.cart_validation import configurations
    configs=configurations('a'*64,'b'*64,'CPU','noisy_attack')
    control=next(c for c in configs if c['runId']=='prior_'+condition)
    candidate=next(c for c in configs if c['runId']=='budgeted_'+condition)
    changes={k for k in control.keys()|candidate.keys() if control.get(k)!=candidate.get(k)}
    assert changes=={'cartBeta','cartMemoryMode','cartCorrectionRatioCap','runId',
        'experimentName','displayName','resultDirectory','plotDirectory'}
    assert (control['localEpochs'],control['batchSize'],control['snapshotAnchorMu'])==(5,512,.01)
    assert control['seed']==candidate['seed']==2025
    assert control['partitionHash']==candidate['partitionHash']
    assert control['initialModelHash']==candidate['initialModelHash']
