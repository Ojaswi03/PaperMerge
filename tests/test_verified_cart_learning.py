"""Bounded learning launcher contracts; tests never start a learning run."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_verified_cart_learning as study


def test_fixed_clean_full_data_scope():
    c = study.configuration()
    assert (c['rounds'], c['nodes'], c['samplesPerNode'], c['localEpochs'], c['batchSize']) == (1, 10, 5000, 5, 512)
    assert (c['stepsPerActivation'], c['finalBatch'], c['snapshotAnchorMu']) == (50, 392, .01)
    assert c['attackerIds'] == [] and c['helperFaults'] == {}
    assert c['sigma_e'] == 0 and not c['ebm'] and not c['wandb'] and not c['researchValid']
    assert c['informationModel'] == 'replicated_train_trusted_coordinator'
    assert study.OUTPUT.is_relative_to(study.ROOT/'newResults/nonIID/cart_verified')
    assert study.PLOTS.is_relative_to(study.ROOT/'newPlots/nonIID/cart_verified')


def test_default_dry_run_and_lazy_imports():
    code = ('import sys; from scripts import run_verified_cart_learning as s; '
            'assert not ({"tensorflow","numpy","matplotlib","wandb"}&sys.modules.keys()); s.main()')
    result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['trainingRunsStarted'] == 0


def test_commit_pointer_preserved_on_failed_checkpoint(tmp_path):
    runner = SimpleNamespace(round=0, node=0, step=0, records=[],
        save=lambda p: p.write_text('{"test":true}'))
    state = dict(status='running')
    study.commit(runner, state, tmp_path)
    original = (tmp_path/'progress.json').read_bytes()
    def fail(path): raise OSError('injected failure')
    runner.save = fail
    with pytest.raises(OSError): study.commit(runner, state, tmp_path)
    assert (tmp_path/'progress.json').read_bytes() == original
    assert state['checkpoint'] == 'checkpoint_0000.json'


def test_uncommitted_checkpoint_not_overwritten(tmp_path):
    runner = SimpleNamespace(round=0, node=0, step=0, records=[])
    target = tmp_path/'checkpoint_0000.json'; target.write_text('evidence')
    with pytest.raises(FileExistsError, match='preserve'): study.commit(runner, {}, tmp_path)
    assert target.read_text() == 'evidence'


def test_current_gpu_proof_required(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    proof = tmp_path/'gui/worker_state/CART_RESOURCE_LIMITED_RECOVERY_GPU.json'
    proof.parent.mkdir(parents=True); proof.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match='proof required'): study.require_proof()


def test_evaluation_does_not_change_parameters():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    model = tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10,
            kernel_initializer='zeros', bias_initializer='zeros')])
    params = [v.numpy() for v in model.trainable_variables]
    x = np.zeros((10,32,32,3), np.float32); y = np.arange(10, dtype=np.int32)
    result = study.evaluate(model, params, x, y, node=0, epoch=0)
    assert result['accuracy'] == .1 and sum(map(sum, result['confusion'])) == 10
    assert params_hash(params) == params_hash([v.numpy() for v in model.trainable_variables])
