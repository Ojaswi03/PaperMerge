#!/usr/bin/env python3
"""Prepare four fresh, idle GPU configurations. Never start a research worker."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Canonical initialization is intentionally CPU-based, even for GPU training.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

PARTITION_HASH = 'db79edd6b35d709b0a40fcc9ca5b5253e1da52a414cfba30a87de6aff4339b29'
INITIAL_HASH = '7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df'
CONFIG_ROOT = ROOT / 'gui/configs/IID/sequential_basil_500r/gpu/fresh'
QUEUE_PRESET = ROOT / 'gui/queues/iid_abcd_500r_gpu_fresh.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare():
    import numpy as np
    from basil_core.iid_campaign import campaign_configs, verify_campaign
    from basil_core.iid_study import partition_audit
    from basil_core.research_protocol import build_model, params_hash
    from gui.state.experiment_state import from_persisted, to_persisted, validate_experiment
    from gui.services.queue_service import QueueService
    from gui.state import QueueStatus

    if (ROOT/'gui/.queue_active.json').exists():
        raise ValueError('Close/stop the active queue before replacing its configuration.')
    cache = ROOT/'experiments/cache/cifar10'
    train = np.load(cache/'y_train_int32.npy', mmap_mode='r').reshape(-1)
    test = np.load(cache/'y_test_int32.npy', mmap_mode='r').reshape(-1)
    audit, _ = partition_audit(train, test, seed=2025)
    model = build_model('basil_paper_cnn', 2025)
    initial = [v.numpy() for v in model.trainable_variables]
    if audit['partitionHash'] != PARTITION_HASH or params_hash(initial) != INITIAL_HASH:
        raise ValueError('Canonical partition/initialization changed; preparation stopped.')
    if sum(p.size for p in initial) != 117706:
        raise ValueError('CNN parameter count changed.')
    configs = campaign_configs(PARTITION_HASH, INITIAL_HASH, device='GPU', rounds=500)
    for config in configs:
        config.update({flag: False for flag in ('attackGaussian','attackSignFlip','attackModelPoison',
            'attackScaling','attackAlie','attackIpm','attackNoiseAmp')})
        if validate_experiment(from_persisted(config)) or to_persisted(from_persisted(config)) != config:
            raise ValueError('Configuration failed GUI validation or JSON round-trip.')
        for key in ('resultDirectory', 'plotDirectory'):
            output = ROOT/config[key]
            if output.exists() and any(output.iterdir()):
                raise ValueError(f'Will not overwrite existing 500-round evidence: {output}')
    differences = verify_campaign(configs)
    for config in configs:
        write_json(CONFIG_ROOT/f"{config['experimentName']}.json", config)
        for key in ('resultDirectory','plotDirectory'):
            (ROOT/config[key]).mkdir(parents=True, exist_ok=True)
    for kind in ('newResults','newPlots'):
        (ROOT/kind/'nonIID').mkdir(parents=True, exist_ok=True)
    for key in ('comparisonResultsDirectory','comparisonPlotsDirectory'):
        (ROOT/configs[0][key]).mkdir(parents=True,exist_ok=True)
    write_json(QUEUE_PRESET, configs)
    write_json(ROOT/'gui/queue_state.json', configs)
    queue = QueueService(QUEUE_PRESET).load()
    QueueService.describe_recovery(queue, ROOT)
    if len(queue.entries) != 4 or any(entry.status is not QueueStatus.PENDING or entry.completed_rounds for entry in queue.entries):
        raise ValueError('Queue must have four fresh pending entries.')
    evidence = dict(status='prepared_idle', researchRunsStarted=0, rounds=500,
        queuePreset=str(QUEUE_PRESET.relative_to(ROOT)), conditionOrder=list('ABCD'),
        partitionAudit=audit, initialModelHash=INITIAL_HASH, parameterCount=117706,
        differences=differences, trainingStarted=False)
    # Preparation evidence is configuration metadata, not a research result.
    write_json(ROOT/'gui/queues/iid_abcd_500r_preparation.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    prepare()
