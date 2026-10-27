"""Fixed paired threats, same base learning recipe and explicitly new policy."""
import pytest


def test_paired_configuration_changes_only_the_declared_cart_mechanism():
    from basil_core.cart_coherent_study import configuration,validate_config,VARIANTS
    configs=[configuration('dirichlet',2025,.01,v) for v in VARIANTS]
    for c in configs:
        validate_config(c)
        assert c['nRounds']==3 and c['attackStartRound']==1
        assert c['localEpochs']==5 and c['snapshotSelection'] and c['useChannelNoise'] and c['attackHidden']
        assert c['snapshotAnchorMu']==0 and not c['logitCalibrationEnabled']
    common=('modelArchitecture','seed','partitionStrategy','dirichletAlpha','resolvedAttackerIds',
        'nRounds','batchSize','momentum','weightDecayCoefficient','ebmMode','channelNoiseSigmaAbsolute',
        'augmentation','learningRate','learningRateSchedule','evaluationDataOrigin','officialTestUsed')
    assert all({k:c[k] for k in common}=={k:configs[0][k] for k in common} for c in configs)
    assert len({c['runId'] for c in configs})==len(VARIANTS)
    for key,value in [('cartChannelCalibrationDraws',3),('nRounds',30),('officialTestUsed',True)]:
        with pytest.raises(ValueError):validate_config(dict(configs[-1],**{key:value}))


def test_clean_controls_are_explicit_and_not_silently_noisy():
    from basil_core.cart_coherent_study import configuration,validate_config
    cfg=configuration('iid',2026,0.,'coherent',case='clean')
    validate_config(cfg)
    assert not cfg['attackHidden'] and not cfg['useChannelNoise']
    assert cfg['ebmMode']=='none' and cfg['resolvedAttackerIds']==[]
    with pytest.raises(ValueError):configuration('iid',2026,.01,'coherent',case='clean')


def test_prior_is_explicit_and_paired_in_every_variant():
    from basil_core.cart_coherent_study import configuration,validate_config,VARIANTS
    for variant in VARIANTS:
        cfg=configuration('dirichlet',2025,.01,variant,training_objective='local_prior')
        validate_config(cfg)
        assert cfg['logitCalibrationEnabled'] and cfg['logitCalibrationTau']==1.
        assert cfg['logitCalibrationMethod']=='prior_adjustment'
        assert cfg['runId']!=configuration('dirichlet',2025,.01,variant)['runId']
        with pytest.raises(ValueError):validate_config(dict(cfg,logitCalibrationEnabled=False))


def test_all_paired_arms_archive_the_same_boundary_implementation():
    from basil_core.cart_coherent_study import configuration,VARIANTS
    from basil_core.protocol_checkpoint import checkpoint_implementation
    sources=[checkpoint_implementation(configuration('dirichlet',2025,.01,v)) for v in VARIANTS]
    assert 'basil_core/cart_boundary_worker.py' in sources[0]
    assert all(source==sources[0] for source in sources)


def test_clean_references_do_not_multiply_unrequested_cart_conditions():
    from argparse import Namespace
    from scripts.run_cart_coherent_pilot import configurations
    args=Namespace(partitions=['iid','dirichlet'],seeds=[2025,2026,2027],sigmas=[.01],
        variants=['none','coherent_commit_ce','coherent_commit_error'],cases=['joint','clean'],
        training_objectives=['local_prior'],stage='pilot',clean_controls_only=True)
    rows=configurations(args)
    assert len(rows)==24
    clean=[r for r in rows if r['coherentCase']=='clean']
    assert len(clean)==6 and all(r['cartVariant']=='none' and not r['cartEnabled'] for r in clean)
    assert len({r['runId'] for r in rows})==24
