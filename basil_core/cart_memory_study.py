"""Fixed, reduced-data CART implementation smoke tests, never production queues."""
from __future__ import annotations

from basil_core.cart_local_memory import REVISION


def configurations(partition_hash, initial_hash, device='GPU', profile='smoke'):
    from basil_core.dirichlet_study import configurations as baseline
    if profile not in {'smoke', 'clean_learning'}:
        raise ValueError('Unknown bounded CART profile.')
    output = []
    for name, letter, beta in [('A_beta0', 'A', 0.), ('A', 'A', 1.), ('B', 'B', 1.),
                                ('C', 'C', 1.), ('D', 'D', 1.)]:
        cfg = dict(baseline(partition_hash, initial_hash)[ord(letter)-ord('A')])
        cfg.update(protocolRevision=REVISION, lossStudyRevision=REVISION, cartRevision=REVISION,
            cartDiagnosticProfile=profile,
            noniidStudyId=f'{REVISION}_{profile}_3r',
            cartBeta=beta, cartProbePerClass=16, cartTemperature=1., cartRegressionEpsilon=1e-8,
            logitCalibrationEnabled=True, logitCalibrationMethod='prior_adjustment',
            logitCalibrationTau=1., logitCalibrationCountFloor=1,
            cartAlgorithm=REVISION, nRounds=3, researchValid=False, fullResearchReady=False,
            purpose='implementation_smoke_test', attackStartRound=1,
            diagnosticSamplesPerNode=1024, diagnosticTestSamplesPerClass=100,
            evaluationSamples=1000, executionDevice=device,
            runId=name, experimentName=name, displayName=f'{name}: bounded CART memory diagnostic',
            localObjective='cart_memory_prior_ebm_anchor' if letter=='D' else 'cart_memory_prior_anchor',
            resultDirectory=f'newResults/nonIID/{REVISION}_smoke/{name}',
            plotDirectory=f'newPlots/nonIID/{REVISION}_smoke/{name}')
        if profile == 'clean_learning':
            if letter != 'A':
                continue
            cfg.pop('diagnosticSamplesPerNode')
            cfg.pop('diagnosticTestSamplesPerClass')
            cfg.update(purpose='bounded_cart_clean_learning', attackStartRound=20,
                evaluationSamples=10000,
                resultDirectory=f'newResults/nonIID/{REVISION}_clean_learning/{name}',
                plotDirectory=f'newPlots/nonIID/{REVISION}_clean_learning/{name}')
        output.append(cfg)
    return output


def validate_config(config):
    from basil_core.research_protocol import validate_config as validate_base
    from basil_core.cart_local_memory import _sha
    if not all(_sha(config.get(k)) for k in ('partitionHash', 'initialModelHash')):
        raise ValueError('Actual partition and initialization SHA256 identities required.')
    if config.get('executionDevice') not in {'CPU', 'GPU'}:
        raise ValueError('Explicit CPU or GPU execution device required.')
    envelope = dict(config, lossStudyRevision='dirichlet_local_snapshot_anchor_v1',
        localObjective='source_ebm_snapshot_anchor' if config.get('ebmMode')=='gradient_norm_objective'
            else 'snapshot_anchor_only')
    validate_base(envelope)
    candidates = configurations(config['partitionHash'], config['initialModelHash'], config['executionDevice'],
        config.get('cartDiagnosticProfile', 'smoke'))
    expected = next((c for c in candidates if c['runId']==config.get('runId')), None)
    if expected is None or config != expected:
        raise ValueError('CART revision is restricted to its fixed three-round smoke profiles.')


def validate_memory_checkpoint(state, cursor, seed):
    from basil_core.cart_local_memory import LocalClassMemory
    memory = LocalClassMemory(seed, state)
    if set(memory.nodes) != {str(i) for i in range(min(cursor, 10))}:
        raise ValueError('CART checkpoint lost a visited receiver memory.')
    for node in memory.nodes.values():
        supported = {str(c) for c, count in enumerate(node['classCounts']) if count > 0}
        if set(node['classes']) != supported:
            raise ValueError('CART checkpoint lost supported class targets.')
        if any(record['activation'] >= cursor for record in node['classes'].values()):
            raise ValueError('CART checkpoint contains future teacher knowledge.')
    return memory.export()
