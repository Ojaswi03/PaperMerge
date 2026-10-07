"""Fixed four-condition IID study; configuration preparation never trains."""
from pathlib import Path

CAMPAIGN_ID = 'iid_abcd_100r_v1'
ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'gui/configs/IID/sequential_basil_100r'
QUEUE_PATH = ROOT / 'gui/queues/iid_abcd_100r.json'
COMPARISON = 'ABCD_100r_comparison'
CONDITIONS = {
    'A': ('A_iid_basil_clean_no_attack_ce_100r', 'A_clean_no_attack_ce_100r', False, False, 'cross_entropy', 'Clean / No attack / No noise / CE'),
    'B': ('B_iid_basil_attack_clean_ce_100r', 'B_attack_clean_ce_100r', True, False, 'cross_entropy', 'Attack / Clean channel / CE'),
    'C': ('C_iid_basil_attack_noise_ce_100r', 'C_attack_noise_ce_100r', True, True, 'cross_entropy', 'Attack / Noise sigma 0.010 / CE'),
    'D': ('D_iid_basil_attack_noise_ebm_100r', 'D_attack_noise_ebm_100r', True, True, 'source_ebm', 'Attack / Noise sigma 0.010 / EBM'),
}
METADATA_KEYS = {'experimentName','runId','conditionId','displayName','resultDirectory','plotDirectory','recoverySourceDirectory'}
CONTRAST_FIELDS = {
    'A/B': {'attackerCount','actualAttackerCount','resolvedAttackerIds','attackerIds','attackHidden'},
    'B/C': {'useChannelNoise','channelNoiseSigmaAbsolute'},
    'C/D': {'localObjective','ebmMode'},
}


def campaign_configs(partition_hash, initial_hash, *, device='CPU', continue_existing=False):
    from basil_core.iid_runtime import execution_device
    execution_device({'executionDevice':device})
    if continue_existing and device!='GPU':raise ValueError('Device transition is only for the explicit GPU queue.')
    from basil_core.iid_study import paired_configs
    common=paired_configs(.01,rounds=100)[0]
    common.update(iidCampaignId=CAMPAIGN_ID,roundEvaluationMetrics=True,approach='basil',nonIID=False,
        useBasil=True,attackHiddenStart=20,purpose='iid_abcd_research',executionStartMode='fresh',
        expectedPartitionHash=partition_hash,expectedInitialModelHash=initial_hash,
        sigmaApproval='a_priori_calibration_0.010',comparisonResultsDirectory=f'newResults/IID/{COMPARISON}',
        comparisonPlotsDirectory=f'newPlots/IID/{COMPARISON}')
    configs=[]
    for letter,(name,folder,attack,noise,objective,label) in CONDITIONS.items():
        ids=[0,1,5,7] if attack else []
        configs.append(dict(common,conditionId=letter,experimentName=name,runId=name,
            displayName=f'{letter} — {label} — Fresh R0→R99',
            resultDirectory=f'newResults/IID/{folder}',plotDirectory=f'newPlots/IID/{folder}',
            attackerCount=len(ids),actualAttackerCount=len(ids),resolvedAttackerIds=ids,
            attackerIds=','.join(map(str,ids)),attackHidden=attack,useChannelNoise=noise,
            channelNoiseSigmaAbsolute=.01 if noise else 0.,localObjective=objective,
            ebmMode='gradient_norm_objective' if objective=='source_ebm' else 'none'))
    if device=='GPU':
        for config in configs:
            if continue_existing:
                config.update(recoverySourceDirectory=config['resultDirectory'],deviceTransition='cpu_to_gpu_verified_state')
            config.update(executionDevice='GPU',executionMode='deterministic_single_process_gpu')
            for key in ('experimentName','runId','resultDirectory','plotDirectory',
                        'comparisonResultsDirectory','comparisonPlotsDirectory'):
                config[key]+='_gpu'
            config['displayName']+=' — GPU'
            if continue_existing:config['displayName']=config['displayName'].replace('Fresh R0→R99','Remaining rounds')
    verify_campaign(configs)
    return configs


def validate_campaign_condition(config):
    if config.get('iidCampaignId')!=CAMPAIGN_ID:raise ValueError('Unknown IID campaign identifier.')
    letter=config.get('conditionId')
    if letter not in CONDITIONS:raise ValueError('Select IID condition A, B, C or D.')
    name,folder,attack,noise,objective,_=CONDITIONS[letter]
    from basil_core.iid_runtime import execution_device
    device=execution_device(config)
    if config.get('recoverySourceDirectory'):
        if device!='GPU' or config['recoverySourceDirectory']!=f'newResults/IID/{folder}' or config.get('deviceTransition')!='cpu_to_gpu_verified_state':
            raise ValueError('Only the matching CPU condition can supply a verified GPU transition.')
    suffix='_gpu' if device=='GPU' else ''
    expected=dict(nRounds=100,executionStartMode='fresh',roundEvaluationMetrics=True,approach='basil',nonIID=False,
        researchValid=True,executionMode=f'deterministic_single_process_{device.lower()}',seed=2025,
        experimentName=name+suffix,runId=name+suffix,useBasil=True,actualAttackerCount=4 if attack else 0,
        attackerCount=4 if attack else 0,resolvedAttackerIds=[0,1,5,7] if attack else [],
        attackerIds='0,1,5,7' if attack else '',attackHidden=attack,attackHiddenStart=20,
        useChannelNoise=noise,channelNoiseSigmaAbsolute=.01 if noise else 0.,localObjective=objective,
        resultDirectory=f'newResults/IID/{folder}{suffix}',plotDirectory=f'newPlots/IID/{folder}{suffix}',
        comparisonResultsDirectory=f'newResults/IID/{COMPARISON}{suffix}',comparisonPlotsDirectory=f'newPlots/IID/{COMPARISON}{suffix}')
    for field,value in expected.items():
        if config.get(field)!=value:raise ValueError(f'Condition {letter}: {field} must be {value!r}.')
    for field in ('expectedPartitionHash','expectedInitialModelHash'):
        value=config.get(field,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError(f'{field} requires the verified SHA-256.')


def verify_campaign(configs):
    from basil_core.research_protocol import validate_config
    if [c.get('conditionId') for c in configs]!=list(CONDITIONS):raise ValueError('Queue order must be A, B, C, D.')
    for config in configs:validate_config(config)
    differences={}
    for label,(left,right) in zip(CONTRAST_FIELDS,((0,1),(1,2),(2,3))):
        a,b=configs[left],configs[right]
        diff={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        if diff-METADATA_KEYS!=CONTRAST_FIELDS[label]:raise ValueError(f'Unexpected scientific differences for {label}: {sorted(diff-METADATA_KEYS)}')
        differences[label]=sorted(diff-METADATA_KEYS)
    return differences


def gui_approved(config):
    if config.get('iidCampaignId')!=CAMPAIGN_ID:return False
    from basil_core.research_protocol import validate_config
    try:validate_config(config)
    except (ValueError,TypeError,KeyError):return False
    return True
