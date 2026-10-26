#!/usr/bin/env python3
"""Verify every fixed replication pair and report all three seeds without selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def numerical_policy(runtime):
    """Fresh independent pairs may use different GPUs of the same tested model."""
    import copy
    value=copy.deepcopy(runtime)
    profile=value.get('profile',{})
    for key in ('gpu','gpu_uuid','output_root'):profile.pop(key,None)
    gpu=value.get('gpu')
    if gpu:
        for key in ('index','uuid'):gpu.pop(key,None)
    return value


def aggregate(rows):
    """Primary endpoint is fixed at the mean of round indices 20..29."""
    import numpy as np
    from basil_core.cart_validation import REPLICATION_SEEDS
    expected={(seed,letter) for seed in REPLICATION_SEEDS for letter in 'ABCD'}
    if len(rows)!=12 or {(r['seed'],r['condition']) for r in rows}!=expected:
        raise ValueError('Exactly all twelve seed/condition comparisons are required.')
    output=[]
    for letter in 'ABCD':
        measured=[]
        for seed in REPLICATION_SEEDS:
            row=next(r for r in rows if r['seed']==seed and r['condition']==letter)
            arrays=[np.asarray(row[k],np.float64) for k in ('controlMean','cartMean','controlWorst','cartWorst')]
            if any(a.shape!=(30,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1) for a in arrays):
                raise ValueError('Each seed requires thirty finite accuracy measurements in [0, 1].')
            control,cart,cworst,worst=arrays
            measured.append(dict(seed=seed,primaryDifferencePp=float(100*np.mean(cart[20:30]-control[20:30])),
                finalMeanDifferencePp=float(100*(cart[-1]-control[-1])),
                finalWorstDifferencePp=float(100*(worst[-1]-cworst[-1])),
                controlFinalMean=float(control[-1]),cartFinalMean=float(cart[-1])))
        differences=[r['primaryDifferencePp'] for r in measured]
        output.append(dict(condition=letter,seeds=measured,
            primaryMeanDifferencePp=float(np.mean(differences)),primarySampleStdPp=float(np.std(differences,ddof=1)),
            improvedSeeds=sum(v>0 for v in differences)))
    return output


def review(root,*,plot=False):
    from basil_core.cart_validation import REPLICATION_SEEDS
    from scripts.report_cart_train_validation import review as review_pair
    plan=json.loads((root/'plan.json').read_text())
    if (plan['revision']!='cart_replication_30r_three_seed_v1' or plan['seeds']!=list(REPLICATION_SEEDS) or
            plan['rounds']!=30 or plan['attackStartRoundIndex']!=20 or plan['totalArms']!=24 or
            plan['officialTestUsed'] is not False or
            plan['primaryEndpoint']!='mean_node_validation_accuracy_round_indices_20_to_29'):
        raise ValueError('Fixed replication plan differs.')
    jobs=plan['jobs']
    expected={(seed,pair) for seed in REPLICATION_SEEDS for pair in 'ABCD'}
    if len(jobs)!=12 or {(j['seed'],j['pair']) for j in jobs}!=expected:
        raise ValueError('Replication plan omitted or duplicated a comparison pair.')
    rows=[];sources=[];policies=[];data=[];seed_identities={}
    for job in jobs:
        pair_root=(root/f'seed-{job["seed"]}'/f'pair-{job["pair"]}').resolve()
        declared=Path(plan['output'])/f'seed-{job["seed"]}'/f'pair-{job["pair"]}'
        if Path(job['output'])!=declared or not pair_root.is_relative_to(root.resolve()):
            raise ValueError('Pair output differs from its declared location.')
        checked=review_pair(pair_root,plot=plot)
        if checked['seed']!=job['seed'] or checked['stage']!='replication_30r':raise ValueError('Pair seed/stage differs.')
        by_name={r['method']:r for r in checked['methods']}
        letter=job['pair'];control=by_name['prior_'+letter];cart=by_name['budgeted_'+letter]
        rows.append(dict(seed=job['seed'],condition=letter,controlMean=control['validationMean'],cartMean=cart['validationMean'],
            controlWorst=control['validationWorst'],cartWorst=cart['validationWorst']))
        run=json.loads((pair_root/('prior_'+letter)/'run.json').read_text())
        sources.append(run['implementationSha256']);policies.append(numerical_policy(run['runtime']));data.append(run['dataSha256'])
        identity=(run['config']['partitionHash'],run['config']['initialModelHash'])
        if job['seed'] in seed_identities and seed_identities[job['seed']]!=identity:
            raise ValueError('A/B/C/D pairs must share partition and initialization within each seed.')
        seed_identities[job['seed']]=identity
    if any(s!=sources[0] for s in sources):raise ValueError('Executed scientific source differs across pairs.')
    if any(p!=policies[0] for p in policies):raise ValueError('Pairs differ in software, GPU model or arithmetic policy.')
    if any(d!=data[0] for d in data):raise ValueError('The fixed TRAIN/validation split differs across seeds.')
    if any(len({identity[i] for identity in seed_identities.values()})!=3 for i in (0,1)):
        raise ValueError('Different seeds did not produce distinct initialization/partition identities.')
    result=dict(status='all_saved_pairs_verified',officialTestUsed=False,researchValid=False,
        primaryEndpoint=plan['primaryEndpoint'],conditions=aggregate(rows),
        interpretation='Three fixed seeds on a TRAIN holdout; descriptive differences, not a novelty or Byzantine robustness proof.')
    (root/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# CART fixed three-seed comparison','',result['interpretation'],'',
        'Primary endpoint: CART minus control mean-node validation accuracy, averaged over round indices 20–29.','',
        '| Condition | Mean difference (pp) | Sample SD (pp) | Positive seeds |',
        '|---|---:|---:|---:|']
    for row in result['conditions']:
        lines.append(f'| {row["condition"]} | {row["primaryMeanDifferencePp"]:+.3f} | {row["primarySampleStdPp"]:.3f} | {row["improvedSeeds"]}/3 |')
    (root/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--plot',action='store_true');args=parser.parse_args();root=args.directory.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):parser.error('Use an isolated local-test directory.')
    print(json.dumps(review(root,plot=args.plot),indent=2))


if __name__=='__main__':main()
