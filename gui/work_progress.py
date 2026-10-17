"""Exact completed work units; file sizes and parameter counts are not clocks."""
from __future__ import annotations


def completed_work(config, activations):
    nodes=int(config.get('nNodes',10));rounds=int(config.get('nRounds',100))
    total=nodes*rounds
    if type(activations) is not int or not 0<=activations<=total:
        raise ValueError('Completed activation cursor is outside the experiment.')
    result=dict(completedActivations=activations,totalActivations=total,
        progress=activations/total)
    weights=config.get('optimizerStepsPerNode')
    if weights is not None:
        if len(weights)!=nodes or any(type(w) is not int or w<=0 for w in weights):
            raise ValueError('Positive optimizer-step counts for every node required.')
        full,partial=divmod(activations,nodes)
        done=full*sum(weights)+sum(weights[:partial]);target=rounds*sum(weights)
        result.update(completedOptimizerSteps=done,totalOptimizerSteps=target,progress=done/target)
    return result


def queue_work(entries,state=None):
    """Totals for the loaded queue, including finished entries and resumed work."""
    from gui.state.experiment_state import to_persisted
    totals=dict(completedRuns=0,totalRuns=len(entries),completedRounds=0,totalRounds=0,
        completedActivations=0,totalActivations=0,completedOptimizerSteps=0,totalOptimizerSteps=0)
    all_steps=True
    for entry in entries:
        config=to_persisted(entry.config)
        target=entry.config.rounds*entry.config.node_count
        completed=entry.status.value=='completed'
        active=entry.status.value=='running' and state is not None
        cursor=target if completed else state.completed_activations if active else entry.completed_activations
        work=completed_work(config,cursor)
        rounds=entry.config.rounds if completed else state.completed_rounds if active else entry.completed_rounds
        totals['completedRuns']+=int(completed)
        totals['completedRounds']+=rounds;totals['totalRounds']+=entry.config.rounds
        for key in ('completedActivations','totalActivations'):totals[key]+=work[key]
        all_steps=all_steps and 'totalOptimizerSteps' in work
        for key in ('completedOptimizerSteps','totalOptimizerSteps'):totals[key]+=work.get(key,0)
    if not all_steps:
        totals['completedOptimizerSteps']=totals['totalOptimizerSteps']=0
    total=totals['totalOptimizerSteps'] or totals['totalActivations']
    done=totals['completedOptimizerSteps'] if totals['totalOptimizerSteps'] else totals['completedActivations']
    totals['progress']=done/total if total else 0.
    return totals


def queue_eta_seconds(entries,state):
    """Active measured rate plus per-condition estimates for pending runs."""
    remaining=0.
    for entry in entries:
        if entry.status.value not in {'pending','running'}:continue
        estimate=entry.estimated_remaining_seconds
        if entry.status.value=='running':
            measured=state.experiment_eta_seconds
            estimate=measured if measured is not None else (
                max(0.,estimate-max(0.,state.elapsed_seconds-state.elapsed_offset)) if estimate is not None else None)
        if estimate is None:return None
        remaining+=estimate
    return remaining
