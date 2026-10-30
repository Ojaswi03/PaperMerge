"""Allowlisted read-only campaign projection for the shareable browser monitor."""
from datetime import datetime,timezone
import math


def finite(value):
    return value if isinstance(value,(int,float)) and math.isfinite(value) else None


def public_snapshot(directory):
    from basil_core.sharp_monitor import snapshot
    state=snapshot(directory)
    if state['status']=='WAITING_FOR_PLAN':raise ValueError('No campaign plan exists yet.')
    public=dict(schema=1,generatedAt=datetime.now(timezone.utc).isoformat(),status=state['status'],
        total=state['total'],completed=state['completed'],failed=state['failed'],pending=state['pending'],
        completedActivations=state['completedActivations'],totalActivations=state['totalActivations'],
        elapsedSeconds=finite(state['elapsed']),etaSeconds=finite(state['eta']),
        activationsPerSecond=finite(state.get('activationsPerSecond')),etaMode=state.get('etaMode'),
        etaLowSeconds=finite(state['etaLow']),etaHighSeconds=finite(state['etaHigh']),
        untimedJobs=state['untimedJobs'],runs=[])
    for row in state['entries']:
        cfg=row['config']
        from basil_core.cart_display_labels import comparison_label
        public['runs'].append(dict(gpu=row['gpu'],status=row['status'],
            **{key:cfg.get(key) for key in ('dataset','partition','seed','sigma','condition','nodes','rounds','localSteps','attackStart')},
            completedRounds=row['rounds'],completedActivations=row['cursor'],totalActivations=row['total'],
            accuracy=finite(row['mean']),elapsedSeconds=finite(row['elapsed']),etaSeconds=finite(row['eta']),
            etaSource=row.get('etaSource'),phase=row.get('phase')))
        label=comparison_label(cfg,include_environment=True)
        if label:public['runs'][-1]['condition']=label
    return public
