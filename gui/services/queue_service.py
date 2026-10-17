from __future__ import annotations
import json
from pathlib import Path
from gui.state.experiment_state import from_persisted, to_persisted
from gui.state.queue_state import QueueEntry, QueueState, QueueStatus

class QueueService:
    """Persists the legacy list-of-configs format while accepting enriched entries."""
    def __init__(self,path: Path | str): self.path=Path(path)
    def load(self) -> QueueState:
        if not self.path.exists(): return QueueState()
        try: raw=json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as error: raise ValueError(f"Could not load queue: {error}") from error
        if not isinstance(raw,list): raise ValueError("Queue persistence must contain a JSON list.")
        entries=[]
        for item in raw:
            if not isinstance(item,dict): continue
            if "config" in item:
                status=QueueStatus(item.get("status","pending"))
                if status is QueueStatus.RUNNING:status=QueueStatus.STOPPED
                entries.append(QueueEntry(from_persisted(item["config"]),status,str(item.get("entryId","")) or __import__("uuid").uuid4().hex,str(item.get("error",""))))
            else: entries.append(QueueEntry(from_persisted(item)))
        return QueueState(entries)
    def save(self,state: QueueState) -> None:
        self.path.parent.mkdir(parents=True,exist_ok=True); temporary=self.path.with_suffix(".json.tmp")
        # Keep the established list-of-configurations format. Completed work is
        # represented by result manifests and was historically removed from
        # the persisted queue; stopped/failed entries reload as retryable work.
        payload=[dict(config=to_persisted(e.config),status=e.status.value,entryId=e.entry_id,error=e.error) if e.config.extra.get('iidCampaignId') or e.config.extra.get('noniidStudyId') else to_persisted(e.config) for e in state.entries if e.status is not QueueStatus.COMPLETED or e.config.extra.get('horizonStudyRevision')]
        temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); temporary.replace(self.path)

    @staticmethod
    def describe_recovery(state: QueueState, root: Path) -> None:
        from basil_core.protocol_checkpoint import recovery_plan
        for entry in state.entries:
            config=to_persisted(entry.config)
            if config.get('experimentProtocol')=='sequential_basil_dirichlet_v1':
                if config.get('horizonStudyRevision')=='dirichlet_prior_fixed_horizons_v1' and config.get('nRounds')==500:
                    from gui.services.staged_queue_service import dependency_state
                    try:
                        dependency=dependency_state(config)
                        if dependency['state']!='ready':
                            from basil_core.dirichlet_horizon_study import EVIDENCE
                            from gui.work_progress import completed_work
                            measured=json.loads(EVIDENCE.read_text());work=completed_work(config,0)
                            entry.completed_rounds=0;entry.completed_activations=0;entry.completed_optimizer_steps=0
                            entry.total_activations=work['totalActivations'];entry.total_optimizer_steps=work['totalOptimizerSteps']
                            entry.estimated_remaining_seconds=measured['runtimeSecondsByCondition'][config['conditionId']]/measured['rounds']*500
                            entry.execution_hint=('Waiting for ALL 100-round A/B/C/D runs, then automatic validation — W&B'
                                if dependency['state']=='waiting' else 'Waiting for automatic 100-round protocol/numerical validation — W&B')
                            continue
                    except (ValueError,OSError,KeyError) as error:
                        entry.execution_hint=f'Recovery blocked: {error}';entry.estimated_remaining_seconds=None
                        continue
                if config.get('horizonStudyRevision')=='dirichlet_prior_fixed_horizons_v1':
                    from basil_core.dirichlet_horizon_study import recovery_plan as dirichlet_recovery
                elif config.get('lossStudyRevision')=='dirichlet_prior_readiness_diagnostic_v1':
                    from basil_core.dirichlet_prior_study import recovery_plan as dirichlet_recovery
                else:
                    from basil_core.dirichlet_study import recovery_plan as dirichlet_recovery
                try:
                    plan=dirichlet_recovery(root/config['resultDirectory'],config)
                    entry.completed_rounds=plan['completedRounds']
                    if plan['mode']=='completed':entry.status=QueueStatus.COMPLETED
                    entry.execution_hint=(f"Fresh R0→R{config['nRounds']-1} — Dirichlet α=0.2 GPU"
                        if plan['mode']=='fresh' else 'Already completed' if plan['mode']=='completed'
                        else f"Exact activation-boundary resume: {plan['startActivation']} — GPU")
                    entry.estimated_remaining_seconds=None
                    if config.get('horizonStudyRevision')=='dirichlet_prior_fixed_horizons_v1':
                        from basil_core.dirichlet_horizon_study import EVIDENCE
                        measured=json.loads(EVIDENCE.read_text())
                        per_round=measured['runtimeSecondsByCondition'][config['conditionId']]/measured['rounds']
                        weights=config['optimizerStepsPerNode']
                        from gui.work_progress import completed_work
                        work=completed_work(config,plan['startActivation'])
                        entry.completed_activations=work['completedActivations'];entry.total_activations=work['totalActivations']
                        entry.completed_optimizer_steps=work['completedOptimizerSteps'];entry.total_optimizer_steps=work['totalOptimizerSteps']
                        full,partial=divmod(plan['startActivation'],10)
                        remaining=config['nRounds']-full-sum(weights[:partial])/sum(weights)
                        entry.estimated_remaining_seconds=max(0.,per_round*remaining)
                except (ValueError,OSError) as error:entry.execution_hint=f'Recovery blocked: {error}'
                continue
            if config.get('noniidStudyId'):
                from basil_core.noniid_verified_study import recovery_plan as noniid_recovery
                try:
                    plan=noniid_recovery(root/config['resultDirectory'],config)
                    entry.completed_rounds=plan['completedRounds']
                    if plan['mode']=='completed':entry.status=QueueStatus.COMPLETED
                    entry.execution_hint=('Fresh R0→R29 — GPU' if plan['mode']=='fresh' else
                        'Completed 30/30' if plan['mode']=='completed' else
                        f"Exact resume: activation {plan['startActivation']}, step {plan['step']}/50 — GPU")
                    entry.estimated_remaining_seconds=None
                except (ValueError,OSError) as error:
                    entry.execution_hint=f'Recovery blocked: {error}'
                continue
            if not config.get('iidCampaignId'):continue
            try:
                plan=recovery_plan(root/config['resultDirectory'],config)
                transition=None
                if plan['mode']=='fresh' and config.get('recoverySourceDirectory'):
                    from basil_core.iid_device_transition import source_for_transition
                    transition=source_for_transition(root,config)
                manifest=root/config['resultDirectory']/'run.json'
                recorded=json.loads(manifest.read_text()).get('completedRounds',0) if manifest.exists() else 0
                entry.completed_rounds=max(entry.completed_rounds,int(recorded),int(plan['completedRounds']))
                if plan['mode']=='completed':entry.status=QueueStatus.COMPLETED
                cursor=plan['startActivation']; saved=plan['savedActivations']
                entry.execution_hint={
                    'fresh':f"Fresh R0→R{config['nRounds']-1}", 'completed':'Already completed',
                    'checkpoint':f"Resume at activation {cursor} / {config['nRounds']*config['nNodes']}",
                    'verified_replay':f'Rebuild/verify {saved} saved activations, then continue',
                }[plan['mode']]
                if transition:
                    _,source_manifest,source_plan=transition
                    entry.completed_rounds=max(entry.completed_rounds,int(source_manifest.get('completedRounds',0)))
                    if source_plan['mode']=='completed':
                        entry.status=QueueStatus.COMPLETED;entry.execution_hint='Already completed on CPU'
                    elif source_plan['mode']=='checkpoint':entry.execution_hint=f"Restore CPU state at activation {source_plan['startActivation']}, then GPU"
                    else:entry.execution_hint=f"CPU rebuild/verify {source_plan['savedActivations']} activations, then GPU"
                if config.get('executionDevice')=='GPU':entry.execution_hint+=' — GPU'
                model=config.get('runtimeEstimateModel',{})
                total=model.get('sourceEbmSeconds' if config.get('localObjective')=='source_ebm' else 'ceSeconds')
                entry.estimated_remaining_seconds=(total*(1-cursor/(config['nRounds']*10))
                    if total is not None and config.get('executionDevice','CPU')=='CPU' else None)
            except (ValueError,OSError) as error:
                entry.execution_hint=f'Recovery blocked: {error}'
                entry.estimated_remaining_seconds=None
