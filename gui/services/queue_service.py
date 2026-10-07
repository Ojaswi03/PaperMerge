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
        payload=[dict(config=to_persisted(e.config),status=e.status.value,entryId=e.entry_id,error=e.error) if e.config.extra.get('iidCampaignId') else to_persisted(e.config) for e in state.entries if e.status is not QueueStatus.COMPLETED]
        temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); temporary.replace(self.path)

    @staticmethod
    def describe_recovery(state: QueueState, root: Path) -> None:
        from basil_core.protocol_checkpoint import recovery_plan
        for entry in state.entries:
            config=to_persisted(entry.config)
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
                    'fresh':'Fresh R0→R99', 'completed':'Already completed',
                    'checkpoint':f'Resume at activation {cursor} / 1000',
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
