"""GUI adapter for the explicitly configured serial verified CART worker."""
from basil_core.noniid_verified_study import ROOT, gui_approved, recovery_plan
from .iid_execution_service import IidExecutionHandle
from .execution_service import ExecutionEvent


class NoniidExecutionHandle(IidExecutionHandle):
    def __init__(self,config,event_sink):
        super().__init__(config,event_sink,_approved=gui_approved,
            _plan=recovery_plan(ROOT/config['resultDirectory'],config),
            _script=ROOT/'scripts/run_noniid_verified_condition.py',
            _work_root=ROOT/'gui/worker_state/noniid_verified')

    def _dispatch(self,payload):
        if payload.get('event')=='epoch_complete':
            rounds=self.config['nRounds']
            position=payload['round']+(payload['node']+payload['epoch']/5)/10
            self.sink(ExecutionEvent('progress_updated',dict(round=payload['round']+1,progress=position/rounds)))
            self.sink(ExecutionEvent('node_accuracy_updated',dict(node=payload['node'],
                accuracy=payload['accuracy'],round_position=position)))
            self.sink(ExecutionEvent('status_updated',dict(message=
                f"{self.config['conditionId']} — round {payload['round']+1}/{rounds}, node {payload['node']}, epoch {payload['epoch']}/5")))
        else:
            super()._dispatch(payload)
