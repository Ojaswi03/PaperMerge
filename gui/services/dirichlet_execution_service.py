"""Serial GPU local-data Dirichlet runner; never uses CART helper services."""
from basil_core.dirichlet_study import ROOT,gui_approved,recovery_plan
from .iid_execution_service import IidExecutionHandle


class DirichletExecutionHandle(IidExecutionHandle):
    def __init__(self,config,event_sink):
        if config.get('lossStudyRevision')=='dirichlet_prior_readiness_diagnostic_v1':
            from basil_core.dirichlet_prior_study import gui_approved as prior_approved,recovery_plan as prior_recovery
            super().__init__(config,event_sink,_approved=prior_approved,
                _plan=prior_recovery(ROOT/config['resultDirectory'],config),
                _script=ROOT/'scripts/run_dirichlet_prior_condition.py',
                _work_root=ROOT/'gui/worker_state/dirichlet_prior')
            return
        super().__init__(config,event_sink,_approved=gui_approved,
            _plan=recovery_plan(ROOT/config['resultDirectory'],config),
            _script=ROOT/'scripts/run_dirichlet_condition.py',
            _work_root=ROOT/'gui/worker_state/dirichlet')
