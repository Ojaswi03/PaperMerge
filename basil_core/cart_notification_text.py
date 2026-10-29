"""Short, spaced phone notifications; detailed errors stay in local logs."""
from pathlib import Path


def event(kind,run,*,completed=None,total=None,failed=0,nodes=None,rounds=None,elapsed=None,commit=None,commands_url=None):
    name=Path(run).name if run else 'CART'
    titles=dict(complete='CART run complete',failed='CART tests failed',interrupted='CART tests stopped',
        worker_failed='CART: one configuration failed',partial='CART run finished with failures',
        report_failed='CART training complete; plots need attention',ready='CART ready for HPC',test='PaperMerge notification test')
    if kind not in titles:raise ValueError('Unknown notification event.')
    sections=[]
    if kind=='test':return titles[kind],'Connection successful.\n\nCART completion and failure alerts will appear here.'
    if completed is not None and total is not None:sections.append(f'Completed: {completed} / {total}\nFailed: {failed}')
    if nodes is not None and rounds is not None:sections.append(f'Clients: {nodes}\nRounds per configuration: {rounds}')
    if elapsed:sections.append('Elapsed: '+str(elapsed))
    messages=dict(complete='Line graphs, bar charts and averages are ready.',
        partial='Other workers continued. Completed results are saved.\nRetry the failed configurations when ready.',
        worker_failed='Other configurations are continuing.\nDetails: failed_runs.json and worker logs.',
        report_failed='Training results are saved.\nDetails: report_status.json. No retraining is needed.',
        interrupted='The run was stopped.\nCommitted checkpoints are preserved.',
        failed='Checkpoints are preserved.\nDetails: managed_status.json and supervisor.log.',
        ready='Code is pushed and the documented pilot checks passed.\nNext: pull, preflight, verify GPUs, then launch.')
    if commit:sections.append('Commit: '+commit[:12])
    sections.append(messages[kind])
    if commands_url:sections.append('Commands:\n'+commands_url)
    sections.append('Run: '+name)
    return titles[kind],'\n\n'.join(sections)
