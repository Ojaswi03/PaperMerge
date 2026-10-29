import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from tests.test_cart_v2_resilient import queue,children


def test_revision_worker_failure_isolation_and_explicit_retry(queue,monkeypatch):
    old,args,plan=queue
    import scripts.run_cart_revision_study as module
    plan.update(revision='test',reference=None);args.ntfy_topic=None
    monkeypatch.setattr(module,'prepared_for_execution',old.prepare)
    monkeypatch.setattr(module,'check_source',lambda _:None)
    plotted=[];monkeypatch.setitem(sys.modules,'reporting.cart_revision',SimpleNamespace(generate=lambda *a,**k:plotted.append(a)))
    started,live=children(monkeypatch,module,args.output,bad=(0,4))
    assert module.execute(args)==2 and sorted(started)==list(range(8)) and not plotted
    assert all(p.poll() is not None for p in live)
    status=json.loads((args.output/'campaign.json').read_text());assert status['completed']==6 and status['failed']==2
    started,_=children(monkeypatch,module,args.output,bad=())
    assert module.execute(args)==0 and sorted(started)==[0,4] and len(plotted)==1
    assert json.loads((args.output/'failed_runs.json').read_text())==[]


def test_default_matrix_is_six_cifar_cart_cases_and_help_stays_lazy(tmp_path):
    import subprocess
    runner=Path(__file__).resolve().parents[1]/'scripts/run_cart_revision_study.py'
    command=[sys.executable,str(runner),'--output',str(tmp_path)]
    default=json.loads(subprocess.check_output(command,text=True))
    assert default['configs']==6 and default['nodes']==100 and default['rounds']==100
    full=json.loads(subprocess.check_output(command+['--datasets','mnist','cifar10','--noise-levels','3'],text=True))
    assert full['configs']==36 and not full['trainingStarted']
    code=(f'import runpy,sys;sys.argv=[{str(runner)!r},"--help"];'
        '\ntry:runpy.run_path(sys.argv[0],run_name="__main__")'
        '\nexcept SystemExit as e:assert e.code==0'
        '\nassert not {"numpy","tensorflow","matplotlib"}.intersection(sys.modules)')
    result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_resume_rejects_new_horizon_before_touching_outputs(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,'scripts/run_cart_revision_study.py','--execute','--resume',
        '--output',str(tmp_path),'--rounds','35'],capture_output=True,text=True)
    assert result.returncode==2 and 'frozen plan' in result.stderr and not list(tmp_path.iterdir())


def test_chart_failure_preserves_completed_training_and_no_worker_is_retried(queue,monkeypatch):
    old,args,plan=queue
    import scripts.run_cart_revision_study as module
    plan.update(revision='test',reference=None);args.ntfy_topic=None
    monkeypatch.setattr(module,'prepared_for_execution',old.prepare);monkeypatch.setattr(module,'check_source',lambda _:None)
    def bad_chart(*a,**k):raise ValueError('fixture chart failure')
    monkeypatch.setitem(sys.modules,'reporting.cart_revision',SimpleNamespace(generate=bad_chart))
    started,live=children(monkeypatch,module,args.output,bad=())
    assert module.execute(args)==3 and len(started)==8
    row=json.loads((args.output/'campaign.json').read_text())
    assert row['completed']==8 and row['failed']==0 and row['status']=='COMPLETED_REPORT_FAILED'
    assert json.loads((args.output/'report_status.json').read_text())['trainingComplete']


def test_compact_audit_preserves_selected_class_evidence_and_all_candidate_scores():
    from scripts.cart_revision_worker import compact_record
    record=dict(selection=dict(selectedSenders=[3],continuity=dict(actualRoute='neighbor',candidates=[
        dict(kind='neighbor',senders=[3],score=.3,fitSkill=[.5],checkSkill=[.6]),
        dict(kind='pool',senders=[1,2],score=.1,fitSkill=[.2],checkSkill=[.2])])),training=dict(optimizerSteps=5))
    compact_record(record);candidates=record['selection']['continuity']['candidates']
    assert candidates[0]['fitSkill']==[.5] and candidates[0]['checkSkill']==[.6]
    assert candidates[1]==dict(kind='pool',senders=[1,2],score=.1)
    assert record['training']['optimizerSteps']==5
