"""Monitor screen refresh and completion are read-only operational behavior."""
import copy
import io
import json
import os
import subprocess
import sys

import pytest

from basil_core.terminal import dashboard


def snapshot(status='RUNNING',count=4):
    entries=[]
    for i in range(count):
        complete=status=='COMPLETED'
        entries.append(dict(name=f'condition_{i}',condition='ABCD'[i%4],rounds=100,
            gpu=i%4,status='COMPLETED' if complete else 'RUNNING' if i<4 else 'PENDING',
            completedRounds=100 if complete else 1,completedActivations=1000 if complete else 10,
            totalActivations=1000,completedOptimizerSteps=51000 if complete else 510,
            totalOptimizerSteps=51000,progress=1. if complete else .01,
            elapsed_seconds=60.,eta_seconds=0. if complete else 600.,
            checkpoint=dict(activation=1000 if complete else 10),metrics={},
            nodes=10,device='GPU',dataset='cifar10',nonIID=True))
    return dict(status=status,host='synthetic-host',profile='hpc',queue='noniid_queue.json',
        entries=entries,resources=[dict(gpu=i) for i in range(4)],elapsed_seconds=60.,
        totals=dict(total=count,completed=count if status=='COMPLETED' else 0,pending=max(0,count-4),
            failed=0,blocked=0,progress=1. if status=='COMPLETED' else .01),
        activity=[dict(time='2026-10-25T12:34:56+00:00',message='condition_0: checkpoint saved at activation 10')])


def test_cli_defaults_to_five_seconds_and_keeps_override():
    from scripts.hpc import parser
    assert parser().parse_args(['monitor']).interval==5.
    assert parser().parse_args(['monitor','--interval','2']).interval==2.
    with pytest.raises(SystemExit):parser().parse_args(['monitor','--plain','--screen'])


def test_refresh_uses_alternate_screen_and_clears_each_frame(tmp_path):
    stream=io.StringIO();waits=[];original=snapshot();before=copy.deepcopy(original)
    def wait(seconds):
        waits.append(seconds)
        if len(waits)==2:raise KeyboardInterrupt
    dashboard.monitor(tmp_path,screen=True,stream=stream,wait=wait,clock=lambda:0.,
        resources=lambda:{},fetch=lambda _:original)
    text=stream.getvalue()
    assert text.startswith('\x1b[?1049h\x1b[?25l')
    assert text.count('\x1b[H\x1b[2J')==2
    assert '\x1b[?25h\x1b[?1049l' in text
    assert waits==[5.,5.] and 'queue execution is independent' in text
    assert original==before and not list(tmp_path.iterdir())


@pytest.mark.parametrize('term',[None,'xterm-256color'])
def test_real_terminal_refreshes_even_when_term_is_missing(tmp_path,monkeypatch,term):
    if term is None:monkeypatch.delenv('TERM',raising=False)
    else:monkeypatch.setenv('TERM',term)
    class Terminal(io.StringIO):
        def isatty(self):return True
    stream=Terminal()
    dashboard.monitor(tmp_path,once=True,stream=stream,resources=lambda:{},fetch=lambda _:snapshot())
    assert '\x1b[H\x1b[2J' in stream.getvalue()
    assert '\x1b[?1049l' in stream.getvalue()


@pytest.mark.parametrize('status',['COMPLETED','FAILED','STOPPED','BLOCKED'])
def test_terminal_queue_stops_refreshing_without_resource_work_or_writes(tmp_path,status):
    stream=io.StringIO();calls=[]
    def forbidden(*args):raise AssertionError('Finished monitor must not sleep or query resources')
    data=snapshot(status)
    result=dashboard.monitor(tmp_path,stream=stream,fetch=lambda _:calls.append('fetch') or data,
        resources=forbidden,wait=forbidden)
    assert result==status and calls==['fetch']
    if status=='COMPLETED':assert 'ALL EXPERIMENTS COMPLETE (4/4)' in stream.getvalue()
    else:assert f'QUEUE {status}' in stream.getvalue()
    assert not list(tmp_path.iterdir())


def test_running_to_complete_leaves_one_final_summary_in_normal_terminal(tmp_path):
    stream=io.StringIO();states=iter([snapshot(),snapshot('COMPLETED')]);waits=[]
    result=dashboard.monitor(tmp_path,screen=True,stream=stream,wait=waits.append,
        clock=lambda:0.,resources=lambda:{},fetch=lambda _:next(states))
    assert result=='COMPLETED' and waits==[5.]
    after=stream.getvalue().split('\x1b[?1049l')[-1]
    assert after.count('ALL EXPERIMENTS COMPLETE (4/4)')==1
    assert all(f'{letter}  COMPLETED' in after for letter in 'ABCD')


@pytest.mark.parametrize('columns,rows',[(80,24),(96,30),(120,40),(40,15)])
def test_four_active_and_eight_queue_entries_fit_screen(columns,rows):
    data=snapshot(count=8)
    data['entries'][0]['metrics']=dict(mean_node_accuracy=.61,worst_node_accuracy=.52,
        latest_node_accuracy=.63,local_training_loss=1.2,full_test_loss=1.3)
    gpus=[dict(index=i,uuid=f'gpu-{i}',name='synthetic GPU',**{'memory.used':8000,
        'memory.total':97000,'memory.free':89000,'utilization.gpu':20,'temperature.gpu':40}) for i in range(4)]
    system=dict(host='synthetic-host',gpus=gpus,gpu_processes=[])
    text=dashboard.render_screen(data,system,5.,columns,rows)
    assert len(text.splitlines())<=rows-1
    assert all(len(line)<=columns-1 for line in text.splitlines())
    if rows>=24:
        assert all(f'{letter} 100r' in text for letter in 'ABCD')
        assert text.count('PENDING')==4
        assert '61.00%' in text and '52.00%' in text and '63.00%' in text
    assert 'Refresh 5s' in text


def test_completed_metrics_remain_visible_and_missing_metrics_stay_unavailable():
    data=snapshot('COMPLETED',1)
    data['entries'][0]['metrics']=dict(mean_node_accuracy=.61,worst_node_accuracy=.52)
    text=dashboard.render(data,{})
    assert 'ALL EXPERIMENTS COMPLETE (1/1)' in text
    assert 'Mean: 61.00%' in text and 'Worst: 52.00%' in text
    assert 'Latest node: unavailable' in text


def test_once_screen_leaves_a_readable_snapshot_after_restoring_terminal(tmp_path):
    stream=io.StringIO()
    dashboard.monitor(tmp_path,screen=True,once=True,stream=stream,
        fetch=lambda _:snapshot(),resources=lambda:{})
    assert 'PAPERMERGE' in stream.getvalue().split('\x1b[?1049l')[-1]


def test_completion_message_does_not_fabricate_missing_or_inconsistent_counts():
    assert 'ALL EXPERIMENTS COMPLETE' not in dashboard.completion_message(dict(status='COMPLETED'))
    data=snapshot();data['status']='COMPLETED'
    assert 'entry status mismatch' in dashboard.completion_message(data)


def test_completed_monitor_cli_reads_state_without_gui_training_or_file_changes(tmp_path):
    data=snapshot('COMPLETED',1)
    path=tmp_path/'status.json';path.write_text(json.dumps(data))
    before=path.read_bytes()
    code='import sys; from scripts.hpc import main; main(sys.argv[1:]); assert "tkinter" not in sys.modules; assert "tensorflow" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code,'monitor','--state-dir',str(tmp_path),'--plain'],
        capture_output=True,text=True,check=True,timeout=10)
    assert 'ALL EXPERIMENTS COMPLETE (1/1)' in result.stdout
    assert path.read_bytes()==before and sorted(p.name for p in tmp_path.iterdir())==['status.json']


@pytest.mark.parametrize('status,code',[('COMPLETED',0),('FAILED',1),('BLOCKED',1),('STOPPED',0)])
def test_cli_monitor_exit_codes(tmp_path,status,code):
    (tmp_path/'status.json').write_text(json.dumps(snapshot(status,1)))
    result=subprocess.run([sys.executable,'scripts/hpc.py','monitor','--state-dir',str(tmp_path),'--plain'],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==code
