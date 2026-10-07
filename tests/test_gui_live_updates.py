"""Live event/rendering regressions, with no experiment execution."""
import copy
import json
import os
from types import SimpleNamespace

import pytest

from gui.services.execution_service import ExecutionEvent, ExecutionService
from gui.services.iid_execution_service import IidExecutionHandle
from gui.services.research_telemetry import activation_payload
from gui.state import ApplicationState, ExecutionState, QueueState, Workspace
from gui.state.experiment_state import from_persisted
from basil_core.iid_campaign import CONFIG_ROOT, CONDITIONS


@pytest.fixture
def config():
    return json.loads((CONFIG_ROOT/f'{CONDITIONS["A"][0]}.json').read_text())


@pytest.fixture
def record():
    return dict(round=0,nodeId=0,inputSourceNode=9,inputSnapshotRound=-1,
        configuredByzantine=False,attackActive=False,attackRelativeParameterChange=0.,
        candidates=[dict(senderId=i,receiverLocalLoss=(10-i)/10) for i in range(5,10)],
        trainedOutputHash='measured_hash',baseGradientNorm=1.,robustObjectiveGradientNorm=1.,
        optimizerSteps=50,ebmMode='none',ebmCoefficient=0.,effectiveCoordinateSigma=0.,
        outgoingLinks=[dict(receiverId=i,noiseL2Norm=0.,relativeNoiseL2=0.,coordinateSigma=0.) for i in range(1,6)],
        globalTestAccuracy=.25,perClassGlobalTestAccuracy=[.25]*10)


def bridge(config,state):
    service=ExecutionService(state)
    handle=IidExecutionHandle.__new__(IidExecutionHandle)
    handle.config=config; handle.sink=service.post
    return handle,service


def test_iid_activation_updates_accuracy_and_network_in_one_notification(config,record):
    original=copy.deepcopy(record)
    state=ApplicationState(experiment=from_persisted(config),execution=ExecutionState.RUNNING)
    notifications=[];state.subscribe(lambda:notifications.append(1))
    handle,service=bridge(config,state)
    service.post(ExecutionEvent('network_started',{'lane':0,'config':config}))
    payload=activation_payload(record,config['experimentName'])
    json.dumps(payload,allow_nan=False)
    handle._dispatch(payload)
    assert service.dispatch_pending()==4 and notifications==[1]
    assert state.latest_node==0 and state.latest_node_accuracy==.25
    assert state.activation_accuracy_history==[(.1,.25)]
    assert state.average_accuracy is None and state.worst_accuracy is None
    assert state.accuracy_history==[]  # Never invent a mean from a partial round.
    assert state.network_events[0][0]=='network_started'
    kind,event=state.network_events[1]
    assert kind=='network_updated' and event['payload']['incoming']['senderId']==9
    assert len(event['payload']['selection']['candidates'])==len(event['payload']['outgoing'])==5
    assert record==original


def test_round_metrics_remain_separate_from_activation_accuracy(config):
    state=ApplicationState(experiment=from_persisted(config),execution=ExecutionState.RUNNING)
    handle,service=bridge(config,state)
    handle._dispatch(dict(roundComplete=dict(round=0,averageAccuracy=.3,worstNodeAccuracy=.2,perNodeAccuracy=[.3]*10)))
    service.dispatch_pending()
    assert state.accuracy_history==[(1,.3,.2)]
    assert state.network_events[-1][0]=='network_round_updated'
    assert state.network_events[-1][1]['payload']['worstNodeAccuracy']==.2


def test_older_worker_events_still_update_live_accuracy_without_fake_selection(config):
    state=ApplicationState(experiment=from_persisted(config))
    handle,service=bridge(config,state)
    handle._dispatch({'round':1,'node':2,'accuracy':.4})
    service.dispatch_pending()
    assert state.activation_accuracy_history==[(1.3,.4)]
    update=state.network_events[-1][1]['payload']
    assert update['telemetryComplete'] is False and 'selection' not in update


def test_new_run_clears_previous_live_curves(config):
    state=ApplicationState(experiment=from_persisted(config),execution=ExecutionState.COMPLETED,
        average_accuracy=.7,worst_accuracy=.6,current_round=100,progress=1.,
        latest_node=9,latest_node_accuracy=.75,accuracy_history=[(100,.7,.6)],activation_accuracy_history=[(100.,.75)])
    state.transition(ExecutionState.PREPARING)
    assert state.average_accuracy is state.worst_accuracy is state.latest_node_accuracy is None
    assert state.current_round==0 and state.progress==0
    assert state.accuracy_history==state.activation_accuracy_history==[]


def test_network_reducer_accepts_iid_node_and_round_metrics(config,record):
    from gui.network_view import NetworkView
    view=NetworkView.__new__(NetworkView)
    view.runs={};view.active_lane=SimpleNamespace(get=lambda:0,set=lambda value:None)
    view._refresh_lane_values=lambda:None;view._refresh_node_values=lambda count:None
    view.round_scale=view.round_label=SimpleNamespace(configure=lambda **kwargs:None)
    view.request_redraw=lambda:None
    view.start_run(0,config);view.apply_node_update(0,config,activation_payload(record,'test')['networkUpdate'])
    assert view.runs[0]['perNode'][0]==.25 and view.runs[0]['activeNode']==0
    view.apply_round(0,config,dict(round=1,averageAccuracy=.3,worstNodeAccuracy=.2,perNodeAccuracy=[.3]*10))
    assert view.runs[0]['worst']==.2 and view.runs[0]['perNode']==[.3]*10
    view.finish_run(0,'stopped');assert view.runs[0]['status']=='stopped'


@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='Optional display integration test')
def test_real_widgets_stay_stable_and_network_and_chart_render(config,record,tmp_path,monkeypatch):
    import tkinter as tk
    from basil_core.research_protocol import validate_config
    from gui.views.application_shell import ApplicationShell
    from gui.services.result_service import ResultService
    from gui.theme import apply_theme
    def forbidden(*args,**kwargs):raise AssertionError('Rendering tests must not launch research processes')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    root=tk.Tk()
    try:
        root.geometry('1440x900');root.rowconfigure(0,weight=1);root.columnconfigure(0,weight=1);apply_theme(root)
        state=ApplicationState(experiment=from_persisted(config),execution=ExecutionState.RUNNING)
        queue=QueueState();entry=queue.add(from_persisted(config))
        commands={name:lambda *args:None for name in ('save','load','queue','run','stop','queue_changed','library','plot','load_queue')}
        shell=ApplicationShell(root,state,queue,ResultService(tmp_path),commands)
        state.subscribe(shell.refresh)
        dashboard=shell.views[Workspace.DASHBOARD];qview=shell.views[Workspace.QUEUE]
        root.update_idletasks();root.update()
        identities=tuple(str(w) for w in dashboard.content.winfo_children())
        chart=dashboard.chart; chart_id=str(chart)
        qview.tree.selection_set(entry.entry_id);qview.tree.focus(entry.entry_id)
        deletes=[];monkeypatch.setattr(qview.tree,'delete',lambda *items:deletes.extend(items))
        handle,service=bridge(config,state)
        service.post(ExecutionEvent('network_started',{'lane':0,'config':config}))
        handle._dispatch(activation_payload(record,'test'));service.dispatch_pending()
        root.update_idletasks();root.update()
        assert dashboard.latest.cget('text').startswith('Latest completed node 0: 25.00%')
        assert config['experimentName'] in dashboard.identity.cget('text')
        assert 'Experiment elapsed' in dashboard.values and 'Queue ETA remaining' in dashboard.values
        assert dashboard.values['Average accuracy'].cget('text')=='—'
        assert str(dashboard.chart)==chart_id and dashboard.chart.winfo_height()>60
        assert 'oval' in [chart.type(item) for item in chart.find_all()]  # The first sample is visible.
        chart_items=chart.find_all()
        for _ in range(12):shell.refresh()
        root.update_idletasks();root.update()
        assert chart.find_all()==chart_items and not deletes
        state.accuracy_history=[(1,.3,.2),(2,.4,.3)]
        state.activation_accuracy_history=[(.1,.25),(.2,.35)]
        dashboard.refresh()
        lines=[item for item in chart.find_all() if chart.type(item)=='line']
        assert len({chart.itemcget(item,'fill') for item in lines})==3
        assert len({chart.itemcget(item,'dash') for item in lines})==3
        assert tuple(str(w) for w in dashboard.content.winfo_children())==identities
        assert qview.tree.selection()==(entry.entry_id,) and qview.tree.focus()==entry.entry_id
        shell.navigate(Workspace.NETWORK);root.update_idletasks();shell.network_view._redraw()
        assert len(shell.network_view.canvas.find_all())>10
        assert '25.00%' in shell.network_view.details.get('1.0','end')
        assert len(shell.network_view.candidate_tree.get_children())==5
        assert len(shell.network_view.link_tree.get_children())==5
        def widget_text(widget):
            try:yield str(widget.cget('text'))
            except tk.TclError:pass
            for child in widget.winfo_children():yield from widget_text(child)
        assert not any('campaign' in text.lower() for text in widget_text(shell))
    finally:root.destroy()
