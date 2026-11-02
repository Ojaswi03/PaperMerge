import fcntl
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from basil_core import stable_monitor as stable
from scripts.share_cart_monitor import view_url
from tests.test_sharp_monitor import setup,running


def saved():
    return dict(site='https://example.test',feedId='00000000-0000-4000-8000-000000000001',writeToken='a'*64)


def test_existing_shared_link_is_adopted_and_reused_for_new_run(tmp_path):
    first,second,state=(tmp_path/name for name in ('first','second','private'))
    first.mkdir();second.mkdir();old=saved()
    (first/'browser_share.json').write_text(json.dumps(old))
    def forbidden(*args):
        pytest.fail('The existing monitor must not get a new feed')
    initial=stable.connection(first,old['site'],state,forbidden,view_url)
    following=stable.connection(second,old['site'],state,forbidden,view_url)
    assert initial==following and initial['viewUrl']==view_url(old)
    assert (state/'connection.json').stat().st_mode & 0o777 == 0o600
    assert json.loads((first/'browser_share.json').read_text())==old


def test_permanent_connection_is_created_only_once(tmp_path):
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir();calls=[]
    def create(*args):
        calls.append(args);return saved()
    one=stable.connection(first,saved()['site'],tmp_path/'state',create,view_url)
    two=stable.connection(second,saved()['site'],tmp_path/'state',create,view_url)
    assert one==two and len(calls)==1
    with pytest.raises(ValueError,match='different dashboard'):
        stable.connection(first,'https://different.test',tmp_path/'state',create,view_url)


def test_selected_run_is_local_to_selected_host(tmp_path,monkeypatch):
    stable.select_run(tmp_path,tmp_path/'run')
    assert stable.selected_run(tmp_path)['output']==str(tmp_path/'run')
    monkeypatch.setattr(stable.socket,'gethostname',lambda:'another-host')
    assert stable.selected_run(tmp_path) is None


def test_publisher_follows_new_run_and_keeps_the_same_feed(tmp_path,monkeypatch):
    from scripts import share_cart_monitor as module
    first=tmp_path/'first';second=tmp_path/'second';state=tmp_path/'private'
    first.mkdir();second.mkdir();state.mkdir()
    setup(first,[(2025,'clean','mnist',running(20,40))])
    setup(second,[(2026,'clean','cifar10',running(30,50))])
    stable.select_run(state,first)
    originals={p:p.read_bytes() for folder in (first,second) for p in folder.rglob('*.json')}
    uploads=[];sleeps=[]
    monkeypatch.setattr(module,'request_json',lambda *args,**kwargs:uploads.append(args))
    def wait(_):
        sleeps.append(1)
        if len(sleeps)==1:stable.select_run(state,second)
        else:raise KeyboardInterrupt
    monkeypatch.setattr(module.time,'sleep',wait)
    args=SimpleNamespace(output=first,state_dir=state,stable=True,interval=1,ntfy_topic=None,once=False)
    with pytest.raises(KeyboardInterrupt):module.publish_loop(args,saved())
    assert [u[2]['completedActivations'] for u in uploads]==[20,30]
    assert len({u[0] for u in uploads})==1
    assert all(path.read_bytes()==value for path,value in originals.items())


def test_finished_run_waits_for_future_selection_without_reuploading(tmp_path,monkeypatch):
    from scripts import share_cart_monitor as module
    first=tmp_path/'first';second=tmp_path/'second';state=tmp_path/'private'
    first.mkdir();second.mkdir();state.mkdir()
    setup(first,[(2025,'clean','mnist',dict(running(100,40),status='COMPLETED'))])
    setup(second,[(2026,'clean','mnist',running(10,10))])
    stable.select_run(state,first)
    uploads=[];waits=[]
    monkeypatch.setattr(module,'request_json',lambda *args,**kwargs:uploads.append(args))
    def wait(_):
        waits.append(1)
        if len(waits)==2:stable.select_run(state,second)
        if len(waits)==3:raise KeyboardInterrupt
    monkeypatch.setattr(module.time,'sleep',wait)
    args=SimpleNamespace(output=first,state_dir=state,stable=True,interval=1,ntfy_topic=None,once=False)
    with pytest.raises(KeyboardInterrupt):module.publish_loop(args,saved())
    assert [u[2]['status'] for u in uploads]==['COMPLETED','RUNNING']


def test_background_reuses_an_already_running_local_publisher(tmp_path,monkeypatch):
    (tmp_path/'publisher.json').write_text(json.dumps(dict(host=socket.gethostname(),pid=12345)))
    monkeypatch.setattr(stable.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Duplicate publisher'))
    with (tmp_path/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert stable.launch_background(SimpleNamespace(state_dir=tmp_path),Path('/tmp/script.py')) is False


def test_background_passes_owned_lock_to_child(tmp_path,monkeypatch):
    captures=[]
    def spawn(command,**kwargs):
        descriptor=kwargs['pass_fds'][0]
        assert descriptor>=0 and kwargs['start_new_session']
        with (tmp_path/'publisher.lock').open('a') as another:
            with pytest.raises(BlockingIOError):fcntl.flock(another,fcntl.LOCK_EX|fcntl.LOCK_NB)
        captures.append(command);return SimpleNamespace(pid=12345)
    monkeypatch.setattr(stable.subprocess,'Popen',spawn)
    args=SimpleNamespace(state_dir=tmp_path,output=tmp_path/'run',site='https://example.test',interval=1,ntfy_topic=None)
    assert stable.launch_background(args,Path('/tmp/script.py'))
    assert '--serve-stable' in captures[0] and '--publisher-lock-fd' in captures[0]
    assert json.loads((tmp_path/'publisher.json').read_text())['pid']==12345


@pytest.mark.parametrize('kind', ['reader','training','different-state','other-host'])
def test_refresh_checks_process_identity_before_signalling(tmp_path,monkeypatch,kind):
    script=tmp_path/'scripts/share_cart_monitor.py';state=tmp_path/'private';state.mkdir()
    entry=tmp_path/'proc/22340';entry.mkdir(parents=True)
    words=['python','-u',str(script),'--serve-stable','--state-dir',str(state)]
    if kind=='training':words[2]=str(tmp_path/'scripts/run_fedsharp_500r.py')
    if kind=='different-state':words[-1]=str(tmp_path/'another-state')
    (entry/'cmdline').write_bytes(('\0'.join(words)+'\0').encode())
    (entry/'cwd').symlink_to(tmp_path,target_is_directory=True)
    stable.write_private(state/'publisher.json',dict(pid=22340,
        host='another-host' if kind=='other-host' else socket.gethostname()))
    original=stable.Path
    monkeypatch.setattr(stable,'Path',lambda value:tmp_path/'proc' if str(value)=='/proc' else original(value))
    monkeypatch.setattr(stable.os,'pidfd_open',lambda pid:os.open('/dev/null',os.O_RDONLY))
    signals=[]
    monkeypatch.setattr(stable.signal,'pidfd_send_signal',lambda fd,sig:signals.append(sig))
    if kind in ('training','different-state'):
        with pytest.raises(RuntimeError,match='safely refresh'):stable.retire_stable_publisher(state,script)
    else:
        assert stable.retire_stable_publisher(state,script)==(kind=='reader')
    assert signals==([stable.signal.SIGTERM] if kind=='reader' else [])


def test_refresh_waits_for_old_reader_lock_and_starts_new_code(tmp_path,monkeypatch):
    import subprocess
    import sys
    import time
    script=tmp_path/'share_cart_monitor.py'
    script.write_text('import fcntl, pathlib, sys, time\n'
        'state=pathlib.Path(sys.argv[sys.argv.index("--state-dir")+1])\n'
        'with (state/"publisher.lock").open("a") as lock:\n'
        ' fcntl.flock(lock,fcntl.LOCK_EX)\n'
        ' (state/"ready").touch()\n'
        ' time.sleep(30)\n')
    old=subprocess.Popen([sys.executable,str(script),'--serve-stable','--state-dir',str(tmp_path)])
    try:
        deadline=time.monotonic()+5
        while not (tmp_path/'ready').exists():
            assert old.poll() is None and time.monotonic()<deadline
            time.sleep(.01)
        stable.write_private(tmp_path/'publisher.json',dict(pid=old.pid,host=socket.gethostname()))
        captures=[]
        def spawn(command,**kwargs):
            with (tmp_path/'publisher.lock').open('a') as another:
                with pytest.raises(BlockingIOError):fcntl.flock(another,fcntl.LOCK_EX|fcntl.LOCK_NB)
            captures.append((command,kwargs));return SimpleNamespace(pid=12345)
        monkeypatch.setattr(stable.subprocess,'Popen',spawn)
        args=SimpleNamespace(state_dir=tmp_path,output=tmp_path/'run',site='https://example.test',
            interval=1,ntfy_topic=None,restart_publisher=True)
        assert stable.launch_background(args,script)
        assert old.wait(timeout=5)==-stable.signal.SIGTERM
        assert len(captures)==1 and captures[0][1]['start_new_session']
    finally:
        if old.poll() is None:old.terminate()
        old.wait(timeout=5)


def test_refresh_option_requires_background(tmp_path,monkeypatch):
    import sys
    from scripts import share_cart_monitor as module
    monkeypatch.setattr(sys,'argv',['share_cart_monitor.py','--restart-publisher'])
    with pytest.raises(SystemExit) as error:module.main()
    assert error.value.code==2


def test_permanent_link_can_be_printed_without_an_output_folder(tmp_path,monkeypatch,capsys):
    import sys
    from scripts import share_cart_monitor as module
    stable.write_private(tmp_path/'connection.json',saved())
    monkeypatch.setattr(sys,'argv',['share_cart_monitor.py','--state-dir',str(tmp_path),'--print-link'])
    module.main()
    assert capsys.readouterr().out.strip()==view_url(saved())


def test_legacy_retirement_targets_only_this_output_reporter(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();script=project/'scripts/share_cart_monitor.py'
    output=tmp_path/'run';signals=[];entries=[]
    commands=[['python',str(script),'--output',str(output)],
        ['python',str(script),'--output',str(tmp_path/'other')],
        ['python',str(project/'scripts/run_basil_sharp_thesis.py'),'--output',str(output)],
        ['python',str(script),'--output',str(output),'--serve-stable']]
    for index,words in enumerate(commands,12340):
        folder=tmp_path/'proc'/str(index);folder.mkdir(parents=True)
        (folder/'cmdline').write_bytes(('\0'.join(words)+'\0').encode())
        (folder/'cwd').symlink_to(project,target_is_directory=True);entries.append(folder)
    original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(entries) if path==Path('/proc') else original(path))
    monkeypatch.setattr(stable.os,'pidfd_open',lambda pid:os.open('/dev/null',os.O_RDONLY))
    monkeypatch.setattr(stable.signal,'pidfd_send_signal',lambda fd,sig:signals.append(sig))
    stable.retire_legacy_publisher(output,script)
    assert signals==[stable.signal.SIGTERM]


def test_stop_old_monitors_keeps_training_and_permanent_publisher(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();state=tmp_path/'state';signals=[];entries=[]
    commands=[['python',str(project/'scripts/share_cart_monitor.py'),'--output','/old'],
        ['python',str(project/'scripts/share_cart_monitor.py'),'--temporary','--output','/old'],
        ['python',str(project/'scripts/share_cart_monitor.py'),'--serve-stable','--state-dir',str(state)],
        ['python',str(project/'scripts/run_cart_comparison.py'),'--monitor','--output','/old'],
        ['python',str(project/'scripts/run_cart_comparison.py'),'--execute','--output','/training'],
        ['python',str(project/'scripts/run_basil_sharp_thesis.py'),'--execute','--output','/training'],
        ['python',str(project/'scripts/basil_sharp_thesis_worker.py'),'--output','/training'],
        ['python',str(project/'scripts/hpc.py'),'monitor'],
        ['python',str(project/'scripts/hpc.py'),'run','--queue','monitor']]
    for index,words in enumerate(commands,22340):
        folder=tmp_path/'proc'/str(index);folder.mkdir(parents=True)
        (folder/'cmdline').write_bytes(('\0'.join(words)+'\0').encode())
        (folder/'cwd').symlink_to(project,target_is_directory=True);entries.append(folder)
    original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(entries) if path==Path('/proc') else original(path))
    descriptors={}
    def descriptor(pid):
        value=os.open('/dev/null',os.O_RDONLY);descriptors[value]=pid;return value
    monkeypatch.setattr(stable.os,'pidfd_open',descriptor)
    monkeypatch.setattr(stable.signal,'pidfd_send_signal',lambda fd,sig:signals.append((descriptors[fd],sig)))
    result=stable.stop_old_monitors(project,state)
    assert [pid for pid,_ in signals]==[22340,22341,22343,22347]
    assert result['permanentPublishersKept']==[22342]
    assert signals[2][1]==stable.signal.SIGINT
