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
