import json
import os
from types import SimpleNamespace
from pathlib import Path
import pytest
from tests.test_sharp_monitor import setup,running


def test_public_projection_omits_paths_errors_source_and_process_identity(tmp_path):
    from basil_core.cart_public_monitor import public_snapshot
    setup(tmp_path,[(2025,'attack_noise_defense_rla_cart','cifar10',dict(running(20,40),error='/home/private/error'))])
    result=public_snapshot(tmp_path);raw=json.dumps(result)
    assert result['completedActivations']==20 and result['runs'][0]['accuracy']==.8
    assert '/home/' not in raw and str(tmp_path) not in raw
    assert not {'error','source','runtime','ownerPid','output'}.intersection(result)


def test_publisher_preserves_one_private_write_key_and_returns_only_public_link(tmp_path,monkeypatch):
    from scripts import share_cart_monitor as module
    setup(tmp_path,[(2025,'clean','mnist',running(20,40))]);calls=[]
    monkeypatch.setattr(module,'request_json',lambda *args,**kwargs:calls.append(args) or dict(feedId='00000000-0000-4000-8000-000000000001',writeToken='a'*64))
    first=module.connection(tmp_path,'https://example.test');second=module.connection(tmp_path,'https://example.test')
    assert first==second and len(calls)==1
    assert 'a'*64 not in first['viewUrl']
    assert os.stat(tmp_path/'browser_share.json').st_mode&0o777==0o600
    with pytest.raises(ValueError,match='different dashboard'):module.connection(tmp_path,'https://other.test')


def test_publisher_uploads_final_state_then_exits_without_touching_training(tmp_path,monkeypatch,capsys):
    from scripts import share_cart_monitor as module
    setup(tmp_path,[(2025,'clean','mnist',dict(running(100,40),status='COMPLETED'))])
    before={str(p):p.read_bytes() for p in tmp_path.glob('*.json')};calls=[]
    monkeypatch.setattr(module,'request_json',lambda *args:calls.append(args) or {})
    args=SimpleNamespace(output=tmp_path,interval=10,once=False,ntfy_topic=None)
    module.publish_loop(args,dict(site='https://example.test',feedId='fixture',writeToken='a'*64))
    assert len(calls)==1 and calls[0][1]=='PUT' and calls[0][2]['status']=='COMPLETED'
    assert before=={str(p):p.read_bytes() for p in tmp_path.glob('*.json')}
    assert capsys.readouterr().out==''


def test_dashboard_origin_rejects_embedded_credentials_and_non_https():
    from scripts.share_cart_monitor import origin
    for value in ('http://example.test','https://user:secret@example.test','https://example.test/path','https://example.test?token=secret'):
        with pytest.raises(ValueError):origin(value)


def test_existing_feed_gets_a_github_share_address_without_changing_credentials():
    from scripts.share_cart_monitor import view_url,SITE
    saved=dict(site=SITE,feedId='00000000-0000-4000-8000-000000000001',writeToken='a'*64,
        viewUrl=SITE+'/?feed=00000000-0000-4000-8000-000000000001')
    before=dict(saved);url=view_url(saved)
    assert url=='https://ojaswi03.github.io/PaperMerge/?feed='+saved['feedId']
    assert 'chatgpt' not in url and saved==before


def test_ctrl_c_revokes_temporary_feed_and_preserves_training_files(tmp_path,monkeypatch):
    from scripts import share_cart_monitor as module
    setup(tmp_path,[(2025,'clean','mnist',running(20,40))]);before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    saved=dict(site='https://example.test',feedId='temporary',writeToken='a'*64,viewUrl='https://example.test/?feed=temporary')
    monkeypatch.setattr(module,'new_connection',lambda *args,**kwargs:saved)
    def interrupt(*_):raise KeyboardInterrupt
    monkeypatch.setattr(module,'publish_loop',interrupt);calls=[]
    monkeypatch.setattr(module,'request_json',lambda *args,**kwargs:calls.append((args,kwargs)) or {})
    with pytest.raises(KeyboardInterrupt):module.temporary_session(SimpleNamespace(output=tmp_path,site='https://example.test'))
    assert calls[0][0][1]=='DELETE' and calls[0][1]['token']==saved['writeToken']
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_temporary_session_requests_a_server_expiry_without_saving_a_key(tmp_path,monkeypatch):
    from scripts import share_cart_monitor as module
    setup(tmp_path,[(2025,'clean','mnist',running(20,40))]);calls=[]
    monkeypatch.setattr(module,'request_json',lambda *args,**kwargs:calls.append(kwargs) or dict(feedId='00000000-0000-4000-8000-000000000001',writeToken='a'*64))
    module.new_connection(tmp_path,'https://example.test',temporary=True)
    assert calls==[dict(lease_seconds=45)] and not (tmp_path/'browser_share.json').exists()
