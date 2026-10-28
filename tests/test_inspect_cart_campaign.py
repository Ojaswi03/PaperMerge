import hashlib
import json
import os
import socket
from pathlib import Path
import pytest


def fixture(root,monkeypatch):
    from tests.test_cart_v2_completed_report import fixture as original
    configs=original(root,monkeypatch)
    for c in configs:
        count=(3,5,7)[c['seed']-2025] if c['cart'] else (6,8,10)[c['seed']-2025]
        path=root/c['runId'];row=json.loads((path/'run.json').read_text())
        row.update(status='RUNNING',completedRounds=count,completedActivations=count*4,elapsedSeconds=count*10,
            meanAccuracy=.5+.01*count+.001*(c['seed']-2025)-(.02 if c['cart'] else 0))
        (path/'run.json').write_text(json.dumps(row))
        (path/'round_metrics.json').write_text(json.dumps([
            dict(round=i,nodeAccuracy=[.5+.01*(i+1)+.001*(c['seed']-2025)-(.02 if c['cart'] else 0)]*4,
                meanAccuracy=.5+.01*(i+1)+.001*(c['seed']-2025)-(.02 if c['cart'] else 0)) for i in range(count)]))
    (root/'campaign.json').write_text(json.dumps(dict(status='RUNNING',host=socket.gethostname(),
        ownerPid=os.getpid(),elapsedSeconds=100)))
    (root/'managed_status.json').write_text(json.dumps(dict(status='RUNNING',ownerPid=os.getpid())))
    return configs


def test_readonly_all_rows_equal_rounds_and_three_seed_alignment(tmp_path,monkeypatch):
    from scripts.inspect_cart_campaign import inspect,render
    configs=fixture(tmp_path,monkeypatch)
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    before=hashes();result=inspect(tmp_path)
    assert hashes()==before and len(result['campaign']['entries'])==12
    assert len(result['matchedSeedPairs'])==6 and len(result['matchedThreeSeedGroups'])==2
    assert {r['round'] for r in result['matchedSeedPairs']}=={3,5,7}
    for group in result['matchedThreeSeedGroups']:
        assert group['round']==3 and group['deltaPercentagePoints']==pytest.approx(-2.)
        assert group['controlMean']==pytest.approx(.531)
        assert group['cartMean']==pytest.approx(.511)
        assert group['sampleStdPercentagePoints']==pytest.approx(0.)
    assert {g['sigma'] for g in result['matchedThreeSeedGroups']}=={.005,.01}
    assert result['manager']['pidExistsOnThisHost'] is True
    assert 'MATCHED CART / CONTROL' in render(result)


def test_missing_seed_history_prevents_three_seed_claim(tmp_path,monkeypatch):
    from scripts.inspect_cart_campaign import inspect
    configs=fixture(tmp_path,monkeypatch)
    c=next(c for c in configs if c['cart'] and c['seed']==2027 and c['sigma']==.01)
    (tmp_path/c['runId']/'round_metrics.json').unlink()
    result=inspect(tmp_path)
    assert len(result['matchedThreeSeedGroups'])==1 and result['matchedThreeSeedGroups'][0]['sigma']==.005
    assert len(result['waitingPairs'])==1


def test_changed_initialization_and_inconsistent_round_metrics_refuse_comparison(tmp_path,monkeypatch):
    from scripts.inspect_cart_campaign import inspect
    configs=fixture(tmp_path,monkeypatch);c=configs[0];folder=tmp_path/c['runId']
    path=folder/'run.json';row=json.loads(path.read_text());row['initialHash']='changed';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='not paired'):inspect(tmp_path)
    row['initialHash']='init-2025';path.write_text(json.dumps(row))
    path=folder/'round_metrics.json';rows=json.loads(path.read_text());rows[0]['meanAccuracy']=.99;path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Invalid saved round'):inspect(tmp_path)


def test_stopped_manager_alert_is_distinguished_from_running_supervisor(tmp_path,monkeypatch):
    from scripts.inspect_cart_campaign import inspect
    fixture(tmp_path,monkeypatch)
    (tmp_path/'managed_status.json').write_text(json.dumps(dict(status='INTERRUPTED',error='old stop')))
    result=inspect(tmp_path)
    assert result['manager']['status']=='INTERRUPTED' and result['supervisor']['status']=='RUNNING'
    assert any('Manager is stopped' in warning for warning in result['warnings'])


def test_help_does_not_load_training_libraries():
    import subprocess,sys
    path=Path(__file__).resolve().parents[1]/'scripts/inspect_cart_campaign.py'
    code=(f'import runpy,sys;sys.argv=[{str(path)!r},"--help"];'
        f'\ntry:runpy.run_path({str(path)!r},run_name="__main__")'
        '\nexcept SystemExit as e:assert e.code==0'
        '\nassert not {"numpy","tensorflow","matplotlib"}.intersection(sys.modules)')
    assert subprocess.run([sys.executable,'-I','-c',code],capture_output=True).returncode==0
