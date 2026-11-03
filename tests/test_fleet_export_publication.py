from pathlib import Path
import pytest
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha
from scripts.check_fedsharp_fleet_exports import check


@pytest.fixture
def fragments(tmp_path):
    queue=tmp_path/'queue';items=[];runs={};exports=[]
    for host,count,start in [('hal9k',35,0),('themachine',115,35)]:
        root=tmp_path/host;configs=[];verified=[]
        for i in range(start,start+count):
            run=f'run{i}';cfg=dict(runId=run,rounds=500);configs.append(cfg);items.append(dict(config=cfg))
            runs[run]=dict(state='DONE',owner=host)
            atomic(root/run/'run.json',dict(status='COMPLETED'));(root/run/'metrics.npz').write_bytes(b'metrics')
            verified.append(dict(runId=run,resultSha256={name:sha(root/run/name) for name in ('run.json','metrics.npz')}))
        atomic(root/'plan.json',dict(configs=configs,host=host,source={},continuationQueueId='queue'))
        atomic(root/'independent_validation.json',dict(status='ALL_COMPARISON_MODELS_VERIFIED',
            officialTestUsed=False,source={},verified=verified))
        atomic(root/'SHA256.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
        exports.append(root)
    atomic(queue/'continuation_plan.json',dict(items=items,source={},queueId='queue'))
    atomic(queue/'fleet/state.json',dict(runs=runs))
    return queue,exports


def test_exact_disjoint_150_run_publication(fragments):
    queue,exports=fragments
    result=check(queue,exports)
    assert result['total']==150 and result['hosts']=={'hal9k':35,'themachine':115}


def test_missing_fragment_and_modified_metrics_refused(fragments):
    queue,exports=fragments
    with pytest.raises(ValueError,match='all 150'):check(queue,exports[:1])
    (exports[0]/'run0/metrics.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):check(queue,exports)


def test_extra_runtime_files_refused(fragments):
    queue,exports=fragments
    (exports[0]/'run0/checkpoint.npz').write_bytes(b'checkpoint')
    with pytest.raises(ValueError,match='coverage'):check(queue,exports)
