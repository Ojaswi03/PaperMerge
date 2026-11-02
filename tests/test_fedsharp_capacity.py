from types import SimpleNamespace

import pytest

from scripts import check_fedsharp_capacity as capacity
from tests.test_live_gpu_scheduler import profile
from basil_core.cart_campaign_io import write_json


@pytest.mark.parametrize('available,expect_more',[(57*1024,False),(256*1024,True)])
def test_capacity_explains_host_ram_limit_despite_free_vram(tmp_path,monkeypatch,available,expect_more):
    from basil_core import continuation_campaign
    queue=tmp_path/'queue';output=tmp_path/'output';output.mkdir();binding=queue/'host_bindings/hal9k'
    binding.mkdir(parents=True);write_json(binding/'owner.json',dict(output=str(output)))
    monkeypatch.setattr(continuation_campaign,'host_name',lambda:'hal9k')
    hardware={gpu:profile(output,gpu) for gpu in (0,1)}
    plan=dict(configs=[dict(runId=f'run{i}') for i in range(9)],maximumWorkersPerGpu=24,
        resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=16384))
    monkeypatch.setattr(capacity.live,'verified',lambda _:(plan,hardware,[],''))
    inventory=[dict(index=g,uuid=hardware[g].gpu_uuid,**{'memory.total':97887,'memory.free':80000,'utilization.gpu':25}) for g in hardware]
    monkeypatch.setattr(capacity.live,'gpu_inventory',lambda:inventory)
    processes=[dict(pid=100+i,gpu_uuid=hardware[i%2].gpu_uuid,memory_mib=3652) for i in range(9)]
    monkeypatch.setattr(capacity.live,'gpu_processes',lambda:processes)
    monkeypatch.setattr(capacity.live,'discover',lambda *a:{i:dict(pid=100+i,profile=dict(gpu=i%2)) for i in range(9)})
    monkeypatch.setattr(capacity,'worker_memory',lambda pid:dict(rssMiB=4096,peakRssMiB=4200,swapMiB=0))
    original=capacity.Path.read_text
    monkeypatch.setattr(capacity.Path,'read_text',lambda path,*a,**kw:
        f'MemTotal: 97517568 kB\nMemAvailable: {available*1024} kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n'
        if str(path)=='/proc/meminfo' else original(path,*a,**kw))
    write_json(output/'campaign.json',dict(pending=3,failed=3,scheduler=dict(blocked={})))
    result=capacity.probe(queue)
    assert not result['trainingModified']
    assert bool(result['additionalWorkersWithinCurrentBudgets'])==expect_more
    if not expect_more:
        for decision in result['nextWorkerAdmission'].values():
            assert decision['limitingResources']==['host RAM']
            assert decision['gpuFreeMiB']>decision['gpuRequiredMiB']
            assert decision['ramAvailableMiB']<decision['ramRequiredMiB']
