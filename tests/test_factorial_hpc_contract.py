"""Parent/child source and device identities without allocating a GPU."""
from argparse import Namespace
import json


def test_hpc_parent_fingerprints_under_selected_profile_and_overrides_cpu_visibility(monkeypatch,tmp_path):
    import basil_core.execution_profile as resources
    import scripts.run_factorial_study as runner
    import subprocess
    from basil_core.factorial_study import configuration,CONDITIONS
    actual_root=runner.ROOT;actual_check=runner.check_source
    def check_real_sources(source):
        with monkeypatch.context() as scoped:
            scoped.setattr(runner,'ROOT',actual_root)
            actual_check(source)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'check_source',check_real_sources)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    inventory=[dict(index=0,uuid='GPU-01234567-0123-0123-0123-0123456789ab',name='test GPU',
        **{'memory.free':32000,'driver_version':'test'})]
    monkeypatch.setattr(resources,'gpu_inventory',lambda:inventory)
    launched=[]
    class Process:
        def __init__(self,cmd,**kwargs):launched.append(kwargs['env'])
        def poll(self):return 0
    monkeypatch.setattr(subprocess,'Popen',Process)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    output=tmp_path/'.papermerge/local-tests/hpc-contract'
    args=Namespace(profile='hpc',gpus='0',output=output,recover=False,stage='smoke',sigmas=[.005,.01,.02])
    cfg=configuration(CONDITIONS[0],'iid',2025,0.,stage='smoke')
    runner.execute(args,configs=[cfg],prepared_hashes={'fitImages':'fake'})
    plan=json.loads((output/'plan.json').read_text())
    assert 'scripts/hpc.py' in plan['source']
    assert 'basil_core/execution_profile.py' in plan['source']
    env=launched[0]
    assert env['CUDA_VISIBLE_DEVICES']==inventory[0]['uuid']
    assert json.loads(env[resources.ENV])['name']=='hpc'
