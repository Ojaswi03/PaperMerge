import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from scripts.select_cart_gpus import selection


def inventory(ids):return [dict(index=i,uuid=f'GPU-{i}-hardware') for i in ids]


@pytest.mark.parametrize('available,expected',[(1,[0]),(2,[0,1]),(3,[0,1,2]),(4,[0,1,2,3]),(8,[0,1,2,3])])
def test_detects_server_size_without_requesting_nonexistent_gpus(available,expected):
    result=selection(inventory(range(available)))
    assert result['gpus']==expected and result['maximumWorkers']==2*len(expected)


def test_explicit_count_and_noncontiguous_ids_and_worker_count():
    assert selection(inventory([1,3,5]),count=2)['gpus']==[1,3]
    r=selection(inventory([0,1,2,3]),ids='2, 3',count=2,jobs=1)
    assert r['gpus']==[2,3] and r['maximumWorkers']==2


@pytest.mark.parametrize('kwargs',[dict(count=0),dict(count=5),dict(count=3),dict(ids='0,0'),
    dict(ids='-1'),dict(ids='0,2'),dict(ids='0,1',count=1),dict(jobs=0),dict(jobs=9)])
def test_rejects_impossible_or_ambiguous_resource_requests(kwargs):
    with pytest.raises(ValueError):selection(inventory([0,1]),**kwargs)


def test_resume_retains_gpu_uuids_and_slots_and_rejects_another_server():
    saved=dict(profiles=[dict(gpu=i,gpu_uuid=f'GPU-{i}-hardware') for i in [0,1,0,1]],jobsPerGpu=2)
    assert selection(inventory([0,1]),saved=saved)['gpus']==[0,1]
    assert selection(inventory([0,1]),count=2,saved=saved)['jobsPerGpu']==2
    for kwargs in [dict(count=1),dict(ids='1,0'),dict(jobs=1)]:
        with pytest.raises(ValueError,match='fresh output'):selection(inventory([0,1]),saved=saved,**kwargs)
    changed=inventory([0,1]);changed[0]['uuid']='GPU-replaced-device'
    with pytest.raises(ValueError,match='fresh output'):selection(changed,saved=saved)


def test_resume_preserves_nondefault_worker_count():
    saved=dict(profiles=[dict(gpu=0,gpu_uuid='GPU-0-hardware')],jobsPerGpu=1)
    assert selection(inventory([0,1]),saved=saved)['jobsPerGpu']==1


def test_launcher_uses_two_gpu_host_and_cli_overrides_old_four_gpu_environment(tmp_path):
    root=Path(__file__).resolve().parents[1];repo=tmp_path/'repo'
    (repo/'scripts').mkdir(parents=True);(repo/'basil_core').mkdir()
    (repo/'.venv-hpc/bin').mkdir(parents=True)
    for name in ('launch_cart_comparison_hpc.sh','select_cart_gpus.py'):
        shutil.copyfile(root/'scripts'/name,repo/'scripts'/name)
    (repo/'basil_core/execution_profile.py').write_text('def gpu_inventory():\n return '+repr(inventory([0,1]))+'\n')
    (repo/'.venv-hpc/bin/python').symlink_to(sys.executable)
    (repo/'.venv-hpc/bin/activate').write_text('export PATH="'+str(repo/'.venv-hpc/bin')+':$PATH"\n')
    env=dict(os.environ,CART_COMPARISON_GPUS='0,1,2,3')
    command=['bash',str(repo/'scripts/launch_cart_comparison_hpc.sh'),'isolation','--gpu-count','2','--dry-run']
    success=subprocess.run(command,env=env,text=True,capture_output=True)
    assert success.returncode==0 and 'Physical GPUs: 0,1' in success.stdout
    assert 'no verification or training started' in success.stdout and not (repo/'.papermerge').exists()
    command[command.index('2')]='3'
    failure=subprocess.run(command,env=env,text=True,capture_output=True)
    assert failure.returncode!=0 and 'only 2' in failure.stderr and not (repo/'.papermerge').exists()


def test_full_launcher_assembles_the_requested_24_case_matrix_without_training(tmp_path):
    root=Path(__file__).resolve().parents[1];repo=tmp_path/'repo'
    (repo/'scripts').mkdir(parents=True);(repo/'basil_core').mkdir();(repo/'.venv-hpc/bin').mkdir(parents=True)
    for name in ('launch_cart_comparison_hpc.sh','select_cart_gpus.py'):
        shutil.copyfile(root/'scripts'/name,repo/'scripts'/name)
    (repo/'basil_core/execution_profile.py').write_text('def gpu_inventory():\n return '+repr(inventory(range(4)))+'\n')
    (repo/'.venv-hpc/bin/python').symlink_to(sys.executable)
    (repo/'.venv-hpc/bin/activate').write_text('export PATH="'+str(repo/'.venv-hpc/bin')+':$PATH"\n')
    (repo/'scripts/hpc.py').write_text('')
    (repo/'scripts/run_cart_comparison.py').write_text(
        'import json,sys\nfrom pathlib import Path\n'
        'with Path("invocations.jsonl").open("a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    result=subprocess.run(['bash',str(repo/'scripts/launch_cart_comparison_hpc.sh'),'full','--gpu-count','4'],
        env={k:v for k,v in os.environ.items() if not k.startswith('CART_COMPARISON_')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    import json
    calls=[json.loads(line) for line in (repo/'invocations.jsonl').read_text().splitlines()]
    assert len(calls)==3 and '--preflight' in calls[0] and '--verify-gpus' in calls[1] and '--execute' in calls[2]
    def values(argv,flag):
        start=argv.index(flag)+1;end=next((i for i in range(start,len(argv)) if argv[i].startswith('--')),len(argv))
        return argv[start:end]
    for call in calls[:2]:
        assert values(call,'--datasets')==['mnist','cifar10']
        assert values(call,'--arms')==['acds','prior_fallback']
        assert values(call,'--partitions')==['iid','dirichlet']
        assert values(call,'--seeds')==['2025','2026','2027']
        assert values(call,'--nodes')==['100'] and values(call,'--rounds')==['100']
        assert values(call,'--gpus')==['0,1,2,3']
