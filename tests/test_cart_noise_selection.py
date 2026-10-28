import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def args(count=1,values=None,conditions=None):
    return SimpleNamespace(noise_levels=count,noise_values=values,datasets=['mnist','cifar10'],
        partitions=['iid','dirichlet'],seeds=[2025,2026,2027],conditions=conditions or ['attack_noise_defense_rla_cart'],
        nodes=100,rounds=100,local_steps=5,model='paper',alpha=.5,attack_start=20,
        train_per_class=None,validation_per_class=None)


@pytest.mark.parametrize('count,total',[(1,12),(2,24),(3,36)])
def test_cart_only_counts_and_explicit_levels(count,total):
    from basil_core.cart_noise_selection import configurations
    selection,configs=configurations(args(count))
    assert len(configs)==total and len({c['sigma'] for c in configs})==count
    assert all(c['nodes']==100 and c['rounds']==100 for c in configs)
    assert set(c['sigma'] for c in configs)==set(selection.values)


@pytest.mark.parametrize('count,total',[(1,72),(2,132),(3,192)])
def test_focused_matrix_does_not_duplicate_noise_free_reference(count,total):
    from basil_core.cart_noise_selection import configurations
    conditions=['clean','attack_noise','attack_noise_defense','attack_noise_rla',
        'attack_noise_defense_rla','attack_noise_defense_rla_cart']
    _,configs=configurations(args(count,conditions=conditions))
    assert len(configs)==total and sum(c['condition']=='clean' for c in configs)==12
    assert len({c['runId'] for c in configs})==total


def test_flexible_generation_matches_frozen_matrix_exactly():
    from basil_core.cart_noise_selection import configurations
    from scripts.run_cart_v2_study import selected
    a=args(3);_,new=configurations(a)
    a.sigmas=[.005,.01,.02];a.methods=['sharp'];a.hop_denominator='actual_selected';a.curvature=1.
    a.mu=1.;a.learning_rate=.03;a.attack_type='hidden';a.budget=None;a.basil_window=None;a.only_sigma=None
    assert new==selected(a)
    a.noise_levels=1;a.noise_values=[.01];a.only_sigma=.01
    assert configurations(a)[1]==selected(a)


@pytest.mark.parametrize('count,values',[(0,None),(4,None),(1,[.01,.02]),(2,[.01]),
    (2,[.01,.01]),(1,[0.]),(1,[-.01]),(1,[float('nan')]),(1,[float('inf')])])
def test_invalid_count_or_sigma_is_rejected(count,values):
    from basil_core.cart_noise_selection import NoiseSelection
    with pytest.raises(ValueError):NoiseSelection.create(count,values)


def test_custom_two_levels_and_default_midpoint():
    from basil_core.cart_noise_selection import NoiseSelection
    assert NoiseSelection.create().values==(.01,)
    assert NoiseSelection.create(2,[.03,.01]).values==(.01,.03)


def test_prepare_preserves_plan_and_refuses_changed_noise(tmp_path,monkeypatch):
    import scripts.prepare_cart_study as builder
    import scripts.run_cart_v2_scaled as scaled
    from basil_core.execution_profile import ExecutionProfile
    from basil_core.cart_noise_selection import configurations
    a=args();a.output=tmp_path;a.jobs_per_gpu=6;a.context_overhead_mib=1024;a.ram_headroom_mib=32768
    a.allow_unattributed_process_memory=False
    p=ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',output_root=str(tmp_path))
    monkeypatch.setattr(scaled,'profiles',lambda _:[p])
    monkeypatch.setattr(scaled.core,'data_contract',lambda _:({},{}))
    monkeypatch.setattr(scaled.core,'fingerprint',lambda:{'source':'same'})
    selection,configs=configurations(a);plan,_=builder.prepare(a,selection,configs)
    before=(tmp_path/'plan.json').read_bytes();builder.prepare(a,selection,configs)
    assert (tmp_path/'plan.json').read_bytes()==before
    assert plan['noiseSelection']==dict(count=1,values=[.01])
    a.noise_levels=2
    with pytest.raises(ValueError,match='noise/source differs'):builder.prepare(a,*configurations(a))
    assert (tmp_path/'plan.json').read_bytes()==before


def test_foreign_active_plan_is_not_touched(tmp_path):
    import scripts.prepare_cart_study as builder
    from basil_core.cart_noise_selection import configurations
    a=args();a.output=tmp_path
    (tmp_path/'plan.json').write_text('{"executionProgram":{"old":"frozen"}}')
    before=(tmp_path/'plan.json').read_bytes()
    with pytest.raises(FileExistsError,match='fresh output'):builder.prepare(a,*configurations(a))
    assert (tmp_path/'plan.json').read_bytes()==before and len(list(tmp_path.iterdir()))==1


def test_preview_and_help_need_no_gpu_and_start_no_training(tmp_path):
    import subprocess,sys
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_cart_study.py'
    for count,values,total in [(1,['.01'],12),(2,['.005','.02'],24),(3,['.005','.01','.02'],36)]:
        result=subprocess.run([sys.executable,str(path),'--noise-levels',str(count),'--noise-values',*values],
            capture_output=True,text=True,check=True)
        summary=json.loads(result.stdout)
        assert summary['configs']==total and summary['trainingStarted'] is False
        assert summary['controlReferenceNeededForComparison'] is True
    code=(f'import runpy,sys;sys.argv=[{str(path)!r},"--help"];'
        f'\ntry:runpy.run_path({str(path)!r},run_name="__main__")'
        '\nexcept SystemExit as e:assert e.code==0'
        '\nassert not {"numpy","tensorflow","matplotlib"}.intersection(sys.modules)')
    assert subprocess.run([sys.executable,'-I','-c',code],capture_output=True).returncode==0
