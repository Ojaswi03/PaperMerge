"""Zero-research-training contracts for the authorized selection ablation."""
import json
import subprocess
import sys
import pytest


def test_three_round_plan_lazy_clean_and_paired():
    command="from scripts.run_cart_owner_selection_diagnostic import main; main(); import sys; assert not ({'numpy','tensorflow','matplotlib','wandb'} & sys.modules.keys())"
    result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['trainingRunsStarted']==0
    assert plan['differences']==['condition','snapshotSelection','startingModelRule']
    for c in plan['configs']:
        assert c['nRounds']==3 and c['localEpochs']==5 and c['batchSize']==512
        assert c['basilMemorySize']==5 and c['cartMuMax']==.01 and c['helperSamples']==16
        assert c['actualAttackers']==[] and not c['ebm'] and not c['channelNoise'] and not c['liveTrackingEnabled']


def test_ss_off_chooses_predecessor_even_if_others_have_lower_local_loss():
    from basil_core.research_protocol import Snapshot
    from scripts.run_cart_owner_selection_diagnostic import choose_start
    candidates=[Snapshot(5,1,[.2]),Snapshot(9,1,[.8])]
    chosen,diagnostics,loss,meta=choose_start(0,candidates,lambda p:(p[0],1-p[0]),snapshot_selection=False)
    assert chosen.sender_id==9 and loss==.8 and meta['minimumLossSender']==5
    assert len(diagnostics)==2 and not meta['selectionApplied']
    chosen,_,loss,_=choose_start(0,candidates,lambda p:(p[0],1-p[0]),snapshot_selection=True)
    assert chosen.sender_id==5 and loss==.2


def test_all_candidates_receive_same_batch_and_tie_is_nearest():
    from basil_core.research_protocol import Snapshot
    from scripts.run_cart_owner_selection_diagnostic import choose_start
    candidates=[Snapshot(s,10,[]) for s in (5,6,7,8,9)]
    batch=object();seen=[]
    def evaluator(params):
        seen.append(batch);return 1.,0.
    chosen,_,_,_=choose_start(0,candidates,evaluator,snapshot_selection=True)
    assert chosen.sender_id==9 and len(seen)==5 and all(x is batch for x in seen)


def test_warmed_owners_exclude_self_no_future_classes_in_first_traversal():
    from basil_core.cart_owner_gradient_worker import available_owner_classes
    for node in range(10):
        assert available_owner_classes(0,node,diagnostic_rounds=3)==tuple(range(node))
        for r in (1,2):
            assert available_owner_classes(r,node,diagnostic_rounds=3)==tuple(c for c in range(10) if c!=node)
    with pytest.raises(ValueError):available_owner_classes(1,0)
    with pytest.raises(ValueError):available_owner_classes(3,0,diagnostic_rounds=3)
    with pytest.raises(ValueError):available_owner_classes(0,0,diagnostic_rounds=31)


def test_three_round_memory_and_exact_predecessor_handoff_with_dummy_parameters():
    import numpy as np
    from basil_core.research_protocol import initialize_memories,Snapshot,broadcast,params_hash
    from scripts.run_strict_cart_diagnostic import expected_candidates
    from scripts.run_cart_owner_selection_diagnostic import choose_start
    initial=[np.array([0.],np.float32)];memories=initialize_memories(initial);last=initial
    for activation in range(30):
        r,node=divmod(activation,10);candidates=memories[node].candidates()
        assert [(s.sender_id,s.round_id) for s in candidates]==expected_candidates(r,node)
        selected,_,_,_=choose_start(node,candidates,lambda p:(float(p[0][0]),0.),snapshot_selection=False)
        assert params_hash(selected.params)==params_hash(last)
        last=[np.array([activation+1.],np.float32)]
        broadcast(memories,Snapshot(node,r,last))


def test_report_rejects_missing_or_incomplete_controls():
    from scripts.report_cart_owner_selection_diagnostic import summarize
    with pytest.raises(ValueError,match='Thirty complete'):summarize([],'basil_ss')
    with pytest.raises(ValueError,match='Thirty complete'):summarize([{}]*30,'unknown')
