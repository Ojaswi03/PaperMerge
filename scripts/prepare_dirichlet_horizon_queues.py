#!/usr/bin/env python3
"""Prepare idle 100/500-round queues or audit finished 100 rounds; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save_new(path,value):
    from scripts.run_noniid_basil_study import save_json
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'Refusing to replace existing different preparation/evidence: {path}')
    else:save_json(path,value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--validate-100',action='store_true')
    mode.add_argument('--combine',action='store_true',help='Combine existing approved presets; no training or source reapproval.')
    parser.add_argument('--load-100-queue',action='store_true')
    parser.add_argument('--load-combined-queue',action='store_true')
    args=parser.parse_args()
    from basil_core.dirichlet_horizon_study import (BASE_STUDY,EVIDENCE,RECEIPT,
        configurations,checksum,inspect_completed,execution_fingerprints,study_name,validate_evidence)
    if args.combine:
        from gui.services.queue_service import QueueService
        from gui.services.staged_queue_service import validate_queue_for_loading
        from gui.state.experiment_state import to_persisted
        if (ROOT/'gui/.queue_active.json').exists():raise ValueError('Do not modify an active queue.')
        combined=[]
        for rounds in (100,500):
            combined.extend(json.loads((ROOT/f'gui/queues/noniid_dirichlet_prior_abcd_{rounds}r_gpu.json').read_text()))
        target=ROOT/'gui/queues/noniid_dirichlet_prior_abcd_100r_then_500r_gpu.json'
        save_new(target,combined)
        prepared=QueueService(target).load();validate_queue_for_loading(prepared.entries)
        if args.load_combined_queue:
            service=QueueService(ROOT/'gui/queue_state.json');current=service.load()
            existing=[to_persisted(e.config) for e in current.entries]
            if existing not in ([],combined[:4],combined):
                raise ValueError('Existing queue is not the matching four-run prefix or combined queue; do not overwrite it.')
            if len(current.entries)==4:
                current.entries.extend(prepared.entries[4:]);prepared=current
            elif len(current.entries)==8:prepared=current
            QueueService.describe_recovery(prepared,ROOT);service.save(prepared)
        print('Prepared eight idle entries: A/B/C/D 100r → validation → A/B/C/D 500r with W&B. No runs started.')
        return
    if args.validate_100:
        receipt=inspect_completed(ROOT/'newResults/nonIID'/study_name(100),expected_rounds=100)
        cfg=json.loads((ROOT/f'gui/configs/nonIID/dirichlet_prior_100r/A_dirichlet_prior_100r.json').read_text())
        validate_evidence(cfg)
        receipt.update(baseEvidenceSha256=checksum(EVIDENCE),implementationSha256=execution_fingerprints(cfg))
        save_new(RECEIPT,receipt)
        print('All four 100-round runs passed protocol/numerical validation. The 500-round preset is unlocked; no runs started.')
        return
    if (ROOT/'gui/.queue_active.json').exists():raise ValueError('Do not replace an active queue.')
    evidence=inspect_completed(ROOT/'newResults/nonIID'/BASE_STUDY,expected_rounds=30)
    template=configurations(evidence['partitionHash'],evidence['initialModelHash'],'0'*64,100)[0]
    evidence['implementationSha256']=execution_fingerprints(template)
    # Pure mathematical worker code must remain identical to the executed 30r
    # source. Only routing/validation is extended in dirichlet_calibration.py.
    import ast
    old=json.loads((ROOT/f'newResults/nonIID/{BASE_STUDY}/A/run.json').read_text())
    for source in ('basil_core/models.py','basil_core/attacks.py','basil_core/trainer.py',
            'basil_core/data/cifar.py','basil_core/research_protocol.py','basil_core/iid_study.py',
            'basil_core/noniid_study.py','basil_core/dirichlet_study.py','basil_core/research_audit.py',
            'basil_core/iid_runtime.py','basil_core/memory_safety.py','basil_core/protocol_compatibility.py'):
        if checksum(ROOT/source)!=old['implementationSha256'][source]:
            raise ValueError(f'Core training/threat source differs from completed confirmation: {source}')
    name='basil_core/dirichlet_calibration.py'
    archived=ROOT/f'newResults/nonIID/{BASE_STUDY}/A/source_snapshots'/old['implementationSha256'][name]
    def math_nodes(text):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(text).body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in
            {'calibration_offsets','calibrated_anchor_gradients','DirichletCalibrationWorker'}}
    if math_nodes(archived.read_text())!=math_nodes((ROOT/name).read_text()):
        raise ValueError('Loss mathematics or training worker changed from the completed confirmation.')
    evidence['unchangedMathematicalWorkerVerified']=True
    evidence['unchangedCoreTrainingAndThreatSourcesVerified']=True
    save_new(EVIDENCE,evidence)
    queues={}
    for rounds in (100,500):
        cfgs=configurations(evidence['partitionHash'],evidence['initialModelHash'],checksum(EVIDENCE),rounds)
        for cfg in cfgs:
            if (ROOT/cfg['resultDirectory']).exists():raise ValueError('Fresh preparation must preserve existing evidence.')
            save_new(ROOT/f"gui/configs/nonIID/dirichlet_prior_{rounds}r/{cfg['conditionId']}_dirichlet_prior_{rounds}r.json",cfg)
        preset=ROOT/f'gui/queues/noniid_dirichlet_prior_abcd_{rounds}r_gpu.json'
        save_new(preset,cfgs);queues[rounds]=cfgs
    if args.load_100_queue:
        from gui.services.queue_service import QueueService
        from gui.state.queue_state import QueueStatus
        service=QueueService(ROOT/'gui/queue_state.json')
        current=service.load()
        if current.entries:raise ValueError('Existing queue is not empty; load the new preset explicitly in the GUI.')
        prepared=QueueService(ROOT/'gui/queues/noniid_dirichlet_prior_abcd_100r_gpu.json').load()
        for cfg in queues[100]:validate_evidence(cfg)
        assert all(e.status is QueueStatus.PENDING for e in prepared.entries)
        service.save(prepared)
    print('Prepared A/B/C/D 100-round local-only queue and gated 500-round W&B preset. Research runs NOT started.')


if __name__=='__main__':main()
