#!/usr/bin/env python3
"""Read-only numerical/protocol audit of the completed single-traversal diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(directory):
    import hashlib
    import numpy as np
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from basil_core.replicated_verification import ComputeRequest, ComputeResult, digest_json, vote_message
    from basil_core.research_protocol import keyed_rng, params_hash
    from basil_core.replica_wire import unpack_arrays
    from basil_core.verified_cart_runner import fingerprints
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    progress = json.loads((directory/'progress.json').read_text())
    assert progress['status']=='completed' and progress['cursor']==[1,0,0] and progress['steps']==500
    envelope = json.loads((directory/progress['checkpoint']).read_text()); state = envelope['state']
    assert digest_json(state)==envelope['digest']
    assert manifest['sourceHashes']==fingerprints()==state['config']['sourceHashes']
    assert manifest['scriptSha256']==hashlib.sha256((ROOT/'scripts/run_verified_cart_learning.py').read_bytes()).hexdigest()
    assert state['config']['dataRoot']==manifest['dataRoot'] and not state['config']['synthetic']
    labels = np.load(ROOT/'experiments/cache/cifar10/y_train_int32.npy', allow_pickle=False)
    assert labels.shape==(50000,) and np.array_equal(np.bincount(labels,minlength=10),[5000]*10)
    records = state['records']; assert len(records)==500
    assert [r['samples'] for r in records]==([512]*9+[392])*50
    assert [(r['round'],r['node'],r['step']) for r in records]==[(0,n,s) for n in range(10) for s in range(50)]
    sessions={s['hash']:s['publicKeys'] for s in state['sessions']}
    votes=0; certificates=0
    def certificate(proof, request):
        nonlocal votes,certificates
        assert proof['requestHash']==request.digest and proof['requiredMatches']==5
        assert proof['committee']==list(request.committee)
        signers=proof['matchingSigners']; assert len(set(signers))>=5
        keys=sessions[proof['membershipHash']]
        for signer in signers:
            assert signer in request.committee
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(keys[str(signer)])).verify(
                bytes.fromhex(proof['signatures'][str(signer)]),
                vote_message(signer,request.digest,proof['resultHash']))
            votes+=1
        certificates+=1
    for r in records:
        node,step=r['node'],r['step']; epoch,batch=divmod(step,10)
        assert np.isfinite(r['objective']) and np.isfinite(r['crossEntropy'])
        assert r['ebmCoefficient']==r['ebmPenalty']==r['ebmCorrectionNorm']==0.
        assert r['learningRate']==.05 and not r['modelAveraging']
        assert set(map(int,r['certificates']))==set(range(node))
        for owner in range(node):
            pool=np.flatnonzero(labels==owner)
            ids=keyed_rng(2025,'cart_owner_gradient_samples',0,node,epoch,batch,owner).choice(pool,16,replace=False)
            key=keyed_rng(2025,'cart_owner_gradient_augmentation',0,node,epoch,batch,owner).integers(0,2**30,size=2,dtype=np.int32)
            request=ComputeRequest(manifest['config']['runId'],node,owner,0,step,'gradient',r['inputHash'],
                manifest['dataRoot'],tuple(map(int,ids)),tuple(map(int,key)))
            certificate(r['certificates'][str(owner)],request)
    for n in range(10):
        group=records[n*50:(n+1)*50]
        before=next(e for e in progress['evaluations'] if e['node']==n and e['epoch']==0)
        assert group[0]['inputHash']==before['parameterHash']
        assert all(r['anchorHash']==before['parameterHash'] for r in group)
        assert all(a['outputHash']==b['inputHash'] for a,b in zip(group,group[1:]))
        assert group[-1]['outputHash']==state['nodeParams'][n]
    # Each selection checkpoint also records the actual five candidate scores.
    selected_nodes=set()
    for path in sorted(directory.glob('checkpoint_*.json')):
        saved=json.loads(path.read_text()); checkpoint=saved['state']
        selection=checkpoint['selection']; n=checkpoint['node']
        if checkpoint['step']!=0 or selection is None or n in selected_nodes: continue
        assert digest_json(checkpoint)==saved['digest']
        candidates=selection['records']
        assert {r['senderId'] for r in candidates}=={(n-d)%10 for d in range(1,6)}
        for r in candidates:
            assert r['snapshotRound']==(0 if r['senderId']<n else -1)
            assert set(map(int,r['ownerLosses']))==set(range(n+1))
            assert abs(r['balancedTrainingLoss']-np.mean(list(r['ownerLosses'].values())))<1e-12
            for owner in range(n+1):
                assert r['certificates'][str(owner)]['resultHash']==ComputeResult.from_values(r['ownerLosses'][str(owner)]).digest
                request=ComputeRequest(manifest['config']['runId'],n,owner,0,0,'selection_ce',r['parameterHash'],
                    manifest['dataRoot'],tuple(manifest['selectionIndices'][str(owner)]))
                certificate(r['certificates'][str(owner)],request)
        minimum=min(r['balancedTrainingLoss'] for r in candidates)
        best=min((r for r in candidates if r['balancedTrainingLoss']<=minimum+1e-8),key=lambda r:(n-r['senderId'])%10)
        assert selection['sender']==best['senderId'] and selection['hash']==best['parameterHash']
        selected_nodes.add(n)
    assert selected_nodes==set(range(10))
    assert len(progress['evaluations'])==60
    for e in progress['evaluations']:
        confusion=np.asarray(e['confusion'])
        assert np.array_equal(confusion.sum(axis=1),[1000]*10)
        assert e['accuracy']==float(np.trace(confusion)/10000)
        np.testing.assert_allclose(e['perClass'],np.diag(confusion)/1000,atol=3e-8,rtol=0)
    for digest,packed in state['weights'].items():
        params=unpack_arrays(packed)
        assert params_hash(params)==digest and all(np.isfinite(p).all() for p in params)
    assert {s['sender'] for s in state['memories'][0]}=={5,6,7,8,9}
    assert all(s['round']==0 for s in state['memories'][0])
    for snapshot in state['memories'][0]:assert snapshot['params']==state['nodeParams'][snapshot['sender']]
    peaks=[]
    for h in progress['resourceHistory']:
        for r in h['replicas'].values():
            assert r['events']['oom']==r['events']['oom_kill']==0
            assert r['peakBytes']<=2*1024**3
            peaks.append(r['peakBytes'])
    return dict(status='passed',steps=500,activations=10,localEpochs=50,evaluations=60,
        verifiedCertificates=certificates,verifiedSignatures=votes,maxReplicaPeakBytes=max(peaks),
        noNoiseOrEbm=True,strictHandoff=True,frozenAnchors=True,selectionMinimum=True,
        nodeZeroNextRoundMemory=True,allParametersFinite=True,sourceHashesMatch=True)


if __name__=='__main__':
    import argparse
    from scripts.run_verified_cart_learning import OUTPUT, atomic_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true',help='Save derived audit in the new diagnostic directory.')
    args=parser.parse_args()
    result=verify(OUTPUT)
    if args.save: atomic_json(OUTPUT/'EVIDENCE_AUDIT.json',result)
    print(json.dumps(result,indent=2))
