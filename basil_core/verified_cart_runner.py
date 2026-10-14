"""Opt-in checkpointed CART step coordinator; not exposed in the research queue.

The coordinator is trusted; adversarial nodes may falsify messages, not modify
its memory. A helper certificate does not turn a malicious SGD proposal honest.
No background execution, test data, W&B or horizon is selected by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .replica_wire import pack_arrays,unpack_arrays
from .replicated_verification import ComputeRequest,digest_json

REVISION='cart_isolated_verified_steps_v2'
ROOT=Path(__file__).resolve().parents[1]


def fingerprints():
    paths=('basil_core/verified_cart_runner.py','basil_core/isolated_replicas.py',
        'basil_core/replica_wire.py','basil_core/sandbox_replica.py',
        'basil_core/replica_resources.py',
        'basil_core/replicated_verification.py','basil_core/replicated_training_service.py',
        'basil_core/cart_owner_ebm.py','basil_core/cart_owner_gradients.py',
        'basil_core/research_protocol.py','basil_core/research_audit.py','basil_core/models.py',
        'basil_core/cart_owner_protocol.py','basil_core/attacks.py')
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


class VerifiedCartRunner:
    def __init__(self,model,reference,service,*,run_id,seed=2025,ebm=False,sigma_e=0.,attacker_ids=(),
                 trainer_failure_policy='abort',device='CPU'):
        import numpy as np
        import tensorflow as tf
        from .research_protocol import initialize_memories,params_hash
        if not run_id or type(seed) is not int:
            raise ValueError('Explicit run identity and integer seed required.')
        if device not in ('CPU','GPU') or any(device+':' not in w.handle.device for w in model.trainable_variables):
            raise ValueError('Explicit coordinator device must match its actual model placement.')
        if trainer_failure_policy not in ('abort','verified_coordinator_recompute'):
            raise ValueError('Explicit supported trainer refusal policy required.')
        if device=='GPU':
            from .iid_runtime import runtime_metadata
            runtime_metadata(tf,{'executionDevice':'GPU'})  # Enforces cap and float32 policy.
        if type(ebm) is not bool or sigma_e not in (0.,.01) or (ebm and sigma_e!=.01):
            raise ValueError('Explicit CE/full EBM and approved absolute sigma required.')
        if tuple(attacker_ids) not in ((),(0,1,5,7)):
            raise ValueError('Canonical actual attackers required.')
        self.model=model;self.reference=reference;self.service=service
        self.config=dict(revision=REVISION,runId=run_id,seed=seed,ebm=ebm,sigma_e=sigma_e,
            attackerIds=list(attacker_ids),mu=.01,epochs=5,batchSize=512,lr=.05,
            dataRoot=reference.root,synthetic=reference.synthetic,device=device,helperDevice='CPU',
            trainerFailurePolicy=trainer_failure_policy,
            researchReady=False,sourceHashes=fingerprints())
        self.params=[w.numpy().astype(np.float32) for w in model.trainable_variables]
        sequential=getattr(model,'model',model)
        if not isinstance(sequential,tf.keras.Sequential):
            raise ValueError('Only sequential stateless model topology is supported.')
        def semantic(value):
            # Keras assigns fresh cosmetic layer names after every build. They
            # do not change a Sequential computation, unlike activation/config.
            if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k not in ('name','shared_object_id')}
            if isinstance(value,list):return [semantic(v) for v in value]
            return value
        self.config.update(initialHash=params_hash(self.params),tensorflowVersion=tf.__version__,
            modelSemantics=semantic(json.loads(sequential.to_json())),
            runtime=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(),
                inter=tf.config.threading.get_inter_op_parallelism_threads(),
                oneDnn=os.environ.get('TF_ENABLE_ONEDNN_OPTS'),deterministic=os.environ.get('TF_DETERMINISTIC_OPS')))
        self.anchor=None;self.round=0;self.node=0;self.step=0
        self.memories=initialize_memories(self.params);self.records=[];self.sessions=[]
        self.node_params=[[p.copy() for p in self.params] for _ in range(10)]
        self.selection=None
        self._record_membership()

    def _record_membership(self):
        roster={str(i):k.hex() for i,k in self.service.public_keys.items()}
        digest=digest_json(roster)
        if not self.sessions or self.sessions[-1]['hash']!=digest:
            self.sessions.append(dict(hash=digest,publicKeys=roster,
                cursor=[self.round,self.node,self.step],reason='trusted local bootstrap / new process session'))

    def select(self,probes):
        from .replicated_training_service import verified_balanced_selection
        from .research_protocol import copy_params,params_hash
        if self.anchor is not None or self.step:raise ValueError('Selection only at activation start.')
        chosen,records=verified_balanced_selection(self.service,self.memories[self.node].candidates(),
            reference=self.reference,run_id=self.config['runId'],receiver=self.node,round_id=self.round,probes=probes)
        self.params=copy_params(chosen.params);self.anchor=copy_params(chosen.params)
        self.selection=dict(sender=chosen.sender_id,round=chosen.round_id,hash=params_hash(chosen.params),records=records)
        return self.selection

    def next_batch(self):
        import numpy as np
        from .research_protocol import keyed_rng
        pool=np.flatnonzero(self.reference.labels==self.node)
        if len(pool)!=5000:raise ValueError('Full local 5000-example dataset required for scheduled training.')
        epoch,batch=divmod(self.step,10)
        if epoch>=5:raise ValueError('Activation already has fifty complete steps.')
        seed=self.config['seed']
        order=keyed_rng(seed,'data_order',self.round,self.node,epoch).permutation(5000)
        ids=pool[order[batch*512:(batch+1)*512]]
        key=keyed_rng(seed,'augmentation',self.round,self.node,epoch,batch).integers(0,2**30,size=2,dtype=np.int32)
        owners=range(self.node) if self.round==0 else (c for c in range(10) if c!=self.node)
        helpers={}
        for owner in owners:
            owner_pool=np.flatnonzero(self.reference.labels==owner)
            hi=keyed_rng(seed,'cart_owner_gradient_samples',self.round,self.node,epoch,batch,owner).choice(owner_pool,16,replace=False)
            hk=keyed_rng(seed,'cart_owner_gradient_augmentation',self.round,self.node,epoch,batch,owner).integers(0,2**30,size=2,dtype=np.int32)
            helpers[owner]=(tuple(map(int,hi)),tuple(map(int,hk)))
        return tuple(map(int,ids)),tuple(map(int,key)),helpers

    def advance(self,*,proposed_params=None,proposal_failure=None,unit_batch=None):
        """One verified honest step, or reject a mismatched active-node proposal.

        unit_batch is solely for synthetic contract fixtures, never real CIFAR.
        Models are restored on any failure; no progress is committed implicitly.
        """
        import numpy as np
        import tensorflow as tf
        from .research_protocol import params_hash,basil_round_learning_rate,copy_params
        from .research_audit import augment_batch
        from .cart_owner_ebm import owner_objective_gradients
        if self.anchor is None or not 0<=self.step<50:raise ValueError('Selected frozen anchor and unfinished activation required.')
        if proposal_failure not in (None,'timeout','exit','refused','invalid_message'):
            raise ValueError('Explicit bounded-transport failure required.')
        if proposed_params is not None and proposal_failure is not None:
            raise ValueError('Cannot have both a proposal and a transport failure.')
        if proposal_failure and self.config['trainerFailurePolicy']=='abort':
            raise ValueError('Active trainer failed; configured abort policy retained.')
        if unit_batch is not None and not self.reference.synthetic:raise ValueError('Unit batches require explicitly synthetic evidence.')
        ids,key,helpers=unit_batch if unit_batch is not None else self.next_batch()
        if set(helpers)!=(set(range(self.node)) if self.round==0 else set(range(10))-{self.node}):
            raise ValueError('Exact already-available class coverage required.')
        if any(type(i) is not int or not 0<=i<len(self.reference.labels) or self.reference.labels[i]!=self.node for i in ids):
            raise ValueError('Only receiver-local TRAIN indices permitted.')
        before=params_hash(self.params);responses=[];proofs={}
        try:
            for owner,(indices,aug) in helpers.items():
                req=ComputeRequest(self.config['runId'],self.node,owner,self.round,self.step,'gradient',before,
                    self.reference.root,tuple(indices),tuple(aug))
                response,proof=self.service.gradient_response(req,self.params)
                responses.append(response);proofs[owner]=proof
            for w,p in zip(self.model.trainable_variables,self.params):w.assign(p)
            with tf.device('/CPU:0'):images=augment_batch(self.reference.images[list(ids)],np.asarray(key,np.int32))
            grads,values,meta=owner_objective_gradients(self.model,images,self.reference.labels[list(ids)],
                self.anchor,responses,receiver=self.node,round_id=self.round,optimizer_step=self.step,
                mu=.01,sigma_e=self.config['sigma_e'],ebm=self.config['ebm'])
            lr=basil_round_learning_rate(.05,self.round)
            # No momentum slots: resetting this stateless optimizer per step is
            # algebraically the same update as reset-per-activation SGD(momentum=0).
            tf.keras.optimizers.SGD(lr,momentum=0.).apply_gradients(zip(grads,self.model.trainable_variables))
            updated=[w.numpy() for w in self.model.trainable_variables]
            if any(not np.isfinite(p).all() for p in updated):raise FloatingPointError('Nonfinite SGD state.')
            rejection=proposal_failure
            if proposed_params is not None and params_hash(proposed_params)!=params_hash(updated):
                rejection='incorrect_update'
            if rejection and self.config['trainerFailurePolicy']=='abort':
                raise ValueError('Active-node SGD proposal differs from verified computation; no progress committed.')
            record=dict(round=self.round,node=self.node,step=self.step,epoch=self.step//10+1,batch=self.step%10,
                samples=len(ids),inputHash=before,outputHash=params_hash(updated),anchorHash=params_hash(self.anchor),
                objective=float(values[2]),learningRate=lr,certificates=proofs,membershipHash=self.sessions[-1]['hash'],**meta)
            record.update(trainerFailure=rejection,recoveredByCoordinator=bool(rejection),
                trainerFailurePolicy=self.config['trainerFailurePolicy'],
                recoveryMeaning='same step recomputed; no node/epoch skip' if rejection else None)
            self.params=copy_params(updated);self.records.append(record);self.step+=1
            return record
        except BaseException:
            for w,p in zip(self.model.trainable_variables,self.params):w.assign(p)
            raise

    def finish_activation(self):
        from .cart_owner_protocol import snapshot_links
        from .research_protocol import Snapshot
        if self.step!=50:raise ValueError('Exactly fifty full-data SGD steps required before outbound delivery.')
        recent=self.records[-50:]
        if len(recent)!=50 or [r['samples'] for r in recent]!=([512]*9+[392])*5:
            raise ValueError('Five full local epochs and final partial batches required.')
        from .research_protocol import params_hash
        if (self.anchor is None or self.selection is None or
                [(r['round'],r['node'],r['step']) for r in recent]!=[(self.round,self.node,i) for i in range(50)] or
                recent[0]['inputHash']!=params_hash(self.anchor) or
                recent[-1]['outputHash']!=params_hash(self.params) or
                any(a['outputHash']!=b['inputHash'] for a,b in zip(recent,recent[1:]))):
            raise ValueError('Activation update lineage is incomplete or inconsistent.')
        links=snapshot_links(self.params,seed=self.config['seed'],round_id=self.round,sender=self.node,
            attacker_ids=self.config['attackerIds'],sigma_e=self.config['sigma_e'])
        self.node_params[self.node]=[p.copy() for p in self.params]
        for link in links:
            self.memories[link['receiver']].receive(Snapshot(self.node,self.round,link['params'],channel_metadata=link['noise']))
        self.node+=1
        if self.node==10:self.node=0;self.round+=1
        self.step=0;self.anchor=None;self.selection=None
        return links

    def save(self,path):
        from .research_protocol import params_hash
        path=Path(path)
        weights={}
        def pack(params):
            digest=params_hash(params)
            if digest not in weights:weights[digest]=pack_arrays(params)
            return digest
        state=dict(config=self.config,round=self.round,node=self.node,step=self.step,
            params=pack(self.params),anchor=pack(self.anchor) if self.anchor is not None else None,
            selection=self.selection,records=self.records,sessions=self.sessions,
            nodeParams=[pack(p) for p in self.node_params],
            memories=[[dict(sender=s.sender_id,round=s.round_id,params=pack(s.params),channel=s.channel_metadata)
                for s in memory.candidates()] for memory in self.memories],weights=weights)
        envelope=dict(format=REVISION,digest=digest_json(state),state=state)
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('format')!=REVISION:raise ValueError('Refusing to overwrite an unrelated file.')
        path.parent.mkdir(parents=True,exist_ok=True)
        encoded_size=sum(len(chunk.encode()) for chunk in json.JSONEncoder(separators=(',',':'),allow_nan=False).iterencode(envelope))
        if encoded_size>256*1024**2:
            raise OSError('Checkpoint exceeds 256 MiB diagnostic storage budget; keep the previous checkpoint.')
        if shutil.disk_usage(path.parent).free<encoded_size+1024**3:
            raise OSError('Checkpoint requires its temporary file plus 1 GiB free-disk headroom.')
        fd,name=tempfile.mkstemp(prefix='.verified-step-',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as handle:
                json.dump(envelope,handle,separators=(',',':'),allow_nan=False);handle.flush();os.fsync(handle.fileno())
            os.replace(name,path)
            directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if os.path.exists(name):os.unlink(name)

    def restore(self,path):
        from .research_protocol import params_hash,RollingMemory,Snapshot
        envelope=json.loads(Path(path).read_text());s=envelope['state']
        if envelope.get('format')!=REVISION or digest_json(s)!=envelope['digest'] or s['config']!=self.config:
            raise ValueError('Checkpoint integrity/configuration/source mismatch.')
        def unpack(digest):
            params=unpack_arrays(s['weights'][digest])
            if params_hash(params)!=digest:raise ValueError('Checkpoint tensor hash mismatch.')
            return [p.copy() for p in params]
        params=unpack(s['params']);anchor=unpack(s['anchor']) if s['anchor'] else None
        node_params=[unpack(d) for d in s['nodeParams']]
        if len(node_params)!=10:raise ValueError('Ten logical trained-node states required.')
        memories=[]
        for node,snapshots in enumerate(s['memories']):
            memory=RollingMemory(node,10,5)
            for snap in snapshots:memory.receive(Snapshot(snap['sender'],snap['round'],unpack(snap['params']),channel_metadata=snap['channel']))
            if len(memory.candidates())!=5:raise ValueError('Incomplete BASIL memory.')
            memories.append(memory)
        if len(memories)!=10 or not 0<=s['step']<=50 or not 0<=s['node']<10 or s['round']<0:
            raise ValueError('Invalid checkpoint cursor.')
        self.params=params;self.anchor=anchor;self.memories=memories;self.node_params=node_params
        self.round=s['round'];self.node=s['node'];self.step=s['step']
        self.records=s['records'];self.selection=s['selection'];self.sessions=s['sessions']
        self._record_membership()
        for w,p in zip(self.model.trainable_variables,self.params):w.assign(p)
