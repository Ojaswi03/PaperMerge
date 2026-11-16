"""Versioned execution optimizations; original scientific files stay frozen."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import zlib

REVISION='fedsharp_execution_v1'
ROOT=Path(__file__).resolve().parents[1]
FILES=('basil_core/fedsharp_runtime_optimization.py',
       'scripts/optimized_fedsharp_worker.py')


def source():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


def validate(policy):
    if policy.get('revision')!=REVISION or policy.get('source')!=source():
        raise ValueError('Frozen optimization implementation differs.')


class HistoryCache:
    """Encode committed records once; callers must not edit committed history.

    The protocol compacts each record before checkpointing and never subsequently
    edits it. Replaced lists, shortened histories or replaced record objects
    invalidate the cache. Encoded bytes never depend on object identity.
    """
    def __init__(self):
        self.records=None;self.identities=[];self.encoded=[]

    def render(self,records):
        if self.records is not records or len(records)<len(self.identities) or any(
                id(r)!=identity for r,identity in zip(records,self.identities)):
            self.records=records;self.identities=[];self.encoded=[]
        for record in records[len(self.encoded):]:
            self.encoded.append(zlib.compress(json.dumps(record,sort_keys=True,allow_nan=False).encode(),1))
            self.identities.append(id(record))
        return b'['+b', '.join(zlib.decompress(value) for value in self.encoded)+b']'


class CheckpointWriter:
    def __init__(self,provenance):
        self.history=HistoryCache();self.provenance=provenance

    def __call__(self,path,state,contract):
        import numpy as np
        path=Path(path);temporary=path.with_suffix('.tmp.npz')
        n,d=state['nodeParams'].shape
        references=np.zeros((n,d),np.float32) if state['references'] else np.empty((0,d),np.float32)
        meta={}
        for node,row in state['references'].items():
            references[int(node)]=row['params']
            meta[str(node)]={k:v for k,v in row.items() if k!='params'}
        metadata=dict(contract=contract,cursor=state['cursor'],references=meta,
            telemetry=None,roundMetrics=state['roundMetrics'],elapsedSeconds=state['elapsedSeconds'],
            executionOptimization=self.provenance)
        raw=b'{'+b', '.join(json.dumps(key).encode()+b': '+(
            self.history.render(state['telemetry']) if key=='telemetry'
            else json.dumps(value,sort_keys=True,allow_nan=False).encode())
            for key,value in sorted(metadata.items()))+b'}'
        with temporary.open('wb') as stream:
            np.savez(stream,nodeParams=state['nodeParams'],outbound=state['outbound'],
                senderRounds=state['senderRounds'],referenceParams=references,metadata=np.frombuffer(raw,np.uint8))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)


class RoundSummary:
    def __init__(self,original):
        self.original=original;self.records=None;self.nodes=None;self.identities=[];self.complete=[]

    def __call__(self,records,nodes):
        if nodes<1:raise ValueError('Positive client count required.')
        if self.records is not records or self.nodes!=nodes or len(records)<len(self.identities) or any(
                id(r)!=identity for r,identity in zip(records,self.identities)):
            self.records=records;self.nodes=nodes;self.identities=[];self.complete=[]
        partial=[]
        for start in range(len(self.complete)*nodes,len(records),nodes):
            block=records[start:start+nodes];adjusted=[]
            for offset,row in enumerate(block):
                i=start+offset
                if (row['round'],row['node'],row['activation'])!=(i//nodes,i%nodes,i+1):
                    raise ValueError('Incomplete or unordered activation telemetry.')
                adjusted.append(dict(row,round=0,activation=offset+1))
            summary=self.original(adjusted,nodes)[0];summary['displayRound']=start//nodes+1
            if len(block)==nodes:
                self.complete.append(summary);self.identities.extend(map(id,block))
            else:partial=[summary]
        # The JSON writer only reads these dictionaries; return a new list.
        return self.complete+partial


def install_zero_step(worker):
    """Install a first-order branch, preserving all five scalar diagnostics."""
    import tensorflow as tf
    if not hasattr(worker,'_training_prior') or worker.cfg['proximalMu']!=1.:
        return False
    original=worker.step
    @tf.function(reduce_retracing=True)
    def step(x,y,anchor,eta,coefficient,mu):
        def zero():
            variables=worker.model.trainable_variables
            with tf.GradientTape() as tape:
                ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y,logits=worker.model(x,training=True)+worker._training_prior))
            base=tape.gradient(ce,variables)
            norm=tf.add_n([tf.reduce_sum(g*g) for g in base])
            prox=mu*.5*tf.add_n([tf.reduce_sum((v-a)**2) for v,a in zip(variables,anchor)])
            gradients=[g+mu*(v-a) for g,v,a in zip(base,variables,anchor)]
            values=(ce,norm,ce+prox,prox,tf.linalg.global_norm(gradients))
            for value in [*gradients,*values]:tf.debugging.assert_all_finite(value,'Nonfinite optimized objective')
            for variable,gradient in zip(variables,gradients):variable.assign_sub(eta*gradient)
            return values
        return tf.cond(tf.equal(coefficient,0.),zero,lambda:original(x,y,anchor,eta,coefficient,mu))
    worker.step=step
    return True


def qualify_zero_step(view,configure,device):
    """Exact device-local five-step comparisons; disable the branch on mismatch."""
    import numpy as np
    if view['comparisonCase']['addon']!='cart':return dict(enabled=False,reason='not a prior-adjusted worker')
    expected=configure(view,device);actual=configure(view,device)
    if not install_zero_step(actual):return dict(enabled=False,reason='unsupported objective')
    cfg=expected.cfg;shape=(28,28,1) if cfg['dataset']=='mnist' else (32,32,3)
    rng=np.random.default_rng(731);size=max(cfg['batchSize'],40)
    x=rng.random((size,*shape),dtype=np.float32);y=np.arange(size,dtype=np.int32)%cfg['classes']
    # Nonuniform prior and two successive updates per branch.
    y[:size//3]=0;batch=np.arange(min(cfg['batchSize'],size))
    checks=[]
    for coefficient in (0.,.003):
        left=expected.export();right=left.copy()
        for r in (120,350):
            for worker in (expected,actual):
                worker.admitted_route='pool';worker.admitted_has_channel=True
            left,a=expected.train(left,x,y,r,0,coefficient,batch)
            right,b=actual.train(right,x,y,r,0,coefficient,batch)
            exact=bool(np.array_equal(left,right) and a==b)
            checks.append(dict(coefficient=coefficient,round=r,exact=exact))
            if not exact:return dict(enabled=False,reason='device comparison differs; original step retained',checks=checks)
    return dict(enabled=True,reason='all device comparisons exact',checks=checks)


@contextmanager
def installed(root,view,policy):
    """Explicit adapter plus hash-pinned provenance in every row/checkpoint."""
    validate(policy)
    from functools import lru_cache
    import numpy as np
    from basil_core import cart_comparison_study as study,cart_campaign_io as io
    from basil_core import sharp_protocol as protocol,cart_mechanism_audit as audit,sharp_cart
    import basil_core.cart_prior_neutral_routing as routing
    import basil_core.cart_prior_neutral_commit as commit
    import basil_core.sharp_cart_v2 as repair
    from scripts import run_cart_v2_study
    from contextlib import ExitStack
    original_configure=study.configure_worker;original_write=io.write_json;original_load=protocol.load_state
    original_probes=sharp_cart.probes
    provenance=dict(revision=policy['revision'],source=policy['source'],
        transitionId=policy['transitionId'],features=['fixed_probes','cached_checkpoint_json','incremental_round_summaries'])
    writer=CheckpointWriter(provenance)
    @lru_cache(maxsize=128)
    def cached(raw,dtype,seed,node,limit):
        return original_probes(np.frombuffer(raw,dtype=np.dtype(dtype)),seed,node,limit)
    def probes(labels,seed,node,limit=16):
        labels=np.asarray(labels);fit,check=cached(labels.tobytes(),labels.dtype.str,seed,node,limit)
        return fit.copy(),check.copy()
    def configure(case,device='GPU'):
        qualification=qualify_zero_step(case,original_configure,device)
        worker=original_configure(case,device)
        if qualification['enabled']:
            original_step=worker.step;install_zero_step(worker);fast_step=worker.step
            original_train=worker.train;checked=False
            def train(flat,x,y,round_id,node,coefficient,first_batch,should_stop=None):
                nonlocal checked
                local=(getattr(worker,'admitted_route',None)=='own'
                    or not getattr(worker,'admitted_has_channel',True))
                if checked or (coefficient!=0. and not local):
                    return original_train(flat,x,y,round_id,node,coefficient,first_batch,should_stop)
                # Qualify once on the real resumed model and local batch too.
                # Both proposals start from the same flat; only one is committed.
                candidate,candidate_row=original_train(flat,x,y,round_id,node,coefficient,first_batch,should_stop)
                worker.step=original_step
                expected,expected_row=original_train(flat,x,y,round_id,node,coefficient,first_batch,should_stop)
                exact=bool(np.array_equal(candidate,expected) and candidate_row==expected_row)
                checked=True
                if exact:worker.step=fast_step
                provenance['firstRealZeroCoefficientCheck']=dict(round=round_id,node=node,exact=exact,
                    fastPathRetained=exact,extraProposalsCommitted=0)
                original_write(Path(root)/view['runId']/'optimization.json',provenance)
                return expected,expected_row
            worker.train=train
        print('Zero-coefficient optimization: '+qualification['reason'],flush=True)
        provenance['zeroCoefficientQualification']=qualification
        provenance['runtime']=run_cart_v2_study.runtime(__import__('tensorflow'))
        provenance['runtime']['device']='/'+device+':0'
        original_write(Path(root)/view['runId']/'optimization.json',provenance)
        return worker
    def write(path,value):
        if Path(path)==Path(root)/view['runId']/'run.json':
            value.setdefault('runtimeAdapter',{})['executionOptimization']=provenance
        return original_write(path,value)
    def load(path,contract,dimension):
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(archive['metadata'].tobytes())
        prior=metadata.get('executionOptimization')
        if prior and (prior.get('revision')!=policy['revision'] or prior.get('source')!=policy['source']):
            raise ValueError('Checkpoint was written by a different optimization.')
        if 'startedCursor' not in provenance:
            same=prior if prior and prior.get('transitionId')==policy['transitionId'] else {}
            provenance['startedCursor']=same.get('startedCursor',metadata['cursor'])
            provenance['startedElapsedSeconds']=same.get('startedElapsedSeconds',metadata['elapsedSeconds'])
            original_write(Path(root)/view['runId']/'optimization.json',provenance)
        return original_load(path,contract,dimension)
    with ExitStack() as stack:
        for module in (sharp_cart,routing,commit,repair):stack.enter_context(patch.object(module,'probes',probes))
        stack.enter_context(patch.object(study,'configure_worker',configure))
        stack.enter_context(patch.object(io,'write_json',write))
        stack.enter_context(patch.object(protocol,'save_state',writer))
        stack.enter_context(patch.object(protocol,'load_state',load))
        stack.enter_context(patch.object(audit,'telemetry_summary',RoundSummary(audit.telemetry_summary)))
        yield provenance
