#!/usr/bin/env python3
"""Infer one official TEST image with frozen final FedSharp/FedSharp+ client models.

This is a demonstration, not a test-set accuracy estimate. No training, channel
simulation, candidate selection, prior adjustment or checkpoint writes occur.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CRITICAL_SOURCES=('basil_core/sharp_worker.py','basil_core/models.py',
    'basil_core/research_protocol.py','basil_core/sharp_protocol.py',
    'basil_core/sharp_math.py','basil_core/basil_sharp_thesis.py')
CIFAR_CLASSES=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
METHODS={'fedsharp':('sharp',None,'FedSharp'),'fedsharp_plus':('sharp','cart','FedSharp+'),
    'basil':('basil',None,'BASIL'),'basil_acds':('basil','acds','BASIL + ACDS')}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def select_runs(queue,datasets,attack,seed):
    """Choose matching cases by configuration, before reading TEST or predictions."""
    queue=Path(queue)
    master=read(queue/'continuation_plan.json');fleet=read(queue/'fleet/state.json')
    selected=[]
    for dataset in datasets:
        for addon in (None,'cart'):
            found=[]
            for item in master['items']:
                view=item['config'];case=view['comparisonCase']
                if (view['dataset'],view['partition'],view['seed'],view['attackType'],
                    case['family'],case['environment'],case['addon'])==(
                    dataset,'dirichlet',seed,attack,'sharp','joint',addon):found.append(view)
            if len(found)!=1:raise ValueError(f'Expected one {dataset} {addon or "FedSharp"} case, found {len(found)}.')
            view=found[0];entry=fleet['runs'][view['runId']]
            if entry['state']!='DONE':raise ValueError('Selected run is not DONE: '+view['runId'])
            if view['rounds']!=500 or view['nodes']!=100:raise ValueError('Expected the completed 500-round, 100-client study.')
            selected.append(dict(method='FedSharp+' if addon else 'FedSharp',view=view,
                owner=entry['owner'],output=Path(entry['location']['output'])))
    for dataset in datasets:
        pair=[s['view'] for s in selected if s['view']['dataset']==dataset]
        keys=('seed','dataset','partition','rounds','nodes','attackType','sigma','batchSize','localSteps','learningRate')
        if any(pair[0][k]!=pair[1][k] for k in keys):raise ValueError('Requested cases are not matched.')
    return selected


def select_local_runs(checkpoint_dir,datasets,attack,seed,methods=('fedsharp','fedsharp_plus')):
    """Use published experiment identities with explicitly copied local weights."""
    checkpoint_dir=Path(checkpoint_dir).resolve();configs=[];origins={}
    for host in ('hal9k','themachine'):
        root=ROOT/'docs/results'/f'{host}_fedsharp-fleet-500r-20261103'
        for cfg in read(root/'plan.json')['configs']:
            configs.append(cfg);origins[cfg['runId']]=(host,root)
    chosen=[]
    for dataset in datasets:
        for method in methods:
            family,addon,label=METHODS[method]
            found=[v for v in configs if (v['dataset'],v['partition'],v['seed'],v['attackType'],
                v['comparisonCase']['family'],v['comparisonCase']['environment'],v['comparisonCase']['addon'])==
                (dataset,'dirichlet',seed,attack,family,'joint',addon)]
            if len(found)!=1:raise ValueError('Expected one published matching case.')
            view=found[0];host,root=origins[view['runId']]
            checkpoint=checkpoint_dir/view['runId']/'checkpoint.npz'
            if not checkpoint.exists() and (checkpoint.parent/'model.npz').exists():checkpoint=checkpoint.parent/'model.npz'
            chosen.append(dict(method=label,view=view,owner=host,
                output=root,checkpoint=checkpoint,allowRuntimeDifference=True))
    return chosen


def preflight(selected):
    """Inspect every required local file before loading any test example."""
    host=socket.gethostname().split('.')[0];result=[]
    for chosen in selected:
        view=chosen['view'];run=view['runId'];root=chosen['output'];folder=root/run
        if 'checkpoint' not in chosen and chosen['owner']!=host:
            raise ValueError(f'{run} is on {chosen["owner"]}; run this command there. Its /var/tmp storage is host-local.')
        checkpoint=chosen.get('checkpoint',folder/'checkpoint.npz')
        names=[root/'plan.json',folder/'run.json',checkpoint,folder/'metrics.npz']
        missing=[str(p) for p in names if not p.is_file()]
        if missing:raise FileNotFoundError('Required frozen file missing: '+', '.join(missing))
        plan=read(root/'plan.json');row=read(folder/'run.json')
        if 'checkpoint' in chosen:
            manifest=read(root/'SHA256.json')
            for rel in ('plan.json',run+'/run.json',run+'/metrics.npz'):
                if sha(root/rel)!=manifest[rel]:raise ValueError('Published experiment identity differs: '+rel)
        if (row['status']!='COMPLETED' or row['config']!=view or row['source']!=plan['source']
                or not any(cfg==view for cfg in plan['configs'])):raise ValueError('Incomplete or mismatched case: '+run)
        if row['methodRevision']!='basil_sharp_thesis_20261029_v1':raise ValueError('Unknown study revision.')
        if row['contract']['config']['rounds']!=500:raise ValueError('Checkpoint is not the 500-round case.')
        for name,value in plan['source'].items():
            if sha(root/'source_snapshots'/name)!=value:raise ValueError('Frozen source snapshot differs: '+name)
        for name in CRITICAL_SOURCES:
            if sha(ROOT/name)!=plan['source'][name]:raise ValueError('Current inference source differs: '+name)
        from basil_core.basil_sharp_thesis import effective_config
        cfg=effective_config(view)
        if cfg!=row['contract']['config'] or row['contract']['comparisonCase']!=view['comparisonCase']:
            raise ValueError('Effective model contract differs: '+run)
        watched=names+[folder/n for n in ('mechanisms.json','continuation.json','optimization.json') if (folder/n).is_file()]
        result.append(dict(chosen,checkpoint=checkpoint,config=cfg,row=row,before={str(p):sha(p) for p in watched}))
    for dataset in {s['view']['dataset'] for s in result}:
        rows=[s['row'] for s in result if s['view']['dataset']==dataset];a=rows[0]
        for b in rows[1:]:
            if (any(a[k]!=b[k] for k in ('data','initialHash')) or
                    {k:v for k,v in a['partition'].items() if k!='acds'}!=
                    {k:v for k,v in b['partition'].items() if k!='acds'}):
                raise ValueError('Paired data, base partition or initialization differs.')
    return result


def load_final_parameters(item,dimension):
    """Check either a complete checkpoint or a compact, hash-verified copy."""
    import zipfile
    import numpy as np
    from basil_core.sharp_protocol import load_state,vector_hash
    row=item['row'];cfg=item['config'];path=item['checkpoint']
    with np.load(item['output']/item['view']['runId']/'metrics.npz',allow_pickle=False) as z:
        accuracy=z['roundNodeAccuracy'].copy()
    if path.name=='model.npz':
        with zipfile.ZipFile(path) as z:
            if set(z.namelist())!={'nodeParams.npy','model_metadata.json'}:raise ValueError('Unknown compact model format.')
            meta=json.loads(z.read('model_metadata.json'))
        if meta['contract']!=row['contract'] or meta['cursor']!=cfg['nodes']*cfg['rounds']:
            raise ValueError('Compact final model contract differs.')
        np.testing.assert_array_equal(meta['finalRoundAccuracy'],accuracy[-1])
        with np.load(path,allow_pickle=False) as z:params=z['nodeParams'].copy()
        hashes=meta['finalModelHashes']
    else:
        state=load_state(path,row['contract'],dimension)
        if state['cursor']!=cfg['nodes']*cfg['rounds'] or len(state['roundMetrics'])!=cfg['rounds']:
            raise ValueError('Incomplete final checkpoint.')
        np.testing.assert_array_equal(accuracy,np.array([v['nodeAccuracy'] for v in state['roundMetrics']]))
        final=state['telemetry'][-cfg['nodes']:]
        if [r['node'] for r in final]!=list(range(cfg['nodes'])):raise ValueError('Final activation order differs.')
        params=state['nodeParams'];hashes=[r['trainedHash'] for r in final]
    if params.shape!=(cfg['nodes'],dimension) or params.dtype!=np.float32 or not np.isfinite(params).all():
        raise ValueError('Invalid final model parameters.')
    if len(hashes)!=cfg['nodes'] or any(vector_hash(p)!=h for p,h in zip(params,hashes)):
        raise ValueError('Final trained parameter hash differs.')
    return params,accuracy


def official_image(dataset,index):
    """Read the cached raw benchmark TEST split, never the fitting/holdout cache."""
    import numpy as np
    from scripts.verify_cart_comparison_results import test_data
    x,y,identity=test_data(dataset)
    shape=(28,28,1) if dataset=='mnist' else (32,32,3)
    if (x.shape!=(10000,*shape) or y.shape!=(10000,) or x.dtype!=np.float32
            or y.dtype!=np.int32 or not np.isfinite(x).all() or x.min()<0 or x.max()>1
            or not np.array_equal(np.unique(y),np.arange(10))):
        raise ValueError('Expected the cached, full official 10000-example TEST split.')
    if not 0<=index<len(y):raise ValueError('TEST index must be between 0 and 9999.')
    image=x[index:index+1].copy();label=int(y[index])
    return image,label,dict(identity,split='official_test',samplesInSplit=len(y),
        distinctTestImagesEvaluated=1,index=index,label=label,
        labelName=str(label) if dataset=='mnist' else CIFAR_CLASSES[label],
        normalizedImageSha256=hashlib.sha256(image.tobytes()).hexdigest(),
        normalization='uint8 pixel / 255, float32; same as training; no augmentation')


def prediction(logits,label):
    """Stable natural-log cross entropy, without training-only penalties."""
    import numpy as np
    z=np.asarray(logits,np.float64)
    if z.shape!=(10,) or not np.isfinite(z).all():raise ValueError('Expected 10 finite raw logits.')
    shifted=z-z.max();logsum=float(np.log(np.exp(shifted).sum()));p=np.exp(shifted-logsum)
    chosen=int(np.argmax(z))
    return dict(predictedClass=chosen,correct=chosen==label,
        predictionConfidence=float(p[chosen]),trueClassProbability=float(p[label]),
        crossEntropyLoss=float(logsum-shifted[label]),probabilities=p.tolist(),rawLogits=z.tolist())


def infer_case(item,image,label):
    import numpy as np
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import load_state,vector_hash
    from basil_core.research_protocol import resolve_attackers
    cfg=item['config'];row=item['row'];run=item['view']['runId']
    runtime_matches=tf.__version__==row['runtime']['tensorflow'] and np.__version__==row['runtime']['numpy']
    if not runtime_matches and not item.get('allowRuntimeDifference',False):
        raise ValueError('Use the original .venv-hpc: evaluation TensorFlow/NumPy versions differ from training.')
    worker=SharpWorker(cfg,'CPU')
    initial_matches=vector_hash(worker.export())==row['initialHash']
    if not initial_matches and not item.get('allowRuntimeDifference',False):raise ValueError('Model initialization/architecture differs.')
    # Local evaluation loads every trained parameter. A fresh initializer is not
    # used for prediction; library/initializer differences are recorded explicitly.
    path=item.get('checkpoint',item['output']/run/'checkpoint.npz')
    state=load_state(path,row['contract'],worker.dimension)
    if state['cursor']!=cfg['nodes']*cfg['rounds'] or len(state['roundMetrics'])!=cfg['rounds']:
        raise ValueError('Incomplete final checkpoint: '+run)
    with np.load(item['output']/run/'metrics.npz',allow_pickle=False) as m:
        np.testing.assert_array_equal(m['roundNodeAccuracy'],np.asarray([v['nodeAccuracy'] for v in state['roundMetrics']]))
    attackers=set(resolve_attackers(cfg['seed'],cfg['actualAttackers'],cfg['nodes']))
    final=state['telemetry'][-cfg['nodes']:]
    if [r['node'] for r in final]!=list(range(cfg['nodes'])):raise ValueError('Final activation order differs.')
    results=[]
    for node,params in enumerate(state['nodeParams']):
        if vector_hash(params)!=final[node]['trainedHash']:raise ValueError('Retained model does not match its final recorded hash.')
        # Evaluate retained trained/committed parameters, never attacked outbound copies.
        result=prediction(worker.outputs(params,image)[0],label)
        result.update(clientId=node,attackerIdentityDuringTraining=node in attackers,modelSha256=vector_hash(params),
            predictedLabel=str(result['predictedClass']) if cfg['dataset']=='mnist' else CIFAR_CLASSES[result['predictedClass']])
        results.append(result)
    def summary(values):
        return dict(clients=len(values),clientsCorrect=sum(r['correct'] for r in values),
            fractionClientsCorrectOnThisImage=float(np.mean([r['correct'] for r in values])),
            meanCrossEntropyLoss=float(np.mean([r['crossEntropyLoss'] for r in values])))
    result=dict(method=item['method'],dataset=cfg['dataset'],runId=run,seed=cfg['seed'],partition=cfg['partition'],
        attackType=cfg['attackType'],sigma=cfg['sigma'],rounds=cfg['rounds'],checkpointSha256=item['before'][str(path)],
        parameterSource='retained nodeParams, raw-logit inference; no test-time prior adjustment',
        allClients=summary(results),honestClients=summary([r for r in results if not r['attackerIdentityDuringTraining']]),
        perClient=results,evaluationRuntime=dict(tensorflow=tf.__version__,numpy=np.__version__,device='CPU',
            matchesTrainingVersions=runtime_matches,initializationMatchesTraining=initial_matches),
        trainingRuntime={k:row['runtime'][k] for k in ('tensorflow','numpy')})
    del state,worker
    tf.keras.backend.clear_session()
    return result


def evaluate(selected,index,destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a fresh report folder; never overwrite a test receipt.')
    cases=preflight(selected)
    if any(destination.is_relative_to(item['output'].resolve()) for item in cases):
        raise ValueError('Save the demo in a separate folder, outside the original campaign outputs.')
    # Freeze and verify cases before reading TEST. No candidate is chosen from its prediction.
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/papermerge-mpl')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    images={};results=[]
    for item in cases:
        ds=item['config']['dataset']
        if ds not in images:images[ds]=official_image(ds,index)
        image,label,_=images[ds]
        print(f'Evaluating {ds} {item["method"]}: TEST image {index}, true label {label}',flush=True)
        results.append(infer_case(item,image,label))
    for item in cases:
        for name,digest in item['before'].items():
            if sha(Path(name))!=digest:raise ValueError('Training artifact changed during inference: '+name)
    receipt=dict(status='COMPLETED_SINGLE_IMAGE_DEMONSTRATION',purpose='one official test image per dataset, not test-set accuracy',
        createdAt=datetime.now(timezone.utc).isoformat(),trainingPerformed=False,optimizerUpdatesAdded=0,
        checkpointsModified=False,decisionsUsedTest=False,officialTestUsed=True,modelSelectionUsedTest=False,
        loss='raw-logit sparse softmax cross entropy in natural-log units; no proximal or gradient penalties',
        interpretation='Each client is scored on the same one image. The fraction of correct clients is not accuracy across independent test images.',
        testImages={k:v[2] for k,v in images.items()},results=results,evaluationProgramSha256=sha(Path(__file__)),
        testLoaderSha256=sha(ROOT/'scripts/verify_cart_comparison_results.py'))
    destination.mkdir(parents=True)
    (destination/'results.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    fields=('dataset','method','clientId','trueClass','predictedClass','predictedLabel','correct',
        'predictionConfidence','trueClassProbability','crossEntropyLoss','attackerIdentityDuringTraining')
    with (destination/'per_client.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for result in results:
            for r in result['perClient']:
                out={k:r[k] for k in fields if k in r};out.update(dataset=result['dataset'],method=result['method'],trueClass=images[result['dataset']][1]);writer.writerow(out)
    from PIL import Image
    import numpy as np
    for ds,(image,_,_) in images.items():
        pixels=np.rint(image[0]*255).astype(np.uint8)
        Image.fromarray(pixels[...,0] if ds=='mnist' else pixels).save(destination/(ds+'_test_image.png'))
    lines=['SINGLE-IMAGE DEMONSTRATION; NOT TEST-SET ACCURACY','',
        'DATASET  METHOD     TRUE LABEL  CORRECT CLIENTS  FRACTION CORRECT  MEAN CE LOSS']
    for result in results:
        s=result['allClients'];true=images[result['dataset']][2]['labelName']
        lines.append(f'{result["dataset"]:8} {result["method"]:10} {true:11} {s["clientsCorrect"]:3}/{s["clients"]:<3}          {s["fractionClientsCorrectOnThisImage"]:7.2%}         {s["meanCrossEntropyLoss"]:.6f}')
    lines.extend(['','Each image is evaluated using every final client model separately. No model averaging or ensemble voting.',
        'Loss is cross entropy on this image. Correct/incorrect for one model corresponds to 100%/0% on that single example.',
        'Training and checkpoints were not changed. See per_client.csv for individual predictions, confidence and loss.'])
    (destination/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Saved report: '+str(destination),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=Path.home()/'Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102')
    parser.add_argument('--checkpoint-dir',type=Path,help='Run locally with copied runId/checkpoint.npz files and the published result identities.')
    parser.add_argument('--datasets',nargs='+',choices=('mnist','cifar10'),default=['mnist','cifar10'])
    parser.add_argument('--attack',choices=('hidden','gaussian'),default='hidden')
    parser.add_argument('--seed',type=int,choices=(2025,2026,2027),default=2025)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--describe',action='store_true',help='Show matched runs/owners without opening checkpoints or TEST data.')
    args=parser.parse_args()
    if len(set(args.datasets))!=len(args.datasets):parser.error('Choose each dataset once.')
    if not 0<=args.index<10000:parser.error('--index must be between 0 and 9999.')
    selected=(select_local_runs(args.checkpoint_dir,args.datasets,args.attack,args.seed) if args.checkpoint_dir
        else select_runs(args.queue,args.datasets,args.attack,args.seed))
    if args.describe:
        print(json.dumps(dict(testIndex=args.index,trainingPerformed=False,officialTestLoaded=False,
            cases=[dict(method=s['method'],dataset=s['view']['dataset'],runId=s['view']['runId'],owner=s['owner'],
                checkpoint=str(s.get('checkpoint',s['output']/s['view']['runId']/'checkpoint.npz')),
                checkpointPresent=s.get('checkpoint',s['output']/s['view']['runId']/'checkpoint.npz').is_file()
                    if 'checkpoint' in s else None) for s in selected]),indent=2));return
    if args.destination is None:parser.error('--destination is required for evaluation.')
    evaluate(selected,args.index,args.destination)


if __name__=='__main__':main()
