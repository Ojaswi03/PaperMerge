#!/usr/bin/env python3
"""Full official TEST evaluation of explicitly selected, frozen final models.

No fitting, admission, repair, threshold tuning, parameter averaging or model
selection is performed. Client dispersion is not variation across random seeds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import test_fedsharp_image as frozen


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def load_test(dataset):
    import numpy as np
    from scripts.verify_cart_comparison_results import test_data
    x,y,identity=test_data(dataset)
    expected=(10000,28,28,1) if dataset=='mnist' else (10000,32,32,3)
    if (x.shape!=expected or y.shape!=(10000,) or x.dtype!=np.float32 or y.dtype!=np.int32
            or not np.isfinite(x).all() or x.min()<0 or x.max()>1
            or not np.array_equal(np.unique(y),np.arange(10))):raise ValueError('Full official TEST split required.')
    x.flags.writeable=False;y.flags.writeable=False
    return x,y,dict(identity,split='official_test',samples=len(y),
        imagesSha256=hashlib.sha256(x.tobytes()).hexdigest(),labelsSha256=hashlib.sha256(y.tobytes()).hexdigest(),
        previouslyDemonstratedIndices=[0],normalization='float32 raw pixels / 255; no augmentation')


def classify(logits,labels):
    import numpy as np
    z=np.asarray(logits,np.float64);y=np.asarray(labels)
    if (z.shape!=(len(y),10) or not len(y) or not np.isfinite(z).all()
            or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=10)):
        raise ValueError('Invalid test logits or class labels.')
    s=z-z.max(axis=1,keepdims=True);logsum=np.log(np.exp(s).sum(axis=1))
    loss=logsum-s[np.arange(len(y)),y];pred=z.argmax(axis=1)
    confidence=np.exp(-logsum)  # Probability of the largest logit.
    confusion=np.bincount(10*y+pred,minlength=100).reshape(10,10)
    counts=confusion.sum(axis=1)
    if np.any(counts==0):raise ValueError('Every class must occur in the full test split.')
    row=dict(testSamples=len(y),testCorrect=int(np.count_nonzero(pred==y)),
        testAccuracy=float(np.mean(pred==y)),meanCrossEntropyLoss=float(loss.mean()),
        meanPredictionConfidence=float(confidence.mean()),classRecall=(np.diag(confusion)/counts).tolist(),
        confusionMatrix=confusion.tolist(),
        excludingPreviouslyDemonstratedImage0=dict(samples=len(y)-1,
            accuracy=float(np.mean(pred[1:]==y[1:])),meanCrossEntropyLoss=float(loss[1:].mean())))
    return row,pred.astype(np.uint8),loss,confidence


def summarize(rows,validation):
    import numpy as np
    accuracy=np.asarray([r['testAccuracy'] for r in rows]);loss=np.asarray([r['meanCrossEntropyLoss'] for r in rows])
    recalls=np.asarray([r['classRecall'] for r in rows]);confusion=np.asarray([r['confusionMatrix'] for r in rows]).sum(axis=0)
    return dict(clients=len(rows),testSamplesPerClient=10000,meanAccuracy=float(accuracy.mean()),
        meanCrossEntropyLoss=float(loss.mean()),medianClientAccuracy=float(np.median(accuracy)),
        worstClientAccuracy=float(accuracy.min()),bestClientAccuracy=float(accuracy.max()),
        clientAccuracyP10=float(np.quantile(accuracy,.10)),
        clientAccuracySD=float(accuracy.std(ddof=1)) if len(rows)>1 else None,
        clientSDInterpretation='Dispersion across client models of one seed; not seed SD or a confidence interval.',
        meanClassRecall=recalls.mean(axis=0).tolist(),worstMeanClassRecall=float(recalls.mean(axis=0).min()),
        summedClientConfusionMatrix=confusion.tolist(),finalRoundValidationMeanAccuracy=float(np.mean(validation)),
        testMinusFinalValidationPercentagePoints=float(100*(accuracy.mean()-np.mean(validation))),
        excludingPreviouslyDemonstratedImage0=dict(testSamplesPerClient=9999,
            meanAccuracy=float(np.mean([r['excludingPreviouslyDemonstratedImage0']['accuracy'] for r in rows])),
            meanCrossEntropyLoss=float(np.mean([r['excludingPreviouslyDemonstratedImage0']['meanCrossEntropyLoss'] for r in rows]))))


def evaluate_case(item,x,y,destination,batch,done_before,total,started):
    import numpy as np
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import load_state,vector_hash
    from basil_core.research_protocol import resolve_attackers
    cfg=item['config'];row=item['row'];run=item['view']['runId'];case_dir=destination/run
    case_dir.mkdir()
    worker=SharpWorker(cfg,'CPU');initial_matches=vector_hash(worker.export())==row['initialHash']
    params_all,validation=frozen.load_final_parameters(item,worker.dimension)
    n=cfg['nodes']
    attackers=set(resolve_attackers(cfg['seed'],cfg['actualAttackers'],n))
    predictions=np.empty((n,len(y)),np.uint8);losses=np.empty((n,len(y)),np.float64)
    confidences=np.empty((n,len(y)),np.float64);rows=[];case_start=time.monotonic()
    for client,params in enumerate(params_all):
        logits=worker.outputs(params,x,batch=batch)
        metrics,pred,loss,confidence=classify(logits,y)
        metrics.update(clientId=client,attackerIdentityDuringTraining=client in attackers,modelSha256=vector_hash(params),
            finalValidationAccuracy=float(validation[-1,client]))
        predictions[client]=pred;losses[client]=loss;confidences[client]=confidence;rows.append(metrics)
        elapsed=time.monotonic()-case_start;eta=elapsed/(client+1)*(n-client-1)
        progress=dict(status='RUNNING',dataset=cfg['dataset'],method=item['method'],runId=run,
            completedClients=done_before+client+1,totalClients=total,caseCompletedClients=client+1,
            elapsedSeconds=time.monotonic()-started,currentCaseEtaSeconds=eta,
            clientImagePredictionsCompleted=(done_before+client+1)*len(y),trainingPerformed=False)
        atomic(destination/'progress.json',progress)
        if client==0 or (client+1)%10==0:
            atomic(case_dir/'client_metrics.json',rows)
            print(f'{cfg["dataset"]:7} {item["method"]:9} clients {client+1:3}/{n} | '
                f'overall {done_before+client+1}/{total} | case ETA {eta/60:.1f} min | '
                f'client accuracy {metrics["testAccuracy"]:.2%}',flush=True)
    with (case_dir/'predictions.npz').open('wb') as stream:
        np.savez_compressed(stream,labels=y,predictedClass=predictions,crossEntropyLoss=losses,predictionConfidence=confidences)
    honest=[r for r in rows if not r['attackerIdentityDuringTraining']]
    honest_ids=[r['clientId'] for r in honest]
    result=dict(method=item['method'],dataset=cfg['dataset'],runId=run,seed=cfg['seed'],partition=cfg['partition'],
        environment='joint',attackType=cfg['attackType'],sigma=cfg['sigma'],rounds=500,
        checkpointSha256=item['before'][str(item['checkpoint'])],
        allClients=summarize(rows,validation[-1]),honestClients=summarize(honest,validation[-1,honest_ids]),
        perClient=rows,elapsedSeconds=time.monotonic()-case_start,
        evaluationRuntime=dict(tensorflow=tf.__version__,numpy=np.__version__,device='CPU',
            initializationMatchesTraining=initial_matches),
        trainingRuntime={k:row['runtime'][k] for k in ('tensorflow','numpy')})
    atomic(case_dir/'result.json',result)
    del params_all,worker,logits
    tf.keras.backend.clear_session()
    return result


def write_report(destination,receipt):
    results=receipt['results']
    lines=['# Official test-set evaluation','',
        'Frozen final round-500 models; non-IID, Hidden attack + noise, seed 2025. No retraining or test-based tuning.',
        'All 10,000 official test images per dataset, evaluated separately by each of 100 retained client models.',
        '', '| Dataset | Method | Test accuracy, all clients | Mean test CE | Honest-client accuracy | Round-500 validation |',
        '|---|---|---:|---:|---:|---:|']
    for r in results:
        a=r['allClients'];h=r['honestClients']
        lines.append(f'| {r["dataset"]} | {r["method"]} | {100*a["meanAccuracy"]:.3f}% | '
            f'{a["meanCrossEntropyLoss"]:.6f} | {100*h["meanAccuracy"]:.3f}% | {100*a["finalRoundValidationMeanAccuracy"]:.3f}% |')
    lines.extend(['','Accuracy is first computed over all test images for each client model, then averaged equally across clients.',
        'Honest-client results use the 67 identities not configured to corrupt outgoing copies during training.',
        'All-client inference uses retained trained parameters, including the trained models of attack-configured identities, not their corrupted outgoing copies.',
        'Loss is ordinary raw-logit cross-entropy in natural-log units, without proximal/gradient penalties or test-time prior adjustment.',
        '', '## Scope and interpretation','',
        f'This evaluation covers {len(results)} runs of one seed and one non-IID attack/noise setting, not the full 150-run study or a three-seed result.',
        'The validation comparison is the final round-500 checkpoint, not the mean of rounds 491 to 500 used for the main validation endpoints.',
        'Client variation is not seed variation, a confidence interval, or evidence of statistical significance.',
        'Test image 0 was demonstrated earlier. No model changes were made in response; the JSON also reports the other 9,999 images separately.',
        'The official test data were not used for fitting, admission, repair, checkpoint selection, or threshold tuning in this evaluation.',
        'Local CPU inference uses TensorFlow 2.18.0; training used 2.21.0. All trained parameters and model-source identities were checked; runtimes are recorded.',
        'Original checkpoints and experiment artifacts were hashed before and after evaluation and remained unchanged.'])
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    fields=('dataset','method','runId','clientId','attackerIdentityDuringTraining','testSamples','testCorrect','testAccuracy',
        'meanCrossEntropyLoss','meanPredictionConfidence','finalValidationAccuracy')
    with (destination/'per_client.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in results:
            for row in r['perClient']:
                writer.writerow({key:(r[key] if key in ('dataset','method','runId') else row[key]) for key in fields})


def evaluate(checkpoint_dir,destination,batch=256,threads=4,methods=('fedsharp','fedsharp_plus')):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a fresh destination; previous TEST results are never overwritten.')
    selected=frozen.select_local_runs(checkpoint_dir,['mnist','cifar10'],'hidden',2025,methods)
    cases=frozen.preflight(selected)
    total=len(cases)*100
    if any(destination.is_relative_to(c['output'].resolve()) or destination.is_relative_to(c['checkpoint'].parent) for c in cases):
        raise ValueError('Report destination must be outside original model/experiment directories.')
    plan=dict(purpose='full official test evaluation, frozen models, no tuning',
        createdAt=datetime.now(timezone.utc).isoformat(),datasetOrder=['mnist','cifar10'],
        datasetSamples=10000,classes=10,seed=2025,partition='dirichlet',attackType='hidden',sigma=.01,
        rounds=500,clientModelsPerCase=100,totalCases=len(cases),totalClientImagePredictions=total*10000,
        batchSize=batch,threads=threads,previouslyDemonstratedTestIndices=[0],
        cases=[dict(method=c['method'],runId=c['view']['runId'],checkpoint=str(c['checkpoint']),
            checkpointSha256=c['before'][str(c['checkpoint'])]) for c in cases],
        evaluationProgramSha256=frozen.sha(Path(__file__)),helperSha256=frozen.sha(Path(frozen.__file__)))
    destination.mkdir(parents=True);atomic(destination/'evaluation_plan.json',plan)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/papermerge-mpl')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    results=[];cache={};started=time.monotonic()
    for c in cases:
        ds=c['config']['dataset']
        if ds not in cache:cache[ds]=load_test(ds)
        x,y,_=cache[ds]
        results.append(evaluate_case(c,x,y,destination,batch,len(results)*100,total,started))
    for c in cases:
        for name,digest in c['before'].items():
            if frozen.sha(Path(name))!=digest:raise ValueError('Frozen input changed during evaluation: '+name)
    receipt=dict(status='COMPLETED_OFFICIAL_TEST_EVALUATION',plan=plan,
        testData={ds:v[2] for ds,v in cache.items()},results=results,trainingPerformed=False,
        optimizerUpdatesAdded=0,checkpointsModified=False,testUsedForTrainingOrSelection=False,
        distinctTestImagesPerDataset=10000,clientImagePredictions=total*10000,elapsedSeconds=time.monotonic()-started)
    atomic(destination/'results.json',receipt);write_report(destination,receipt)
    atomic(destination/'progress.json',dict(status=receipt['status'],completedClients=total,totalClients=total,
        elapsedSeconds=receipt['elapsedSeconds'],trainingPerformed=False))
    manifest={str(p.relative_to(destination)):frozen.sha(p) for p in destination.rglob('*') if p.is_file()}
    atomic(destination/'SHA256.json',manifest)
    for r in results:
        a=r['allClients'];print(f'{r["dataset"]} {r["method"]}: TEST {a["meanAccuracy"]:.3%}, CE {a["meanCrossEntropyLoss"]:.6f}',flush=True)
    print('Saved full test results: '+str(destination),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--methods',nargs='+',choices=tuple(frozen.METHODS),default=['fedsharp','fedsharp_plus'])
    args=parser.parse_args()
    if not 1<=args.batch_size<=1024:parser.error('Batch size must be between 1 and 1024.')
    if not 1<=args.threads<=8:parser.error('Use between 1 and 8 CPU threads.')
    if len(args.methods)!=len(set(args.methods)):parser.error('Choose each method once.')
    evaluate(args.checkpoint_dir,args.destination,args.batch_size,args.threads,args.methods)
