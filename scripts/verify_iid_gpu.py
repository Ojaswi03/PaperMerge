#!/usr/bin/env python3
"""Synthetic CPU/GPU contracts and timing only: no dataset or research queue."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basil_core.iid_runtime import prepare_environment, configure_tensorflow, POLICY_REVISION, gpu_worker_hash, GPU_MEMORY_LIMIT_MB
prepare_environment('GPU')

import numpy as np
import tensorflow as tf
from basil_core.iid_study import IidWorker
from basil_core.iid_gpu_worker import GpuIidWorker, GpuProtocolAudit
from basil_core.research_protocol import build_model, params_hash, gradient_norm_objective_gradients
from basil_core.research_audit import ProtocolAudit, augment_batch, array_hash
from basil_core.protocol_checkpoint import implementation_fingerprint


def differences(left, right, *, atol, rtol):
    worst=0.
    for a,b in zip(left,right):
        a,b=np.asarray(a),np.asarray(b)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('Non-finite verification tensor')
        np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
        worst=max(worst,float(np.max(np.abs(a-b))))
    return worst


def analytic_ebm(device):
    with tf.device(device):
        model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(10,use_bias=False)])
        theta=np.linspace(-.4,.4,10,dtype=np.float32)
        model.set_weights([theta[None,:]])
        grads,ce,norm_sq,_,_=gradient_norm_objective_gradients(model,tf.ones((1,1)),tf.constant([3]),.0001)
        probability=np.exp(theta.astype(np.float64));probability/=probability.sum()
        base=probability-np.eye(10)[3]
        hessian=np.diag(probability)-np.outer(probability,probability)
        expected=base+.0002*hessian@base
        differences([grads[0].numpy().reshape(-1)],[expected],atol=2e-7,rtol=1e-5)
        np.testing.assert_allclose(float(ce)+.0001*float(norm_sq),
            -np.log(probability[3])+.0001*np.sum(base**2),atol=3e-7)
        return dict(gradientMaxAbsoluteError=float(np.max(np.abs(grads[0].numpy().reshape(-1)-expected))),
            correctionNorm=float(np.linalg.norm(expected-base)),coefficient=.0001)


def main():
    configure_tensorflow(tf,'GPU')
    with tf.device('/CPU:0'):
        canonical=build_model('basil_paper_cnn',2025)
        initial=[v.numpy() for v in canonical.trainable_variables]
    if sum(p.size for p in initial)!=117706:raise AssertionError('Wrong CNN parameter count')
    x_np=np.random.default_rng(17).normal(0,.2,(512,32,32,3)).astype(np.float32)
    y_np=np.arange(512,dtype=np.int32)%10
    measurements={};outputs={};repeat={}
    augmented_hashes={}
    gpu_adapter_equivalence={}
    for device in ('CPU','GPU'):
        measurements[device]={};outputs[device]={}
        with tf.device(f'/{device}:0'):
            worker=(GpuIidWorker if device=='GPU' else IidWorker)(build_model('basil_paper_cnn',2025));worker.load(initial)
            if device=='GPU' and not all('GPU:' in v.handle.device for v in worker.model.trainable_variables):
                raise AssertionError('GPU variables were placed on CPU')
            x=tf.constant(x_np);y=tf.constant(y_np);key=tf.constant([1,2])
            augmented_hashes[device]=array_hash(augment_batch(x,key).numpy())
            for mode in ('none','gradient_norm_objective'):
                worker.load(initial)
                results=[]
                for legacy in (1.,999999.):
                    grads,values=worker._gradients(x,y,key,mode,'paper_absolute_gaussian',tf.constant(.01),tf.constant(legacy))
                    results.append(([g.numpy() for g in grads],np.array([float(v) for v in values])))
                for a,b in zip(results[0][0],results[1][0]):np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(results[0][1],results[1][1])
                repeat[f'{device}/{mode}']='bit_identical; legacy lambda ignored'
                if device=='GPU':
                    reference=IidWorker(build_model('basil_paper_cnn',2025));reference.load(initial)
                    original,original_values=reference._gradients(x,y,key,mode,'paper_absolute_gaussian',tf.constant(.01),tf.constant(1.))
                    for first,second in zip(results[0][0],original):np.testing.assert_array_equal(first,second.numpy())
                    np.testing.assert_array_equal(results[0][1],[float(v) for v in original_values])
                    gpu_adapter_equivalence[mode]='bit_identical to original GPU objective'
                partial,partial_values=worker._gradients(x[:392],y[:392],key,mode,'paper_absolute_gaussian',tf.constant(.01),tf.constant(0.))
                if not all(np.isfinite(g.numpy()).all() for g in partial):raise AssertionError('Non-finite partial-batch gradient')
                optimizer=tf.keras.optimizers.SGD(.05,momentum=0.)
                optimizer.apply_gradients(zip(grads,worker.model.trainable_variables))
                outputs[device][mode]=dict(grads=results[0][0],values=results[0][1],params=worker.export())
                timings=[]
                with tempfile.TemporaryDirectory(prefix='papermerge-gpu-check-') as temporary:
                    observer=(GpuProtocolAudit if device=='GPU' else ProtocolAudit)(Path(temporary)/'audit')
                    for visit in range(7):
                        start=time.perf_counter()
                        observer.begin_batch(worker,x_np,y_np,np.array([1,2],np.int32),np.arange(512),
                            round_id=0,node_id=0,epoch=1,batch=visit,mode=mode,
                            semantics='paper_absolute_gaussian',sigma=.01,legacy_lambda=0.)
                        gradients,values=worker._gradients(x,y,key,mode,'paper_absolute_gaussian',tf.constant(.01),tf.constant(0.))
                        observer.gradients(gradients,values)
                        optimizer.apply_gradients(zip(gradients,worker.model.trainable_variables))
                        observer.end_batch(worker)
                        timings.append(time.perf_counter()-start)
                    observer.close()
                # First two measurements exclude warm-up/autotuning overhead.
                measurements[device][mode]=dict(auditedStepMedianSeconds=float(np.median(timings[2:])),
                    measuredSteps=5,parametersDevice=worker.model.trainable_variables[0].handle.device,
                    gradientDevice=gradients[0].device)
            worker.load(initial)
            layer_values=[];pool_indices=[];value=augment_batch(x,key)
            for layer in worker.model.model.layers:
                if isinstance(layer,tf.keras.layers.MaxPooling2D):
                    _,indices=tf.nn.max_pool_with_argmax(value,ksize=layer.pool_size,
                        strides=layer.strides,padding=layer.padding.upper())
                    pool_indices.append(indices.numpy())
                value=layer(value);layer_values.append(value.numpy())
            outputs[device]['layers']=layer_values;outputs[device]['poolIndices']=pool_indices
            predictions=[];times=[]
            for visit in range(12):
                start=time.perf_counter();logits=worker.predict(x).numpy();times.append(time.perf_counter()-start)
                if visit in (10,11):predictions.append(logits)
            np.testing.assert_array_equal(*predictions)
            if not np.isfinite(worker.predict(x[:272]).numpy()).all():raise AssertionError('Non-finite evaluation partial batch')
            outputs[device]['logits']=logits
            measurements[device]['evaluationBatchMedianSeconds']=float(np.median(times[2:]))
    comparison={}
    if augmented_hashes['CPU']!=augmented_hashes['GPU']:
        raise AssertionError('CPU/GPU augmentation differs for the same keyed batch')
    layer_differences=[float(np.max(np.abs(a-b))) for a,b in zip(outputs['CPU']['layers'],outputs['GPU']['layers'])]
    pool_differences=[int(np.count_nonzero(a!=b)) for a,b in zip(outputs['CPU']['poolIndices'],outputs['GPU']['poolIndices'])]
    print('FORWARD_COMPARISON='+json.dumps(dict(layerMaxAbsoluteDifferences=layer_differences,
        maxPoolArgmaxDifferences=pool_differences)),flush=True)
    for mode in ('none','gradient_norm_objective'):
        a,b=outputs['CPU'][mode],outputs['GPU'][mode]
        difference=np.concatenate([(x-y).reshape(-1) for x,y in zip(a['grads'],b['grads'])])
        base=np.concatenate([x.reshape(-1) for x in a['grads']])
        relative=float(np.linalg.norm(difference)/max(np.linalg.norm(base),1e-12))
        print('GRADIENT_COMPARISON='+json.dumps(dict(mode=mode,maxAbsoluteDifference=float(np.max(np.abs(difference))),
            relativeL2Difference=relative,
            cpuNorm=float(np.linalg.norm(base)),gpuNorm=float(np.linalg.norm(np.concatenate([x.reshape(-1) for x in b['grads']]))),
            augmentationHashes=augmented_hashes)),flush=True)
        # CPU/cuDNN gradient kernels need not be bit-identical. Compare the
        # whole update direction, not relative error in almost-zero coordinates.
        if relative>.01:raise AssertionError('CPU/GPU gradient relative L2 difference exceeds 1%')
        comparison[mode]=dict(gradientRelativeL2Difference=relative,
            gradientMaxAbsoluteDifference=float(np.max(np.abs(difference))),
            lossMaxAbsoluteDifference=differences(a['values'][:4],b['values'][:4],atol=3e-7,rtol=1e-4),
            updateMaxAbsoluteDifference=differences(a['params'],b['params'],atol=2e-6,rtol=1e-5),
            auditedStepSpeedup=measurements['CPU'][mode]['auditedStepMedianSeconds']/measurements['GPU'][mode]['auditedStepMedianSeconds'])
    logit_difference=differences([outputs['CPU']['logits']],[outputs['GPU']['logits']],atol=1e-5,rtol=1e-4)
    np.testing.assert_array_equal(outputs['CPU']['logits'].argmax(1),outputs['GPU']['logits'].argmax(1))
    build=tf.sysconfig.get_build_info();gpu=tf.config.list_physical_devices('GPU')[0]
    numerical=dict(policyRevision=POLICY_REVISION,device='GPU',
        gpuName=tf.config.experimental.get_device_details(gpu)['device_name'],tensorflowVersion=tf.__version__,
        tensorFloat32=False,intraOpThreads=1,interOpThreads=1,
        cudaVersion=build.get('cuda_version'),cudnnVersion=build.get('cudnn_version'),gpuWorkerSha256=gpu_worker_hash(),
        gpuMemoryLimitMb=GPU_MEMORY_LIMIT_MB)
    memory=tf.config.experimental.get_memory_info('GPU:0')
    if memory['peak']>.75*GPU_MEMORY_LIMIT_MB*1024**2:raise AssertionError('Insufficient memory margin for GPU research')
    report=dict(status='passed',purpose='synthetic_hardware_verification',researchValid=False,
        researchRunsStarted=0,numericalExecution=numerical,implementationSha256=implementation_fingerprint(),
        parameterCount=117706,initialModelHash=params_hash(initial),batchSize=512,
        analyticEbm={d:analytic_ebm(f'/{d}:0') for d in ('CPU','GPU')},
        sameDeviceRepeatability=repeat,cpuGpuComparison=comparison,measurements=measurements,
        gpuAdapterEquivalence=gpu_adapter_equivalence,testedTrainingBatchSizes=[512,392],testedEvaluationBatchSizes=[512,272],
        gpuMemory=memory,gpuMemorySafetyMarginBytes=GPU_MEMORY_LIMIT_MB*1024**2-memory['peak'],
        logitMaxAbsoluteDifference=logit_difference,predictionsIdentical=True,
        forwardLayerMaxAbsoluteDifferences=layer_differences,maxPoolArgmaxDifferences=pool_differences,
        evaluationSpeedup=measurements['CPU']['evaluationBatchMedianSeconds']/measurements['GPU']['evaluationBatchMedianSeconds'],
        tolerances=dict(gradientRelativeL2=.01,lossAtol=3e-7,updateAtol=2e-6,analyticGradientAtol=2e-7,
            updateRtol=1e-5,logitAtol=1e-5,logitRtol=1e-4),
        limitations=['Synthetic batch only; not full CIFAR convergence or whole-queue timing',
            'CPU/GPU are tolerance-equivalent, not bit-identical; device transitions must preserve state and record provenance',
            'Multiprocessing remains unverified and is not used'])
    print('GPU_VERIFICATION_JSON='+json.dumps(report),flush=True)


if __name__=='__main__':main()
