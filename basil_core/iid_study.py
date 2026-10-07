"""Paired IID BASIL configurations and instrumentation; no parameter tuning."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from basil_core.protocol_compatibility import IID_PROTOCOL
from basil_core.research_protocol import (
    SequentialWorker, gradient_norm_objective_gradients, iid_partition_indices,
)
from basil_core.research_audit import augment_batch, tensor_stats
from basil_core.trainer import lossFn
from gui.research_protocol import base_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "newResults" / "IID"
PLOT_ROOT = PROJECT_ROOT / "newPlots" / "IID"
TEST_NAMES = ("test_01_basil_iid_no_noise_ce", "test_02_basil_iid_noise_ebm")
CALIBRATED_SIGMAS = (0.005, 0.010, 0.020)
PAIR_DIFFERENCES = frozenset({"experimentName", "conditionId", "runId", "useChannelNoise",
    "channelNoiseSigmaAbsolute", "ebmMode", "localObjective", "sigmaApproval"})


def paired_configs(sigma_absolute=None, *, seed=2025, rounds=100):
    common = base_config(seed=seed)
    common.update(experimentProtocol=IID_PROTOCOL, protocolRevision="iid_basil_source_ebm_v1",
        partitionStrategy="iid", classAssignment="partition_rng", snapshotSelection=True,
        attackHidden=True, channelNoiseSemantics="paper_absolute_gaussian",
        ebmCoefficientSemantics="channel_coordinate_variance", purpose="iid_paired_research",
        snapshotAnchorMu=0., snapshotAnchorMuMax=0., monteCarloNoiseSamples=0,
        executionMode="deterministic_single_process_cpu")
    common["nRounds"]=rounds
    if rounds==50: common["roundEvaluationMetrics"]=True
    configs = []
    for index, name in enumerate(TEST_NAMES):
        cfg = dict(common)
        cfg.update(experimentName=name, conditionId=name, runId=name,
            useChannelNoise=bool(index), channelNoiseSigmaAbsolute=sigma_absolute if index else 0.,
            localObjective="source_ebm" if index else "cross_entropy",
            ebmMode="gradient_norm_objective" if index else "none",
            sigmaApproval="unresolved" if index and sigma_absolute is None else
                "explicit_operator_choice" if index else "channel_disabled")
        configs.append(cfg)
    verify_pair(*configs)
    return configs


def verify_pair(clean, noisy):
    differing = {key for key in clean.keys() | noisy.keys() if clean.get(key) != noisy.get(key)}
    unexpected = differing - PAIR_DIFFERENCES
    if unexpected:
        raise ValueError(f"Unpaired settings: {sorted(unexpected)}")
    return sorted(differing)


def preflight_config(config, *, rounds=3, smoke=False):
    cfg = dict(config, nRounds=rounds, researchValid=False,
               purpose="implementation_smoke_test" if smoke else "full_data_preflight")
    if smoke:
        cfg.update(attackStartRound=1, diagnosticSamplesPerClass=512, diagnosticTestSamplesPerClass=100)
    return cfg


def iid_output(path, *, plots=False):
    root = PLOT_ROOT if plots else RESULT_ROOT
    target = Path(path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"IID output must be within {root}")
    return target


def partition_audit(train_labels, test_labels, *, seed=2025):
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    if len(train_labels) != 50000 or not np.array_equal(np.bincount(train_labels,minlength=10),[5000]*10):
        raise ValueError("Expected the complete CIFAR-10 training set.")
    if len(test_labels) != 10000 or not np.array_equal(np.bincount(test_labels,minlength=10),[1000]*10):
        raise ValueError("Expected the complete evaluation-only CIFAR-10 test set.")
    indices = iid_partition_indices(train_labels, seed)
    repeat = iid_partition_indices(train_labels, seed)
    joined = np.concatenate(indices)
    digest = hashlib.sha256()
    rows = []
    for node, chunk in enumerate(indices):
        counts = np.bincount(train_labels[chunk], minlength=10)
        if len(chunk) != 5000 or np.any(counts == 0) or not np.array_equal(chunk, repeat[node]):
            raise ValueError("IID cardinality, class coverage or determinism failed.")
        digest.update(np.asarray(chunk,dtype="<i8").tobytes())
        rows.append({"nodeId":node,"samples":len(chunk),"classCounts":counts.tolist()})
    if not np.array_equal(np.sort(joined),np.arange(50000)):
        raise ValueError("IID training assignments overlap or omit samples.")
    return {"strategy":"existing_seeded_random_disjoint_equal_split", "partitionHash":digest.hexdigest(),
        "seed":seed,"rngStream":"iid_partition", "assignedSamples":len(joined),
        "uniqueSamples":len(np.unique(joined)),"nodes":rows,"deterministic":True,
        "trainClassCounts":np.bincount(train_labels,minlength=10).tolist(),
        "testClassCounts":np.bincount(test_labels,minlength=10).tolist(),
        "testUsage":"evaluation_only; partition and selection receive training indices only"}, indices


class IidWorker(SequentialWorker):
    extra_metric_names = ("ebmCorrectionNorm", "trainingBatchAccuracy")

    def probe(self, params, x, y):
        try:
            for value in params:
                tf.debugging.assert_all_finite(value,"Non-finite received snapshot")
            self.load(params)
            logits = self.predict(tf.convert_to_tensor(x))
            tf.debugging.assert_all_finite(logits,"Non-finite Snapshot Selection logits")
            ce = lossFn(y,logits)
            tf.debugging.assert_all_finite(ce,"Non-finite receiver-local CE")
            prediction = tf.argmax(logits,axis=1,output_type=tf.int32)
            return float(ce),float(tf.reduce_mean(tf.cast(prediction==tf.cast(y,tf.int32),tf.float32)))
        except Exception as error:
            if self.audit:
                self.audit.write("snapshot_failure.json",{**self.audit.context,"error":str(error),
                    "parameterStats":[tensor_stats(p) for p in params]})
                np.savez_compressed(self.audit.directory/"snapshot_failure_checkpoint.npz",x=x,y=y,
                    **{f"weight_{i}":p for i,p in enumerate(params)})
            raise

    def _batch_gradients(self, x, y, augmentation_key, mode, semantics, sigma, legacy_lambda):
        # The legacy argument is deliberately never read in this worker.
        if semantics != "paper_absolute_gaussian" or mode not in {"none","gradient_norm_objective"}:
            raise ValueError("IID pair supports clean CE and absolute-noise source EBM only.")
        x = augment_batch(x, augmentation_key)
        for variable in self.model.trainable_variables:
            tf.debugging.assert_all_finite(variable,"Non-finite incoming parameter")
        if mode == "gradient_norm_objective":
            grads, values = gradient_norm_objective_gradients(self.model,x,y,sigma**2,details=True)
        else:
            with tf.GradientTape() as tape:
                logits = self.model(x,training=True)
                tf.debugging.assert_all_finite(logits,"Non-finite training logits")
                ce = lossFn(y,logits)
            grads = tape.gradient(ce,self.model.trainable_variables)
            if any(g is None for g in grads):
                raise ValueError("Unexpected disconnected CE gradient.")
            norm = tf.linalg.global_norm(grads)
            accuracy = tf.reduce_mean(tf.cast(tf.argmax(logits,axis=1,output_type=tf.int32)==y,tf.float32))
            values = (ce,norm**2,ce,tf.constant(0.),norm,norm,tf.constant(0.,tf.float64),accuracy)
        for name,value in zip(("CE","gradient norm squared","objective","coefficient",
                               "base norm","applied norm","EBM correction norm","accuracy"),values):
            tf.debugging.assert_all_finite(value,f"Non-finite {name}")
        for gradient in grads:
            tf.debugging.assert_all_finite(gradient,"Non-finite applied gradient")
        return grads,values
