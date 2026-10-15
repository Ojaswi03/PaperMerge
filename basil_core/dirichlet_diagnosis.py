"""Read-only analysis of completed local-only Dirichlet runs.

Nothing here trains, selects production hyperparameters or changes BASIL. Heavy
libraries are imported only when a numerical operation is explicitly requested.
"""
from __future__ import annotations

import math


def array_identity_hash(array, *, chunk_rows=1024):
    """Match research_audit.array_hash without copying the full image cache."""
    import hashlib
    import numpy as np
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError('Positive integer chunk_rows required.')
    value = np.asarray(array)
    digest = hashlib.sha256(str(value.shape).encode())
    if value.ndim == 0:
        digest.update(value.tobytes())
    else:
        for start in range(0, len(value), chunk_rows):
            digest.update(np.ascontiguousarray(value[start:start + chunk_rows]).tobytes())
    return digest.hexdigest()


def actually_corrupted(sender, snapshot_round, config):
    """Designated attackers' warm-up snapshots are honest in this study."""
    return bool(config['attackHidden'] and sender in config['resolvedAttackerIds']
                and snapshot_round >= config['attackStartRound'])


def summarize_activations(records, partition, config):
    """Check saved contracts and measure changes, never reinterpret missing data."""
    import numpy as np
    rounds = config['nRounds']
    if len(records) != rounds * 10:
        raise ValueError('Incomplete activation history.')
    nodes = partition['nodes']
    first_corrupted = None
    corrupt_selections = 0
    node_changes = [[] for _ in nodes]
    class_changes = [[] for _ in nodes]
    node_missing_forgetting = [[] for _ in nodes]
    ratios = []
    ebm_ratios = []
    total_steps = 0
    for index, record in enumerate(records):
        round_id, node = divmod(index, 10)
        if (record['round'], record['nodeId']) != (round_id, node):
            raise ValueError('Duplicated, missing or unordered activation.')
        candidates = record['candidates']
        expected_senders = [(node - distance) % 10 for distance in range(5, 0, -1)]
        expected_rounds = [-1 if round_id == 0 and sender >= node else
                           round_id if sender < node else round_id - 1
                           for sender in expected_senders]
        if ([c['senderId'] for c in candidates] != expected_senders or
                [c['snapshotRound'] for c in candidates] != expected_rounds):
            raise ValueError(f'Rolling memory mismatch at R{round_id} N{node}.')
        losses = [c['receiverLocalLoss'] for c in candidates]
        if not all(math.isfinite(loss) for loss in losses):
            raise ValueError('Non-finite candidate CE.')
        minimum = min(losses)
        tied = [c for c in candidates if c['receiverLocalLoss'] <= minimum + 1e-8]
        chosen = min(tied, key=lambda c: (node - c['senderId']) % 10)
        if (record['inputSourceNode'], record['inputSnapshotRound']) != (
                chosen['senderId'], chosen['snapshotRound']):
            raise ValueError('Selection does not match local-CE / nearest tie rule.')
        if record['snapshotAnchorReferenceHash'] != record['inputHash']:
            raise ValueError('Anchor is not the fully selected incoming snapshot.')
        corrupted = actually_corrupted(chosen['senderId'], chosen['snapshotRound'], config)
        corrupt_selections += int(corrupted)
        if corrupted and first_corrupted is None:
            first_corrupted = dict(round=round_id, node=node, candidates=candidates,
                selectedSender=chosen['senderId'], selectedLoss=chosen['receiverLocalLoss'],
                beforeAccuracy=record['beforeTraining']['globalTestAccuracy'],
                afterAccuracy=record['globalTestAccuracy'])
        epochs = record['epochs']
        samples = nodes[node]['samples']
        batch_sizes = [512] * (samples // 512) + ([samples % 512] if samples % 512 else [])
        if len(epochs) != 5 or [e['epoch'] for e in epochs] != list(range(1, 6)):
            raise ValueError('Five complete epochs required.')
        if any(e['samplesSeen'] != samples or e['batchSizes'] != batch_sizes for e in epochs):
            raise ValueError('Local coverage / partial-batch mismatch.')
        if record['optimizerSteps'] != 5 * len(batch_sizes):
            raise ValueError('Optimizer-step mismatch.')
        steps = record['optimizerTelemetry']
        if len(steps) != record['optimizerSteps']:
            raise ValueError('Missing optimizer telemetry.')
        if any(not math.isfinite(v) for step in steps for v in step.values()
               if isinstance(v, (int, float))):
            raise ValueError('Non-finite step telemetry.')
        ratios.extend(s['anchorCorrectionRatio'] for s in steps)
        ebm_ratios.extend(s['ebmCorrectionRatio'] for s in steps)
        total_steps += record['optimizerSteps']
        before = record['beforeTraining']
        node_changes[node].append(record['globalTestAccuracy'] - before['globalTestAccuracy'])
        class_changes[node].append(np.asarray(record['perClassGlobalTestAccuracy']) -
                                   np.asarray(before['perClassGlobalTestAccuracy']))
        absent = np.asarray(nodes[node]['classCounts']) == 0
        if absent.any():
            forgetting = (np.asarray(before['perClassGlobalTestAccuracy']) -
                          np.asarray(record['perClassGlobalTestAccuracy']))
            node_missing_forgetting[node].append(float(forgetting[absent].mean()))
    return dict(checkedActivations=len(records), optimizerSteps=total_steps,
        stepsPerRing=total_steps // rounds, rollingMemoryCorrect=True,
        lowestLocalLossSelectionCorrect=True, frozenSelectedAnchorCorrect=True,
        fiveCompleteEpochsCorrect=True, actuallyCorruptedSelections=corrupt_selections,
        firstCorruptedSelection=first_corrupted,
        meanAnchorCorrectionRatio=float(np.mean(ratios)),
        meanEbmCorrectionRatio=float(np.mean(ebm_ratios)),
        nodes=[dict(nodeId=i, meanActivationAccuracyChangePp=100 * float(np.mean(changes)),
                    meanPerClassChangePp=(100 * np.mean(class_changes[i], axis=0)).tolist(),
                    meanAbsentClassForgettingPp=(100 * float(np.mean(node_missing_forgetting[i]))
                        if node_missing_forgetting[i] else None))
               for i, changes in enumerate(node_changes)])


def confusion_summary(confusion):
    import numpy as np
    matrix = np.asarray(confusion)
    if (matrix.shape != (10, 10) or not np.issubdtype(matrix.dtype, np.integer)
            or np.any(matrix < 0) or np.any(matrix.sum(axis=1) == 0)):
        raise ValueError('Nonempty ten-class integer confusion matrix required.')
    counts = matrix.sum(axis=0)
    return dict(predictedCounts=counts.tolist(),
        constantPredictionClass=int(counts.argmax()) if counts.max() == matrix.sum() else None,
        accuracy=float(np.trace(matrix) / matrix.sum()),
        perClassAccuracy=(np.diag(matrix) / matrix.sum(axis=1)).tolist())


def probe_saved_parameters(model, params, images, labels):
    """Forward + ordinary CE differentiation on TRAIN only; zero SGD updates.

    This differentiates CE at an immutable saved state to inspect gradient paths,
    not to optimize or select a loss. Caller owns explicit CPU placement.
    """
    import numpy as np
    import tensorflow as tf
    from basil_core.forgetting_audit import classification_metrics
    from basil_core.research_protocol import params_hash
    from basil_core.trainer import lossFn
    original = [v.numpy().copy() for v in model.trainable_variables]
    source_hash = params_hash(params)
    try:
        model.set_weights(params)
        network = model.model
        hidden_layers = [layer for layer in network.layers
                         if getattr(getattr(layer, 'activation', None), '__name__', None) == 'relu']
        features = tf.keras.Model(network.inputs, [layer.output for layer in hidden_layers])
        values = features(tf.convert_to_tensor(images), training=False)
        with tf.GradientTape() as tape:
            logits = model(tf.convert_to_tensor(images), training=False)
            loss = lossFn(labels, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(g is None for g in gradients):
            raise ValueError('Unexpected disconnected probe gradient.')
        if not all(np.isfinite(g.numpy()).all() for g in gradients):
            raise ValueError('Non-finite probe gradient.')
        result = dict(parameterHash=source_hash, optimizerSteps=0,
            probeUsage='balanced_TRAIN_evaluation_only',
            meanLogitVarianceAcrossImages=float(np.var(logits.numpy().astype(np.float64), axis=0).mean()),
            classification=classification_metrics(logits.numpy(), labels),
            gradients=[dict(tensorIndex=i, shape=list(g.shape),
                l2=float(np.linalg.norm(g.numpy().astype(np.float64))),
                zeroFraction=float(np.mean(g.numpy() == 0))) for i, g in enumerate(gradients)],
            hiddenActivations=[dict(layerIndex=i, layerType=type(layer).__name__,
                shape=list(value.shape), zeroFraction=float(np.mean(value.numpy() == 0)),
                meanFeatureVariance=float(np.var(value.numpy().astype(np.float64), axis=0).mean()))
                for i, (layer, value) in enumerate(zip(hidden_layers, values))])
        if params_hash(model.get_weights()) != source_hash or params_hash(params) != source_hash:
            raise ValueError('Read-only probe changed saved parameters.')
        return result
    finally:
        model.set_weights(original)


def full_local_training_metrics(model, params, images, labels, indices, *, batch_size=512):
    """Correctly named whole-partition TRAIN metrics, streamed and zero-update.

    Historical localTrainingAccuracy uses only class-grouped x[:512]; do not
    rewrite that field or confuse this training fit with test generalization.
    """
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    ids = np.asarray(indices)
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
            or ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(labels)) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Unique valid local TRAIN indices and positive batch size required.')
    original = [v.numpy().copy() for v in model.trainable_variables]
    source_hash = params_hash(params)
    confusion = np.zeros((10, 10), np.int64)
    weighted_ce = 0.
    try:
        model.set_weights(params)
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            targets = np.asarray(labels[chunk], np.int32)
            logits = model(tf.convert_to_tensor(np.asarray(images[chunk])), training=False)
            tf.debugging.assert_all_finite(logits, 'Non-finite full-local TRAIN logits')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            tf.debugging.assert_all_finite(loss, 'Non-finite full-local TRAIN CE')
            weighted_ce += float(tf.reduce_sum(tf.cast(loss, tf.float64)))
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
            np.add.at(confusion, (targets, prediction), 1)
        if params_hash(model.get_weights()) != source_hash or params_hash(params) != source_hash:
            raise ValueError('Full-local measurement changed saved parameters.')
        return dict(samples=len(ids), accuracy=float(np.trace(confusion) / len(ids)),
            crossEntropy=weighted_ce / len(ids), confusion=confusion.tolist(),
            predictedCounts=confusion.sum(axis=0).tolist(), usage='full_local_TRAIN_evaluation_only',
            augmentation=False, optimizerSteps=0)
    finally:
        model.set_weights(original)
