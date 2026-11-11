"""Evaluate the saved CIFAR-10 models on a fixed EuroSAT RGB sample."""
import argparse
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_ENABLE_ONEDNN_OPTS="0",
                  OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
import numpy as np
from PIL import Image
import tensorflow as tf

from basil_core.sharp_protocol import vector_hash
from basil_core.sharp_worker import SharpWorker
from scripts import test_fedsharp_image as frozen

WORK = REPO / ".papermerge"
REFERENCES = REPO / "docs/results/official-test-20261110"
ARCHIVE = WORK / "datasets/eurosat-rgb/EuroSAT_RGB.zip"
DESTINATION = WORK / "evaluations/eurosat-ood-20261110"
SOURCE_URL = "https://zenodo.org/records/7711810"
CLASSES = ("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake")
SAMPLE_SEED = 2025
SAMPLES_PER_CLASS = 200


def sample_images():
    rng = np.random.default_rng(SAMPLE_SEED)
    selected = []
    images = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = {label: [] for label in CLASSES}
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.suffix.lower() == ".jpg" and path.parent.name in members:
                members[path.parent.name].append(name)
        assert sum(map(len, members.values())) == 27000
        for label, files in members.items():
            files.sort()
            for index in sorted(rng.choice(len(files), SAMPLES_PER_CLASS, replace=False)):
                name = files[index]
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    assert image.mode == "RGB" and image.size == (64, 64), name
                    images.append(np.asarray(image.resize((32, 32), Image.Resampling.BOX)))
                selected.append(dict(file=name, landCoverClass=label))
    return np.asarray(images, dtype=np.float32) / np.float32(255), selected


def separation_auc(cifar, eurosat):
    """Rank confidence with CIFAR-10 as the positive class and half credit for ties."""
    scores = np.sort(eurosat)
    left = np.searchsorted(scores, cifar, side="left")
    right = np.searchsorted(scores, cifar, side="right")
    return float(np.mean((left + right) / (2 * len(scores))))


def reference_rows():
    rows = {}
    for folder in ("full-official-test-20261110", "basil-full-test-20261110"):
        root = REFERENCES / folder
        manifest = frozen.read(root / "SHA256.json")
        assert frozen.sha(root / "results.json") == manifest["results.json"]
        result = frozen.read(root / "results.json")
        for case in result["results"]:
            if case["dataset"] == "cifar10":
                relative = case["runId"] + "/predictions.npz"
                assert frozen.sha(root / relative) == manifest[relative]
                with np.load(root / relative, allow_pickle=False) as archive:
                    confidence = archive["predictionConfidence"]
                assert confidence.shape == (100, 10000)
                rows[case["runId"]] = case, confidence
    return rows


def main():
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    selected = frozen.select_local_runs(WORK / "models/final-500r", ["cifar10"],
        "hidden", 2025, ("basil", "basil_acds", "fedsharp", "fedsharp_plus"))
    cases = frozen.preflight(selected)
    references = reference_rows()
    images, sample = sample_images()
    DESTINATION.mkdir()
    (DESTINATION / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    summaries = []
    client_rows = []
    for case in cases:
        run = case["view"]["runId"]
        worker = SharpWorker(case["config"], "CPU")
        params, _ = frozen.load_final_parameters(case, worker.dimension)
        reference, cifar_conf = references[run]
        assert [vector_hash(p) for p in params] == [r["modelSha256"] for r in reference["perClient"]]
        predicted = np.empty((100, len(images)), dtype=np.uint8)
        confidence = np.empty((100, len(images)), dtype=np.float64)
        auc = []
        for client, weights in enumerate(params):
            logits = np.asarray(worker.outputs(weights, images, batch=256), dtype=np.float64)
            assert logits.shape == (len(images), 10) and np.isfinite(logits).all()
            predicted[client] = logits.argmax(axis=1)
            shifted = logits - logits.max(axis=1, keepdims=True)
            confidence[client] = 1 / np.exp(shifted).sum(axis=1)
            auc.append(separation_auc(cifar_conf[client], confidence[client]))
            client_rows.append(dict(method=case["method"], client=client,
                cifar10_mean_confidence=float(cifar_conf[client].mean()),
                eurosat_mean_confidence=float(confidence[client].mean()),
                eurosat_confidence_at_least_90=float((confidence[client] >= .9).mean()),
                confidence_auroc=auc[-1]))
        counts = np.bincount(predicted.ravel(), minlength=10)
        summary = dict(method=case["method"], runId=run,
            cifar10MeanConfidence=float(cifar_conf.mean()),
            eurosatMeanConfidence=float(confidence.mean()),
            eurosatConfidenceAtLeast90=float((confidence >= .9).mean()),
            meanConfidenceAUROC=float(np.mean(auc)),
            cifar10LabelCounts=dict(zip(frozen.CIFAR_CLASSES, counts.tolist())))
        summaries.append(summary)
        np.savez_compressed(DESTINATION / f"{run}.npz",
                            predictedCifar10Class=predicted, predictionConfidence=confidence)
        print(json.dumps(summary), flush=True)
        tf.keras.backend.clear_session()
    for case in cases:
        for name, expected in case["before"].items():
            assert frozen.sha(Path(name)) == expected, name
    with (DESTINATION / "per_client.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(client_rows[0]))
        writer.writeheader()
        writer.writerows(client_rows)
    result = dict(dataset="EuroSAT RGB", source=SOURCE_URL, archiveSha256=frozen.sha(ARCHIVE),
        sampleSeed=SAMPLE_SEED, samplesPerLandCoverClass=SAMPLES_PER_CLASS, images=len(images),
        preprocessing="RGB box resize from 64x64 to 32x32, float32 pixels / 255",
        trainingSeed=2025, trainingRounds=500, trainingPartition="non-IID",
        trainingAttack="Hidden attack + noise", clientModelsPerMethod=100,
        totalEuroSATPredictions=len(images)*len(cases)*100,
        reference="Saved official CIFAR-10 test predictions for the same client parameters",
        tensorflow=tf.__version__, numpy=np.__version__, results=summaries)
    (DESTINATION / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Downloaded original EuroSAT_RGB.zip')
    parser.add_argument('--destination', type=Path, required=True, help='Fresh output directory')
    args = parser.parse_args()
    ARCHIVE = args.archive.resolve()
    DESTINATION = args.destination.resolve()
    if DESTINATION.exists():
        parser.error('Preserve earlier results: destination must not exist.')
    main()
