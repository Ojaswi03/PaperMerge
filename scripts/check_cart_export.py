#!/usr/bin/env python3
"""Verify compact export hashes and recompute its paired metric summary."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def check(root):
    from scripts.report_cart_v2_completed import summarize
    from scripts.run_cart_v2_transfer import program_hashes
    root=Path(root).resolve();hashes=json.loads((root/'SHA256.json').read_text())
    for name,expected in hashes.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path.')
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:raise ValueError(f'Export hash mismatch: {name}')
    summary=summarize(root)
    if summary!=json.loads((root/'summary.json').read_text()):raise ValueError('Saved summary differs from metrics.')
    plan=json.loads((root/'plan.json').read_text());execution=json.loads((root/'compiled_transfer_execution.json').read_text())
    if execution['programSha256']!=program_hashes():raise ValueError('Frozen runtime implementation changed.')
    if execution['planSha256']!=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest():raise ValueError('Runtime plan identity mismatch.')
    figures=root/'figures'/'artifacts_manifest.json'
    figure_count=0
    if figures.exists():
        manifest=json.loads(figures.read_text())
        for name,digest in manifest['outputSha256'].items():
            if hashlib.sha256((root/'figures'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Figure output hash mismatch.')
        for run,digest in manifest['inputMetricSha256'].items():
            if hashlib.sha256((root/run/'metrics.npz').read_bytes()).hexdigest()!=digest:raise ValueError('Figure metric input hash mismatch.')
        figure_count=len(manifest['files'])
    return dict(status='EXPORTED_METRICS_AND_FIGURES_VERIFIED',files=len(hashes),configs=len(plan['configs']),
        figureArtifacts=figure_count,sourceSha256=plan['source'],runtimeProgramSha256=execution['programSha256'],
        pairedCartEffects=summary['pairedCartEffects'],independentInferencePerformed=False,
        interpretation='Checks exported metric arrays, source contracts and figure provenance. Model checkpoints were excluded; no fresh model inference was performed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path);args=p.parse_args();row=check(args.output)
    if args.receipt:
        if args.receipt.exists():raise FileExistsError('Preserve earlier receipts; use a fresh path.')
        args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(row,indent=2)+'\n')
    print(f'Verified {row["configs"]} configurations, {row["files"]} files and {row["figureArtifacts"]} figure artifacts.')
    print('Metric verification only; checkpoints remain on HPC for independent inference.')
