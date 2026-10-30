#!/usr/bin/env python3
"""Remove duplicate plot PDFs from a verified export; retain papers and data."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root):
    root=Path(root).resolve();index=root/'SHA256.json'
    manifest=json.loads(index.read_text());original_sha=sha(index)
    for name,value in manifest.items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path)!=value:
            raise ValueError('Export checksum/path mismatch: '+name)
    def plot(path):
        parts=path.relative_to(root).parts
        return 'source_snapshots' not in parts and any(p in ('thesis_figures','publication_curves','figures') for p in parts)
    removed={str(p.relative_to(root)):sha(p) for p in root.rglob('*.pdf') if plot(p)}
    if not removed:return 0
    if (root/'publication_cleanup.json').exists():
        raise FileExistsError('This export already records a different packaging change.')
    changed={}
    for path in root.rglob('*.json'):
        if not plot(path):continue
        value=json.loads(path.read_text());modified=False
        if not isinstance(value,dict):continue
        for key in ('outputSha256','files','plotFiles'):
            old=value.get(key)
            if isinstance(old,dict):new={k:v for k,v in old.items() if not k.lower().endswith('.pdf')}
            elif isinstance(old,list):new=[v for v in old if not isinstance(v,str) or not v.lower().endswith('.pdf')]
            else:continue
            if new!=old:value[key]=new;modified=True
        if modified:
            name=str(path.relative_to(root));changed[name]=sha(path)
            path.write_text(json.dumps(value,indent=2)+'\n');manifest[name]=sha(path)
    for name in removed:(root/name).unlink();manifest.pop(name,None)
    note=dict(reason='Author requests plot PNG and SVG only; duplicate plot PDFs removed.',
        originalExportManifestSha256=original_sha,originalVerifiedFiles=len(manifest)+len(removed),
        removedPlotPdfs=removed,modifiedPlotManifestOriginalSha256=changed,
        scientificRecordsAndSourceSnapshotsUnchanged=True)
    path=root/'publication_cleanup.json';path.write_text(json.dumps(note,indent=2)+'\n')
    manifest[path.name]=sha(path);index.write_text(json.dumps(manifest,indent=2)+'\n')
    return len(removed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(f'Removed {package(args.output)} duplicate plot PDFs; scientific evidence retained.')
