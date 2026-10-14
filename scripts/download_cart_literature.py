#!/usr/bin/env python3
"""Download explicitly listed open primary PDFs without replacing old papers.

No model import, training, authentication, W&B or experiment queue. Standard
library only. Defaults to a plan; --download explicitly performs network I/O.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'papers/prototype_retention'
MANIFEST = DIRECTORY/'download_manifest.json'


def download(entry):
    name = entry['id']
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError('Unsafe paper ID.')
    destination = DIRECTORY/f'{name}.pdf'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}.')
    errors = []
    for url in entry['pdfUrls']:
        if not url.startswith('https://'):
            raise ValueError('HTTPS primary sources required.')
        temporary = None
        try:
            request = Request(url, headers={'User-Agent': 'PaperMerge-Literature-Review/1.0 (academic PDF retrieval)'})
            with urlopen(request, timeout=45) as response, tempfile.NamedTemporaryFile(dir=DIRECTORY, suffix='.part', delete=False) as handle:
                temporary = Path(handle.name)
                count = 0
                digest = hashlib.sha256()
                while block := response.read(1024*1024):
                    count += len(block)
                    if count > 50*1024*1024:
                        raise ValueError('PDF exceeds 50-MiB safety bound.')
                    handle.write(block)
                    digest.update(block)
                resolved = response.url
            with temporary.open('rb') as handle:
                if handle.read(5) != b'%PDF-':
                    raise ValueError('Response is not a PDF (HTML/captcha/error page).')
                handle.seek(max(0, count-4096))
                if b'%%EOF' not in handle.read():
                    raise ValueError('Incomplete PDF trailer.')
            # Hard-link is atomic and cannot overwrite a concurrently created file.
            destination.hardlink_to(temporary)
            temporary.unlink()
            return dict(id=name, status='downloaded', file=str(destination.relative_to(ROOT)),
                requestedURL=url, resolvedURL=resolved, bytes=count, sha256=digest.hexdigest(),
                previousSourceAttempts=errors)
        except Exception as error:
            if temporary is not None and temporary.exists():
                temporary.unlink()
            errors.append(dict(url=url, error=str(error)))
    return dict(id=name, status='failed', errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text())['papers']
    if not args.download:
        print(json.dumps(dict(papers=len(entries), networkRequests=0, trainingRuns=0), indent=2))
        return
    receipt = DIRECTORY/'download_receipt.json'
    if receipt.exists():
        raise FileExistsError('Refusing to replace an existing literature receipt.')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, entries))
    receipt.write_text(json.dumps(dict(results=results, trainingRunsStarted=0), indent=2)+'\n')
    print(json.dumps(dict(downloaded=sum(r['status'] == 'downloaded' for r in results),
        failed=[r for r in results if r['status'] == 'failed'], receipt=str(receipt))))
    if any(r['status'] != 'downloaded' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
