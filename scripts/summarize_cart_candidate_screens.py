#!/usr/bin/env python3
"""Make a compact, auditable receipt from completed local candidate screens."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(roots,destination):
    studies=[]
    for root in roots:
        plan=json.loads((root/'plan.json').read_text())
        validation=root/'independent_validation.json'
        if not validation.is_file():raise ValueError('Independent validation is missing: '+str(root))
        cases=[]
        for case in plan['cases']:
            path=root/case['name']/'result.json';row=json.loads(path.read_text())
            if row['status']!='COMPLETED_DIAGNOSTIC':raise ValueError('Incomplete candidate screen.')
            fields=('meanAccuracy','worstNodeAccuracy','elapsedSeconds','actualRouteCounts','meanLastRoundTrainCE',
                'qualityCommitChecks','qualityScoreChecks','scaledCommits','candidateScoring','calibrationDraws')
            cases.append(dict(name=case['name'],config=case['config'],
                results={k:row[k] for k in fields if k in row},
                roundMetrics=[{k:m[k] for k in ('round','meanAccuracy','worstNodeAccuracy','worstClassMean') if k in m} for m in row['roundMetrics']],
                contract=row['contract'],rawResultSha256=digest(path)))
        studies.append(dict(name=root.name,planSha256=digest(root/'plan.json'),cases=cases,
            independentValidation=json.loads(validation.read_text()),independentValidationSha256=digest(validation)))
    result=dict(status='COMPLETED_LOCAL_DEVELOPMENT_SCREENS',studies=studies,officialTestUsed=False,
        interpretation='Short development screens; horizons/attack onsets differ from the HPC study. No 100-round, generalization, novelty or universal robustness claim.',
        programSha256=digest(Path(__file__)))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Saved {len(studies)} screen receipts to {destination}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args();summarize(a.inputs,a.destination)
