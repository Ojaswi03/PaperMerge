"""Publication figures retain every condition, true seed uncertainty and names."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from reporting import fedsharp_publication as publication


def test_professor_names_are_display_aliases_not_scientific_redefinitions():
    assert publication.label('sharp',None)=='FedSharp'
    assert publication.label('sharp','cart')=='FedSharp+'
    assert publication.label('basil','acds')=='BASIL + ACDS'


def test_tampered_export_is_rejected_before_figures_are_created(tmp_path):
    (tmp_path/'metrics.npz').write_bytes(b'changed')
    (tmp_path/'SHA256.json').write_text(json.dumps({'metrics.npz':hashlib.sha256(b'original').hexdigest()}))
    with pytest.raises(ValueError,match='checksum'):
        publication.audit(tmp_path)


def test_export_manifest_cannot_escape_archive(tmp_path):
    (tmp_path/'SHA256.json').write_text(json.dumps({'../outside.json':'fake'}))
    with pytest.raises(ValueError,match='path'):
        publication.audit(tmp_path)


def test_four_vector_figures_include_all_seeds_and_unmodified_curve_values(tmp_path):
    rows=[]
    for dataset in ('mnist','cifar10'):
        for partition in ('iid','dirichlet'):
            for family in ('basil','sharp'):
                specs=list(publication.ENVIRONMENTS)+([('joint','acds' if family=='basil' else 'cart')] if partition=='dirichlet' else [])
                for env,addon in specs:
                    for seed in (2025,2026,2027):
                        values=np.linspace(.70,.90,100) if dataset=='mnist' else np.linspace(.10,.35,100)
                        rows.append(dict(dataset=dataset,partition=partition,family=family,environment=env,
                            addon=addon,seed=seed,runId=f'{dataset}-{partition}-{family}-{env}-{addon}-{seed}',
                            curve=values+(seed-2026)*.01))
    files,figures=publication.render(tmp_path,rows)
    assert len(files)==8 and len(figures)==4
    assert not any(path.suffix=='.pdf' for path in tmp_path.iterdir())
    for stem,curves in figures.items():
        assert len(curves)==(8 if 'non_iid' in stem else 6)
        for curve in curves:
            assert curve['seeds']==[2025,2026,2027]
            expected=np.linspace(70,90,100) if stem.startswith('mnist') else np.linspace(10,35,100)
            np.testing.assert_allclose(curve['meanPercent'],expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(curve['sampleSDPercent'],1,atol=1e-12,rtol=0)
        svg=tmp_path/(stem+'.svg');tree=ET.parse(svg)
        assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
        assert tree.findall('.//{http://www.w3.org/2000/svg}path')
        text=svg.read_text()
        assert 'FedSharp' in text and 'Validation accuracy' not in text  # axis is explicitly the mean client measure
        assert 'Mean client validation accuracy' in text
        assert 'Dirichlet' not in text and 'SHARP' not in text and 'CART' not in text
        if 'non_iid' in stem:assert 'FedSharp+' in text and 'BASIL + ACDS' in text


def test_presentation_module_is_outside_frozen_hpc_source():
    from basil_core import basil_sharp_thesis as study
    assert 'reporting/fedsharp_publication.py' not in study.source()
