"""Pinned HPC dependencies satisfy the supported TensorFlow release metadata."""
from pathlib import Path

from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[1]


def pinned_requirements(filename):
    requirements = {}
    for line in (ROOT / 'environment' / filename).read_text().splitlines():
        line = line.partition('#')[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        versions = list(requirement.specifier)
        assert len(versions) == 1 and versions[0].operator == '=='
        requirements[requirement.name] = versions[0].version
    return requirements


# Published Requires-Dist entries for the direct dependencies we pin:
# https://pypi.org/pypi/tensorflow/2.21.0/json
@pytest.mark.parametrize('declared', [
    'h5py>=3.11.0,<3.15.0',
    'keras>=3.12.0',
    'numpy>=1.26.0',
])
def test_hpc_direct_pins_satisfy_tensorflow_requirements(declared):
    pins = pinned_requirements('requirements-hpc.txt')
    assert pins['tensorflow'] == '2.21.0', 'Update the release metadata contract for a different TensorFlow version.'
    requirement = Requirement(declared)
    assert pins[requirement.name] in requirement.specifier


def test_hpc_h5py_compatibility_does_not_change_legacy_environment():
    local = pinned_requirements('requirements.txt')
    hpc = pinned_requirements('requirements-hpc.txt')
    assert local['tensorflow'] == '2.18.0'
    assert local['h5py'] == '3.15.1'
    assert local['tensorflow'] != hpc['tensorflow']
    assert local['h5py'] != hpc['h5py']
