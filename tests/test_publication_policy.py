"""Keep local machine state out of published source and evidence."""
import pytest
from scripts.check_publication_policy import forbidden


@pytest.mark.parametrize('name',['.codex','.codex/config.toml','.agents/notes.md',
    '.venv-hpc/bin/python','environment/basil-noise-env/bin/wandb',
    '.papermerge/local-tests/run/plan.json','gui/queue_state.json',
    'gui/worker_state/runtime.json','experiments/cache/mnist.npz',
    'newResults/IID/old.json','newPlots/nonIID/old.svg','browser_share.json',
    'docs/thesis-figures/fedsharp-hal9k-20261030/preview.pdf',
    'docs/thesis-figures/fedsharp-hal9k-20261030/installation.json'])
def test_runtime_and_agent_state_is_never_publishable(name):
    assert forbidden(name)


@pytest.mark.parametrize('name',['environment/requirements-hpc.txt',
    'basil_core/basil_sharp_thesis.py','tests/fixtures/legacy_eight_peer_routing.py',
    'docs/results/hal9k_basil-sharp-thesis-100r-20261030/plan.json',
    'docs/thesis-figures/fedsharp-hal9k-20261030/mnist_iid_accuracy.svg',
    'docs/thesis-figures/fedsharp-hal9k-20261030/mnist_iid_accuracy.png'])
def test_source_specs_fixtures_and_curated_evidence_remain_publishable(name):
    assert not forbidden(name)
