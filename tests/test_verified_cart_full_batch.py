"""Gate configuration, independent channel algebra and lazy-import contracts."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import check_verified_cart_full_batch as gate


def test_default_never_starts_training():
    code=('from scripts import check_verified_cart_full_batch as g; import sys; '
          'assert not ({"tensorflow","numpy","cryptography","wandb"}&sys.modules.keys()); g.main()')
    result=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['trainingRunsStarted']==0
    c=gate.scope()
    assert c['logicalRound']==20 and c['roundsActuallyExecuted']==0
    assert c['stepsPerActivation']==50 and c['localEpochs']==5 and c['batchSize']==512 and c['lastBatchSize']==392
    assert c['sigma_e']==.01 and c['mu']==.01 and c['testExamplesLoaded']==0
    assert c['attackerIds']==[0,1,5,7] and not c['wandb'] and not c['researchValid']


@pytest.mark.parametrize('round_id',[0,19,20])
def test_channel_independent_reconstruction_and_boundary(round_id):
    import numpy as np
    params=[np.linspace(-.2,.2,117706,dtype=np.float32)]
    saved=params[0].copy()
    values=gate.channel_contract(params,round_id)
    assert all(r['attackActive']==(round_id>=20) for r in values)
    assert len({r['noiseHash'] for r in values})==5
    assert all(r['effectiveCoordinateSigma']==.01 for r in values)
    np.testing.assert_array_equal(params[0],saved)


def test_closed_peer_does_not_claim_zero_oom_counters():
    result=gate.health(SimpleNamespace(endpoints={5:SimpleNamespace(process=None)}))
    assert result['5']['memoryCountersAvailable'] is False and 'events' not in result['5']
