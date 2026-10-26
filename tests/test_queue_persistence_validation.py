"""Malformed queues must never silently run a smaller subset of experiments."""
import json

import pytest


@pytest.mark.parametrize('bad',[None,False,17,'experiment',[]])
def test_non_object_entry_blocks_whole_queue_without_rewriting(tmp_path,bad):
    from gui.services.queue_service import QueueService
    path=tmp_path/'queue.json'
    path.write_text(json.dumps([{'experimentName':'A'},bad,{'experimentName':'C'}]))
    original=path.read_bytes()
    with pytest.raises(ValueError,match='Queue entry 2'):
        QueueService(path).load()
    assert path.read_bytes()==original


@pytest.mark.parametrize('bad',[None,[],False,'config.json'])
def test_enriched_queue_requires_a_configuration_object(tmp_path,bad):
    from gui.services.queue_service import QueueService
    path=tmp_path/'queue.json';path.write_text(json.dumps([{'config':bad}]))
    with pytest.raises(ValueError,match='invalid configuration object'):
        QueueService(path).load()
