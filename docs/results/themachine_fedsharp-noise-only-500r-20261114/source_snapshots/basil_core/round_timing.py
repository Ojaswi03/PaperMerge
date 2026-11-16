"""Operational timing only; model states and scientific callbacks are unchanged."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import patch
import uuid


@contextmanager
def recording(output, index):
    import basil_core.sharp_cart_v2_protocol as protocol
    root = Path(output)
    plan = json.loads((root / 'plan.json').read_text())
    view = plan['configs'][index]
    folder = root / view['runId']
    folder.mkdir(exist_ok=True)
    original = protocol.run

    def measured(cfg, *args, **kwargs):
        callback = kwargs.get('on_round')
        initial = kwargs.get('state')
        cursor = 0 if initial is None else initial['cursor']
        started = previous = time.monotonic()
        segment = uuid.uuid4().hex

        def append(row):
            with (folder / 'round_timings.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(segment=segment, **row), sort_keys=True) + '\n')

        def on_round(row, state):
            nonlocal cursor, previous
            now = time.monotonic()
            end = state['cursor']
            append(dict(kind='round', displayRound=row['round'] + 1,
                startCursor=cursor, endCursor=end, activations=end - cursor,
                fullRound=end - cursor == cfg['nodes'], elapsedSeconds=now - previous,
                timingIncludes='protocol work, validation and preceding checkpoint/status I/O'))
            cursor, previous = end, now
            if callback:
                callback(row, state)

        kwargs['on_round'] = on_round
        append(dict(kind='segment_start', cursor=cursor,
                    utc=datetime.now(timezone.utc).isoformat()))
        try:
            return original(cfg, *args, **kwargs)
        finally:
            append(dict(kind='segment_end', elapsedSeconds=time.monotonic() - started,
                        utc=datetime.now(timezone.utc).isoformat()))

    with patch.object(protocol, 'run', measured):
        yield
