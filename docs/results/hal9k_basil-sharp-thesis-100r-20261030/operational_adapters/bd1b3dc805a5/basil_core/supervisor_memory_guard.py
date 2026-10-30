"""Conservative process-memory telemetry when a worker exits during admission."""
from pathlib import Path

from basil_core.cart_worker_admission import Resident


def residents_of(active, assigned):
    result = []
    for index, (process, _) in active.items():
        if process.poll() is not None:
            continue
        rss = 0
        try:
            lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
            values = {key: value.strip() for key, value in
                      (line.split(':', 1) for line in lines if ':' in line)}
            # A zero measurement retains the full unallocated reservation. It
            # does not declare the slot free; the scheduler must reap it first.
            if 'VmRSS' in values:
                amount = int(values['VmRSS'].split()[0]) + int(values.get('VmSwap', '0 kB').split()[0])
                if amount >= 0:
                    rss = amount // 1024
        except (OSError, ValueError, IndexError):
            pass
        result.append(Resident(process.pid, assigned[index].gpu, rss))
    return result
