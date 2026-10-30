#!/usr/bin/env python3
"""Resume the frozen thesis producer with a recorded process-exit telemetry guard."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if '--resume' not in sys.argv or '--execute' not in sys.argv:
        raise SystemExit('This recovery launcher requires --resume --execute and the original --output.')
    from basil_core.supervisor_memory_guard import residents_of
    from basil_core.cart_campaign_io import write_json
    from scripts import run_basil_sharp_thesis as runner
    original_schedule = runner.adaptive.schedule

    def schedule(args, plan, hardware, assigned, digest, leases):
        names = ('basil_core/supervisor_memory_guard.py', 'scripts/resume_basil_sharp_resilient.py')
        sources = {name: runner.adaptive.sha(ROOT / name) for name in names}
        folder = args.output / 'operational_adapters'
        folder.mkdir(exist_ok=True)
        snapshots = {}
        for name, value in sources.items():
            target = folder / value[:12] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and runner.adaptive.sha(target) != value:
                raise ValueError('Preserve existing operational adapter snapshot before changing it.')
            if not target.exists():
                target.write_bytes((ROOT / name).read_bytes())
            snapshots[str(target.relative_to(args.output))] = value
        write_json(folder / ('memory-guard-' + sources[names[0]][:12] + '.json'),
            dict(kind='process_memory_telemetry_guard', planSha256=digest, source=sources,
                 snapshots=snapshots,
                 scientificProducerUnmodified=True, resourceBudgetsUnmodified=True,
                 behavior='Missing process memory retains its full reservation until the scheduler reaps the worker.'))
        return original_schedule(args, plan, hardware, assigned, digest, leases)

    with patch.object(runner.adaptive, 'residents_of', residents_of), \
            patch.object(runner.adaptive, 'schedule', schedule):
        return runner.main()


if __name__ == '__main__':
    raise SystemExit(main())
