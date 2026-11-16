"""Conservative live admission; never change a worker's numerical profile."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Resident:
    pid: int
    gpu: int
    rss_mib: int


def admission(profile, residents, inventory, processes, available_ram, *,
              context_mib, ram_headroom_mib, maximum):
    """Reserve full future allocations, including workers with no CUDA context yet.

    Free memory already excludes resident allocations. Subtract only their
    *unallocated* budgets, rather than double-counting either free memory or
    startup workers. Other users' allocations are never credited to us.
    """
    profile.bind(inventory)
    same = [r for r in residents if r.gpu == profile.gpu]
    if len(same) >= maximum:
        return dict(admit=False, reason='worker ceiling')
    if processes is None:
        return dict(admit=False, reason='GPU process telemetry unavailable')
    allocated = Counter()
    try:
        for row in processes:
            if row['gpu_uuid'].strip() == profile.gpu_uuid:
                allocated[int(row['pid'])] += int(row['memory_mib'])
    except (ValueError, TypeError, KeyError):
        return dict(admit=False, reason='GPU process allocation is unknown')
    budget = profile.memory_mib + context_mib
    gpu_reserve = sum(max(0, budget - allocated[r.pid]) for r in same)
    # All profiles in an admitted cohort use the same host budget. A caller
    # must enforce that property before entering the scheduling loop.
    ram_reserve = sum(max(0, profile.ram_mib - r.rss_mib) for r in residents)
    gpu_free = next(r['memory.free'] for r in inventory if r['index'] == profile.gpu)
    gpu_needed = profile.headroom_mib + gpu_reserve + budget
    ram_needed = ram_headroom_mib + ram_reserve + profile.ram_mib
    ok = gpu_free >= gpu_needed and available_ram >= ram_needed
    return dict(admit=ok, reason='fits' if ok else 'waiting for memory',
                gpuFreeMiB=gpu_free, gpuRequiredMiB=gpu_needed,
                ramAvailableMiB=available_ram, ramRequiredMiB=ram_needed)


def placement(plan):
    """Retain every experiment's original GPU/profile for checkpoint recovery."""
    from basil_core.execution_profile import ExecutionProfile
    slots = [ExecutionProfile(**p).validate() for p in plan['profiles']]
    if not slots or not plan['configs']:
        raise ValueError('A nonempty verified plan is required.')
    hardware = {}
    for p in slots:
        if p.gpu in hardware and hardware[p.gpu] != p:
            raise ValueError('Conflicting profiles on one physical GPU.')
        hardware[p.gpu] = p
    if len({p.ram_mib for p in slots}) != 1:
        raise ValueError('Adaptive admission requires equal per-worker RAM budgets.')
    assigned = [slots[i % len(slots)] for i in range(len(plan['configs']))]
    return hardware, assigned
