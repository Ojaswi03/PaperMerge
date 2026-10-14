"""Kernel-enforced local replica budgets; no best-effort fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import uuid

MIB=1024**2
REPLICA_MEMORY_BYTES=2048*MIB
MAX_REPLICAS=10
POOL_MEMORY_CEILING_BYTES=MAX_REPLICAS*REPLICA_MEMORY_BYTES


@dataclass(frozen=True)
class ResourceScope:
    unit: str
    memory_bytes: int = REPLICA_MEMORY_BYTES
    tasks: int = 64

    def __post_init__(self):
        if re.fullmatch(r'papermerge-replica-[0-9a-f]{32}\.scope',self.unit) is None:
            raise ValueError('Only dedicated PaperMerge replica scopes may be managed.')

    @classmethod
    def create(cls,*,memory_bytes=REPLICA_MEMORY_BYTES,tasks=64):
        if type(memory_bytes) is not int or not 16*MIB<=memory_bytes<=REPLICA_MEMORY_BYTES:
            raise ValueError('Bounded positive replica RAM limit required.')
        if type(tasks) is not int or not 4<=tasks<=64:raise ValueError('Task budget 4..64 required.')
        return cls('papermerge-replica-'+uuid.uuid4().hex+'.scope',memory_bytes,tasks)

    def command(self,command):
        return ['systemd-run','--user','--scope','--quiet','--unit='+self.unit,
            '--property=MemoryMax='+str(self.memory_bytes),'--property=MemorySwapMax=0',
            '--property=OOMPolicy=kill','--property=TasksMax='+str(self.tasks),
            '--property=CPUQuota=100%',*command]

    def verify(self):
        # Read kernel controllers, not only the requested systemd properties.
        result=subprocess.run(['systemctl','--user','show',self.unit,'--property=ControlGroup','--value'],
            check=True,capture_output=True,text=True,timeout=10)
        relative=result.stdout.strip()
        if not relative.startswith('/user.slice/') or '..' in Path(relative).parts:
            raise RuntimeError('Unexpected replica cgroup location.')
        root=Path('/sys/fs/cgroup')/relative.lstrip('/')
        values={name:(root/name).read_text().strip() for name in
            ('memory.max','memory.swap.max','memory.oom.group','pids.max','cpu.max')}
        quota,period=values['cpu.max'].split()
        if (values['memory.max']!=str(self.memory_bytes) or values['memory.swap.max']!='0' or
                values['memory.oom.group']!='1' or values['pids.max']!=str(self.tasks) or
                quota=='max' or int(quota)>int(period)):
            raise RuntimeError('Kernel resource limits do not match the required budget.')
        return dict(unit=self.unit,cgroup=relative,kernelLimits=values,
            memoryBytes=self.memory_bytes,tasks=self.tasks,cpuCoresMaximum=1)

    def stop(self):
        # Only this unique scope created by this object; never stop other jobs.
        subprocess.run(['systemctl','--user','stop',self.unit],check=False,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)


def require_host_headroom():
    fields={line.split(':',1)[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    required=POOL_MEMORY_CEILING_BYTES+4*1024**3
    if fields['MemAvailable']<required:
        raise RuntimeError(f'Insufficient RAM for bounded pool: {fields["MemAvailable"]} available, {required} required.')
    return dict(**fields,requiredBytes=required,poolMaximumBytes=POOL_MEMORY_CEILING_BYTES,
        outsidePoolHeadroomBytes=4*1024**3)
