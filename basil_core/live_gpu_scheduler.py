"""Operational GPU work stealing and live controller handover, outside producers.

An existing CUDA process cannot migrate GPUs. Adopt it until it exits, and
dispatch untouched experiments to any verified, healthy GPU with memory.
"""
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time

from basil_core.cart_campaign_io import read_json, write_json
from basil_core.cart_worker_admission import admission, placement
from basil_core.execution_profile import ExecutionProfile, gpu_inventory, gpu_processes
from basil_core.supervisor_memory_guard import residents_of

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'scripts/basil_sharp_thesis_worker.py'
OWNERS = {'run_basil_sharp_thesis.py', 'resume_basil_sharp_resilient.py',
          'rebalance_basil_sharp_live.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argv_value(argv, flag):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return next((v.split('=', 1)[1] for v in argv if v.startswith(flag + '=')), None)


def identity(pid):
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid():
        raise ValueError('Process belongs to a different user.')
    # The command name can contain spaces and parentheses.
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    return dict(pid=pid, start=stat[19], ppid=int(stat[1]), argv=argv)


def same_process(expected):
    try:
        current = identity(expected['pid'])
        return current['start'] == expected['start'] and current['argv'] == expected['argv']
    except (OSError, ValueError, IndexError):
        return False


class Adopted:
    """Wait on the original process identity, including orphaned workers."""
    def __init__(self, expected):
        self.pid = expected['pid']
        self.expected = expected
        self.fd = os.pidfd_open(self.pid)
        if not same_process(expected):
            os.close(self.fd)
            raise ProcessLookupError('Worker exited during adoption.')

    def poll(self):
        # Exit code is unavailable for nonchildren; run.json is authoritative.
        return 0 if select.select([self.fd], [], [], 0)[0] else None

    def close(self):
        os.close(self.fd)


def discover(root, config_count):
    workers = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            item = identity(int(proc.name)); argv = item['argv']
            if len(argv) < 2 or Path(argv[1]).resolve() != WORKER:
                continue
            output = argv_value(argv, '--output')
            if output is None or Path(output).resolve() != root:
                continue
            index = int(argv_value(argv, '--worker'))
            if not 0 <= index < config_count or index in workers:
                raise RuntimeError('Duplicate or invalid worker index; handover refused.')
            raw = proc.joinpath('environ').read_bytes().split(b'\0')
            env = dict(v.split(b'=', 1) for v in raw if b'=' in v)
            profile = ExecutionProfile(**json.loads(env[b'PAPERMERGE_EXECUTION_PROFILE'])).validate()
            item['profile'] = asdict(profile)
            workers[index] = item
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return workers


def verified(root):
    from basil_core import basil_sharp_thesis as study
    plan = read_json(root / 'plan.json')
    if plan['revision'] != study.REVISION or plan['host'] != socket.gethostname():
        raise ValueError('This launcher requires the original thesis plan and server.')
    study.check_source(plan['source'])
    hardware, assigned = placement(plan)
    digest = sha(root / 'plan.json')
    for gpu, profile in hardware.items():
        receipt = read_json(root / f'revision_gpu_{gpu}.json')
        if (receipt.get('status') != 'PASSED' or receipt.get('planSha256') != digest or
                receipt.get('source') != plan['source'] or
                receipt['runtime']['host'] != socket.gethostname() or
                receipt['runtime']['profile'] != asdict(profile)):
            raise ValueError('Physical GPU verification does not match the frozen plan.')
    return plan, hardware, assigned, digest


def controller(root):
    row = read_json(root / 'campaign.json')
    if row.get('host') != socket.gethostname():
        raise ValueError('Supervisor belongs to another server.')
    pid = row.get('ownerPid')
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        return None
    try:
        item = identity(pid)
    except FileNotFoundError:
        return None
    argv = item['argv']
    script = Path(argv[1]).resolve() if len(argv) > 1 else Path('/')
    if (script.parent != ROOT / 'scripts' or script.name not in OWNERS or
            '--execute' not in argv or
            Path(argv_value(argv, '--output') or '/').resolve() != root):
        raise ValueError('Recorded owner is not this campaign supervisor.')
    if row.get('status') not in ('RUNNING', 'WAITING_FOR_RESOURCES', 'SUPERVISOR_RECOVERY_NEEDED'):
        raise ValueError('Supervisor is not scheduling verified training.')
    return item


def replace_controller(root, owner, workers):
    """Terminate only a validated controller; never signal a training worker.

    Its SIGTERM handler stops training children, so SIGKILL intentionally avoids
    that handler. Children have no parent-death signal and retain GPU leases.
    Every active worker has a pidfd before this operation; no work is restarted.
    """
    if owner is None:
        return
    if owner['pid'] in {v['pid'] for v in workers.values()}:
        raise ValueError('Controller and training PID overlap.')
    fd = os.pidfd_open(owner['pid'])
    try:
        if not same_process(owner):
            raise ProcessLookupError('Supervisor identity changed; no signal sent.')
        # Opening pidfd before identity validation prevents PID reuse races.
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        if not select.select([fd], [], [], 10)[0]:
            raise RuntimeError('Controller has not exited; workers remain untouched.')
    finally:
        os.close(fd)


def eligible_profile(root, cfg, hardware):
    folder = root / cfg['runId']
    row = read_json(folder / 'run.json') if (folder / 'run.json').exists() else {}
    if (folder / 'checkpoint.npz').exists() or row.get('contract'):
        saved = row.get('contract', {}).get('runtime', row.get('runtime', {})).get('profile')
        if not saved:
            raise ValueError('Existing checkpoint has no recorded resource contract.')
        p = ExecutionProfile(**saved).validate()
        if p.gpu not in hardware or p != hardware[p.gpu]:
            raise ValueError('Checkpoint GPU profile differs from verified hardware.')
        return {p.gpu}
    return set(hardware)


def choose(pending, configs, allowed, hardware, residents, inventory, processes,
           available_ram, budget, maximum, quarantined, leases):
    """Shared queue; balance active counts and then usable free memory."""
    counts = Counter(r.gpu for r in residents)
    free = {r['index']: r['memory.free'] for r in inventory}
    candidates = []
    decisions = {}
    for gpu, p in hardware.items():
        if gpu in quarantined or gpu not in leases or gpu not in free:
            continue
        decision = admission(p, residents, inventory, processes, available_ram,
            context_mib=budget['context_overhead_mib'],
            ram_headroom_mib=budget['ram_headroom_mib'], maximum=maximum)
        decisions[gpu] = decision
        if not decision['admit']:
            continue
        index = next((i for i in pending if gpu in allowed[i]), None)
        if index is not None:
            candidates.append((counts[gpu], -free[gpu], pending.index(index), gpu, index))
    if not candidates:
        return None, decisions
    *_, gpu, index = min(candidates)
    return (index, hardware[gpu]), decisions


def gpu_failure(text):
    value = text.lower()
    return any(s in value for s in ('resourceexhausted', 'out of memory', 'out_of_memory',
        'cuda_error', 'cuda error', 'gpu has fallen off', 'device is lost', 'xid error'))


def notify(root, topic, title, message):
    event = dict(at=time.time(), title=title, message=message)
    if topic:
        try:
            from basil_core.ntfy_notifications import publish
            event['delivery'] = publish(topic, title, message)
        except Exception as error:
            event['deliveryError'] = type(error).__name__
    with (root / 'operational_adapters' / 'live-notifications.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')


def lease_available(profile, leases):
    if profile.gpu in leases:
        return True
    path = Path('/tmp') / f'papermerge-{os.getuid()}-{profile.gpu_uuid}.lock'
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    leases[profile.gpu] = handle
    return True


def execute(args):
    root = args.output.resolve()
    plan, hardware, assigned, digest = verified(root)
    folder = root / 'operational_adapters'; folder.mkdir(exist_ok=True)
    with (root / 'live_scheduler.lock').open('a') as operation:
        fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = controller(root)
        # Freeze only dispatch for the few milliseconds needed to collect ALL
        # children; this does not pause a single training worker.
        owner_fd = None; replaced = False
        if owner:
            owner_fd = os.pidfd_open(owner['pid'])
            if not same_process(owner):
                os.close(owner_fd); raise ProcessLookupError('Controller exited before handover.')
            signal.pidfd_send_signal(owner_fd, signal.SIGSTOP)
        active = {}; leases = {}
        try:
            # Confirm stop before /proc scan, avoiding a concurrent worker launch.
            if owner:
                deadline = time.monotonic() + 5
                while True:
                    state = Path(f'/proc/{owner["pid"]}/status').read_text()
                    if any(line.startswith('State:') and line.split()[1] in ('T','t') for line in state.splitlines()):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('Could not suspend dispatch; training remains untouched.')
                    time.sleep(.01)
            found = discover(root, len(assigned))
            if owner:
                children = Path(f'/proc/{owner["pid"]}/task/{owner["pid"]}/children').read_text().split()
                for child in set(map(int, children)) - {v['pid'] for v in found.values()}:
                    try:
                        state = Path(f'/proc/{child}/stat').read_text().rsplit(')',1)[1].split()[0]
                    except FileNotFoundError:
                        continue
                    if state != 'Z':
                        raise ValueError('Unrecognized supervisor children; handover refused.')
            for i, item in found.items():
                p = ExecutionProfile(**item['profile'])
                if p.gpu not in hardware or p != hardware[p.gpu]:
                    raise ValueError('Active worker has an unverified GPU profile.')
                try:
                    active[i] = (Adopted(item), None)
                except ProcessLookupError:
                    continue
                assigned[i] = p
            # Discover incompatible recovery contracts while the old controller
            # can still be resumed. Never strand it on a pre-existing bad row.
            for i,cfg in enumerate(plan['configs']):
                path = root/cfg['runId']/'run.json'
                row = read_json(path) if path.exists() else {}
                if i not in active and row.get('status') not in ('COMPLETED','FAILED'):
                    eligible_profile(root,cfg,hardware)
            receipt = dict(kind='live_gpu_work_stealing', planSha256=digest,
                previousOwner=owner, adoptedWorkers=found, scientificProducerUnmodified=True,
                resourceBudgetsUnmodified=True, activeWorkersSignalled=False)
            for name in ('basil_core/live_gpu_scheduler.py', 'scripts/rebalance_basil_sharp_live.py'):
                value = sha(ROOT / name); target = folder / value[:12] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists(): target.write_bytes((ROOT / name).read_bytes())
                if sha(target) != value: raise ValueError('Operational snapshot differs.')
                receipt.setdefault('source', {})[name] = value
            stamp = str(time.time_ns())
            write_json(folder / f'live-handover-{stamp}.json', receipt)
            replace_controller(root, owner, found)
            replaced = True
            if owner_fd is not None: os.close(owner_fd); owner_fd = None
            with (root / 'supervisor.lock').open('a') as supervisor_lock:
                fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f'Adopted {len(active)} training workers without stopping them.', flush=True)
                return schedule(args, plan, hardware, assigned, digest, active, leases)
        finally:
            if owner_fd is not None:
                # Validation failure resumes the old controller, not its workers.
                if not replaced:
                    try: signal.pidfd_send_signal(owner_fd, signal.SIGCONT)
                    except ProcessLookupError: pass
                os.close(owner_fd)
            for process, log in active.values():
                if isinstance(process, Adopted): process.close()
                if log: log.close()
            for handle in leases.values(): handle.close()


def schedule(args, plan, hardware, assigned, digest, active, leases):
    from scripts.run_cart_v2_study import worker_environment
    from scripts.run_cart_v2_scaled import available_ram_mib
    from basil_core import basil_sharp_thesis as study
    root = args.output; configs = plan['configs']; budget = plan['resourceBudget']
    pending = []; completed = set(); failures = []
    for i, cfg in enumerate(configs):
        path = root / cfg['runId'] / 'run.json'
        row = read_json(path) if path.exists() else {}
        if i in active: continue
        if row.get('status') == 'COMPLETED': completed.add(i)
        elif row.get('status') == 'FAILED':
            failures.append(dict(index=i, runId=cfg['runId'], error=row.get('error','Earlier worker failure')))
        else: pending.append(i)
    allowed = {i: eligible_profile(root, configs[i], hardware) for i in pending}
    old = read_json(root / 'campaign.json')
    base_elapsed = old.get('elapsedSeconds', 0) or 0
    started = time.monotonic(); stopped = False; status = 'RUNNING'
    quarantine_path = root / 'operational_adapters' / 'gpu-quarantine.json'
    quarantined = read_json(quarantine_path) if quarantine_path.exists() else {}
    quarantined = {int(k): v for k,v in quarantined.items()}
    blocked = {}; telemetry_delayed = False

    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

    def campaign():
        lanes = {cfg['runId']: f'{assigned[i].gpu}/live{i}' for i,cfg in enumerate(configs)}
        row = dict(status=status, total=len(configs), completed=len(completed), failed=len(failures),
            pending=len(pending), activeSlots=list(active), activeRunIds=[configs[i]['runId'] for i in active],
            pendingRunIds=[configs[i]['runId'] for i in pending], failures=failures,
            workerFailurePolicy='continue', runtimeAdapter=plan['revision'], host=socket.gethostname(),
            profile='hpc', ownerPid=os.getpid(), elapsedSeconds=base_elapsed + time.monotonic()-started,
            scheduler=dict(mode='adaptive', dispatch='shared_queue_verified_gpus', maximumPerGpu=args.maximum,
                activePerGpu=dict(Counter(assigned[i].gpu for i in active)), blocked=blocked,
                quarantined=quarantined, lanesByRunId=lanes,
                gpuByRunId={cfg['runId']:assigned[i].gpu for i,cfg in enumerate(configs)}))
        write_json(root / 'campaign.json', row)

    def quarantine(gpu, reason):
        if gpu in quarantined: return
        quarantined[gpu] = dict(reason=reason, at=time.time())
        write_json(quarantine_path, quarantined)
        notify(root, args.ntfy_topic, f'PaperMerge GPU {gpu} paused',
            f'{reason}. No new work will start on this GPU. Untouched queued configurations can use other verified GPUs; active workers are not stopped.')

    notify(root, args.ntfy_topic, 'PaperMerge live balancing enabled',
        f'Adopted {len(active)} active workers without restarting training. New configurations use a shared queue with GPU and host RAM reservations.')
    campaign()
    try:
        while pending or active:
            for i, (process, log) in list(active.items()):
                code = process.poll()
                if code is None: continue
                if log: log.close()
                if isinstance(process, Adopted): process.close()
                del active[i]
                path = root / configs[i]['runId'] / 'run.json'
                row = read_json(path) if path.exists() else {}
                if code == 0 and row.get('status') == 'COMPLETED':
                    completed.add(i)
                else:
                    reason = row.get('error') or f'Worker exited {code} without a completed result.'
                    row.update(status='FAILED', error=reason, workerExitCode=code)
                    path.parent.mkdir(exist_ok=True); write_json(path, row)
                    event = dict(index=i, runId=configs[i]['runId'], gpu=assigned[i].gpu, exitCode=code, error=reason)
                    failures.append(event); write_json(root / 'failed_runs.json', failures)
                    with (root/'worker_failure_events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
                    notify(root, args.ntfy_topic, 'PaperMerge worker failed',
                        f'{configs[i]["runId"]} on GPU {assigned[i].gpu} was set aside. {len(pending)} configurations remain queued. Other workers continue.')
                    tail = ''
                    logpath = root / (configs[i]['runId']+'.log')
                    if logpath.exists():
                        with logpath.open('rb') as stream:
                            stream.seek(max(0, logpath.stat().st_size-16384)); tail=stream.read().decode(errors='replace')
                    if gpu_failure(reason+'\n'+tail): quarantine(assigned[i].gpu, 'CUDA memory/device failure detected in a worker')
            if stopped:
                status='SUPERVISOR_RECOVERY_NEEDED'; campaign()
                print('Dispatch stopped. Training workers continue; rerun this launcher to adopt them.',flush=True)
                return 4
            study.check_source(plan['source'])
            if sha(root/'plan.json') != digest: raise ValueError('Frozen plan changed.')
            try:
                latest_inventory = gpu_inventory()
                present = {row['index'] for row in latest_inventory}
                for gpu in hardware:
                    if gpu not in present: quarantine(gpu, 'GPU missing from NVIDIA inventory')
                for gpu,p in hardware.items():
                    if gpu not in quarantined and gpu in present:
                        try:
                            p.bind(latest_inventory)
                        except RuntimeError:
                            quarantine(gpu, 'Physical GPU identity changed')
                            continue
                        lease_available(p, leases)
                # Sample and reserve anew after EVERY launch, including startup
                # workers which have not allocated RAM or a CUDA context yet.
                for _ in hardware:
                    resident = residents_of(active, assigned)
                    choice, blocked = choose(pending, configs, allowed, hardware, resident,
                        gpu_inventory(), gpu_processes(), available_ram_mib(), budget,
                        args.maximum, quarantined, leases)
                    if choice is None: break
                    i,p = choice
                    fd = leases[p.gpu].fileno(); cfg = configs[i]
                    log = (root/(cfg['runId']+'.log')).open('a')
                    try:
                        process = subprocess.Popen([sys.executable,str(WORKER),'--worker',str(i),
                            '--output',str(root),'--lease-fd',str(fd)], cwd=ROOT,
                            env=worker_environment(p),pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT)
                    except OSError:
                        log.close(); raise
                    assigned[i] = p; active[i]=(process,log); pending.remove(i)
                    with (root/'operational_adapters'/'live-dispatch.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(at=time.time(),runId=cfg['runId'],pid=process.pid,profile=asdict(p)))+'\n')
                    print(f'GPU {p.gpu}: started {cfg["runId"]}; {len(active)} total active',flush=True)
                status = 'RUNNING' if active else 'WAITING_FOR_RESOURCES'
                telemetry_delayed = False
            except (OSError, RuntimeError) as error:
                # Unknown telemetry never permits extra allocations.
                blocked={'telemetry':type(error).__name__}; status='RUNNING' if active else 'WAITING_FOR_RESOURCES'
                if not telemetry_delayed:
                    notify(root,args.ntfy_topic,'PaperMerge resource checks delayed',
                        'GPU/RAM checks are unavailable. New launches are paused; active training continues. The scheduler will retry automatically.')
                telemetry_delayed = True
            campaign()
            if pending or active: time.sleep(args.poll_seconds)
        status='COMPLETED_WITH_FAILURES' if failures else 'COMPLETED'
        if not failures:
            try:
                from reporting.basil_sharp_thesis import generate
                generate(root, root/'thesis_figures')
            except Exception as error:
                status='COMPLETED_REPORT_FAILED'
                write_json(root/'report_status.json',dict(trainingComplete=True,status=status,error=str(error)))
        campaign()
        notify(root,args.ntfy_topic,'PaperMerge campaign finished',
            f'{len(completed)}/{len(configs)} completed; {len(failures)} failed. Status: {status}.')
        return 2 if failures else (3 if status=='COMPLETED_REPORT_FAILED' else 0)
    except BaseException:
        status='SUPERVISOR_RECOVERY_NEEDED'; campaign()
        notify(root,args.ntfy_topic,'PaperMerge supervisor needs recovery',
            'Scheduling stopped after an operational error. Active training workers were left running. Inspect live_scheduler.log and rerun the live balancing launcher.')
        raise
