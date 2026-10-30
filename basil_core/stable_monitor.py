"""Private, shared-home publisher state for one monitor URL across run folders."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import uuid

STATE_ROOT = Path.home() / '.config/papermerge/live-monitor'


def write_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def connection(root, site, state, create, view_url):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved_path = state / 'connection.json'
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
        elif (Path(root) / 'browser_share.json').exists():
            # Preserve the URL the user may already have shared.
            saved = json.loads((Path(root) / 'browser_share.json').read_text())
        else:
            saved = create(root, site)
        if saved['site'] != site:
            raise ValueError('The permanent monitor already uses a different dashboard service.')
        saved['viewUrl'] = view_url(saved)
        write_private(saved_path, saved)
        return saved


def select_run(state, root):
    value = dict(output=str(Path(root).resolve()), host=socket.gethostname(), selection=uuid.uuid4().hex)
    write_private(Path(state) / 'selected.json', value)
    return value


def selected_run(state):
    value = json.loads((Path(state) / 'selected.json').read_text())
    if value['host'] != socket.gethostname():
        return None
    return value


@contextmanager
def publisher_lock(args):
    stable = getattr(args, 'stable', False)
    path = (args.state_dir / 'publisher.lock') if stable else args.output / (
        'browser_temporary_publisher.lock' if getattr(args, 'temporary', False) else 'browser_publisher.lock')
    inherited = getattr(args, 'publisher_lock_fd', None)
    if inherited is not None:
        if os.fstat(inherited).st_ino != path.stat().st_ino or os.fstat(inherited).st_dev != path.stat().st_dev:
            raise ValueError('Publisher lock descriptor differs from the permanent monitor lock.')
        handle = os.fdopen(inherited, 'a')
    else:
        handle = path.open('a')
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stable:
            write_private(args.state_dir / 'publisher.json', dict(pid=os.getpid(), host=socket.gethostname()))
        yield


def retire_legacy_publisher(root, script):
    """Stop only old reporter processes for this exact output, never trainers."""
    root, script = Path(root).resolve(), Path(script).resolve()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        descriptor = None
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            descriptor = os.pidfd_open(int(entry.name))
            words = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if '--serve-stable' in words or '--temporary' in words:
                continue
            cwd = (entry / 'cwd').resolve()
            if not any((Path(word) if Path(word).is_absolute() else cwd / word).resolve() == script
                       for word in words[1:] if word and not word.startswith('-')):
                continue
            if '--output' in words:
                value = words[words.index('--output') + 1]
            else:
                value = next(word.split('=', 1)[1] for word in words if word.startswith('--output='))
            target = Path(value) if Path(value).is_absolute() else cwd / value
            if target.resolve() == root:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except (OSError, ValueError, StopIteration, IndexError):
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)


def stop_old_monitors(project, state):
    """Stop this user's read-only project monitors; retain the permanent daemon."""
    project, state = Path(project).resolve(), Path(state).resolve()
    browser = project / 'scripts/share_cart_monitor.py'
    terminals = {project / 'scripts/run_cart_comparison.py', project / 'scripts/run_sharp_study.py'}
    hpc = project / 'scripts/hpc.py'
    stopped, kept = [], []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        descriptor = None
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            descriptor = os.pidfd_open(int(entry.name))
            words = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            cwd = (entry / 'cwd').resolve()
            positions = {index:(Path(word) if Path(word).is_absolute() else cwd / word).resolve()
                         for index,word in enumerate(words[1:],1) if word and not word.startswith('-')}
            paths = set(positions.values())
            kind, stop_signal = None, signal.SIGTERM
            if browser in paths:
                value = next((word.split('=', 1)[1] for word in words if word.startswith('--state-dir=')), None)
                if '--state-dir' in words:
                    value = words[words.index('--state-dir') + 1]
                folder = Path(value) if value else STATE_ROOT
                if not folder.is_absolute():folder = cwd / folder
                if '--serve-stable' in words and folder.resolve() == state:
                    kept.append(int(entry.name));continue
                kind = 'browser publisher'
            elif (paths & terminals and '--monitor' in words and '--execute' not in words) or (
                    any(path==hpc and words[index+1:index+2]==['monitor'] for index,path in positions.items())
                    and '--execute' not in words):
                kind, stop_signal = 'terminal monitor', signal.SIGINT
            if kind:
                signal.pidfd_send_signal(descriptor, stop_signal)
                stopped.append(dict(pid=int(entry.name), kind=kind))
        except (OSError, ValueError, IndexError):
            continue
        finally:
            if descriptor is not None:os.close(descriptor)
    return dict(stopped=stopped, permanentPublishersKept=kept)


def launch_background(args, script):
    """Reuse a local publisher, or start one with the already-owned global lock."""
    lock = (args.state_dir / 'publisher.lock').open('a')
    deadline = time.monotonic() + 20
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            owner_path = args.state_dir / 'publisher.json'
            owner = json.loads(owner_path.read_text()) if owner_path.exists() else {}
            if owner.get('host') == socket.gethostname():
                lock.close()
                return False
            if time.monotonic() >= deadline:
                lock.close()
                raise RuntimeError('The previous server publisher has not handed over yet; retry this command.')
            time.sleep(.2)
    try:
        command = [os.sys.executable, str(script), '--output', str(args.output), '--site', args.site,
                   '--interval', str(args.interval), '--state-dir', str(args.state_dir),
                   '--serve-stable', '--publisher-lock-fd', str(lock.fileno())]
        if args.ntfy_topic:
            command.extend(['--ntfy-topic', args.ntfy_topic])
        with (args.state_dir / 'publisher.log').open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        write_private(args.state_dir / 'publisher.json', dict(pid=process.pid, host=socket.gethostname()))
        return True
    finally:
        lock.close()
