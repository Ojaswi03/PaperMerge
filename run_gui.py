
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Exit code the GUI sends to request an immediate reload
RELOAD_EXIT_CODE = 42

# Directories and files watched for changes
_WATCH_TARGETS = [
    os.path.join(PROJECT_ROOT, 'basil_core'),
    os.path.join(PROJECT_ROOT, 'gui'),
    os.path.join(PROJECT_ROOT, 'scripts'),
    os.path.join(PROJECT_ROOT, 'noise_comm'),
    os.path.join(PROJECT_ROOT, 'run_gui.py'),
]


# ── GUI subprocess mode ────────────────────────────────────────────────────────
def _run_gui():
    """Run the actual GUI. Invoked when --gui flag is present."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    from gui.app import main

    print("""
====================================================================
          BASIL + Noisy Channel Experiment GUI

   A graphical interface for configuring and running experiments
====================================================================
Starting GUI...
    """)

    main()


# ── Watcher / launcher mode ────────────────────────────────────────────────────
def _collect_mtimes():
    """Return {filepath: mtime} for every .py file under watched targets."""
    mtimes = {}
    for target in _WATCH_TARGETS:
        if os.path.isfile(target):
            try:
                mtimes[target] = os.stat(target).st_mtime
            except OSError:
                pass
        elif os.path.isdir(target):
            for dirpath, _, files in os.walk(target):
                for fname in files:
                    if fname.endswith('.py'):
                        path = os.path.join(dirpath, fname)
                        try:
                            mtimes[path] = os.stat(path).st_mtime
                        except OSError:
                            pass
    return mtimes


# Written by the GUI while a queue is running; the watcher defers restarts
# while it exists so training workers are never orphaned by a code change.
_QUEUE_SENTINEL = os.path.join(PROJECT_ROOT, 'gui', '.queue_active.json')


def _launch_subprocess():
    import subprocess
    # New session so the GUI and every worker it spawns share a process
    # group the launcher can terminate as a tree.
    return subprocess.Popen(
        [sys.executable, __file__, '--gui'], start_new_session=True
    )


def _stop_gui_tree(proc):
    """Terminate the GUI and all of its worker processes.

    Workers trap SIGTERM as a graceful stop request, so the whole group is
    hard-killed once the GUI is gone. Worker saves are atomic and stopped
    runs stay in the queue, so this loses at most the in-flight round.
    Without this, research workers survive the GUI and keep training on the
    GPU invisibly.
    """
    import signal
    import subprocess

    def _signal_tree(signum):
        # start_new_session=True makes the GUI its own group leader, so its
        # PID doubles as the process-group ID even after the GUI is reaped.
        try:
            os.killpg(proc.pid, signum)
            return True
        except (OSError, ProcessLookupError):
            return False

    if not _signal_tree(signal.SIGTERM):
        proc.terminate()
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        pass
    if not _signal_tree(signal.SIGKILL):
        proc.kill()
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        pass


def _queue_is_active(proc):
    return proc.poll() is None and os.path.exists(_QUEUE_SENTINEL)


def _run_watcher():
    import time

    mtimes = _collect_mtimes()
    proc = _launch_subprocess()
    deferred_announced = False

    try:
        while True:
            time.sleep(0.8)

            retcode = proc.poll()

            # Manual reload requested by the GUI (Reload button)
            if retcode == RELOAD_EXIT_CODE:
                print('\n[Launcher] Reload requested - restarting GUI...\n')
                mtimes = _collect_mtimes()
                proc = _launch_subprocess()
                deferred_announced = False
                continue

            # GUI closed normally (user clicked Exit / closed window)
            if retcode is not None:
                print(f'\n[Launcher] GUI exited (code {retcode}). Shutting down.')
                break

            # Check for file changes
            new_mtimes = _collect_mtimes()
            changed = [
                os.path.relpath(path, PROJECT_ROOT)
                for path, mtime in new_mtimes.items()
                if mtimes.get(path) != mtime
            ]

            if changed:
                labels = ', '.join(changed[:4])
                if len(changed) > 4:
                    labels += f' (+{len(changed) - 4} more)'
                if _queue_is_active(proc):
                    # Restarting now would kill or orphan the training
                    # workers mid-run. Hold the restart until the queue
                    # finishes; keep the old mtimes so the pending change
                    # keeps being detected.
                    if not deferred_announced:
                        print(f'\n[Launcher] Changed: {labels}')
                        print(
                            '[Launcher] Queue is running - restart deferred '
                            'until it finishes.'
                        )
                        deferred_announced = True
                    continue
                print(f'\n[Launcher] Changed: {labels}')
                print('[Launcher] Restarting GUI...\n')
                _stop_gui_tree(proc)
                time.sleep(0.3)
                mtimes = _collect_mtimes()
                proc = _launch_subprocess()
                deferred_announced = False

    except KeyboardInterrupt:
        print('\n[Launcher] Interrupted - stopping GUI and workers...')
        _stop_gui_tree(proc)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if '--gui' in sys.argv:
        _run_gui()
    else:
        _run_watcher()
