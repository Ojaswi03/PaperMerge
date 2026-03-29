
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
    os.path.join(PROJECT_ROOT, 'runGui.py'),
]


# ── GUI subprocess mode ────────────────────────────────────────────────────────
def _run_gui():
    """Run the actual GUI. Invoked when --gui flag is present."""
    from scripts.common import setupGpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    setupGpu()

    from gui.experimentGui import main

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


def _launch_subprocess():
    import subprocess
    return subprocess.Popen([sys.executable, __file__, '--gui'])


def _run_watcher():
    import subprocess
    import time

    print("""
====================================================================
          BASIL + Noisy Channel Experiment GUI - Auto-Reload

   Watching for .py file changes. The GUI will restart automatically
   whenever you save an edited file.
   Use the [Reload] button inside the GUI to reload manually.
   Press Ctrl+C here to exit completely.
====================================================================
    """)

    mtimes = _collect_mtimes()
    proc = _launch_subprocess()

    try:
        while True:
            time.sleep(0.8)

            retcode = proc.poll()

            # Manual reload requested by the GUI (Reload button)
            if retcode == RELOAD_EXIT_CODE:
                print('\n[Launcher] Reload requested - restarting GUI...\n')
                mtimes = _collect_mtimes()
                proc = _launch_subprocess()
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
                print(f'\n[Launcher] Changed: {labels}')
                print('[Launcher] Restarting GUI...\n')
                proc.terminate()
                try:
                    proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(0.3)
                mtimes = new_mtimes
                proc = _launch_subprocess()

    except KeyboardInterrupt:
        print('\n[Launcher] Interrupted - stopping GUI...')
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if '--gui' in sys.argv:
        _run_gui()
    else:
        _run_watcher()
