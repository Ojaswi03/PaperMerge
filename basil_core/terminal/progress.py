"""Terminal activity for blocking operational checks; never training progress."""
from __future__ import annotations

import math
import os
import statistics
import sys
import threading
import time

from basil_core.terminal.state import safe_text


def duration(seconds):
    seconds=max(0,int(seconds))
    hours,remainder=divmod(seconds,3600)
    minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class DiagnosticProgress:
    """Refresh stderr during a blocking check, including redirected SSH output.

    Estimates use completed successful invocations of the same stage plan.
    The caller supplies a session-local timing dictionary; nothing is persisted
    to research results or used to approve execution.
    """

    def __init__(self,scope,stages,*,estimates=None,stream=None,interval=1.,clock=time.monotonic):
        if not math.isfinite(interval) or interval<=0:
            raise ValueError('Progress interval must be finite and positive.')
        self.scope=safe_text(scope);self.stages=tuple(stages)
        if not self.stages or len(set(self.stages))!=len(self.stages):
            raise ValueError('Distinct diagnostic stages are required.')
        self.estimates=estimates if estimates is not None else {}
        self.stream=stream if stream is not None else sys.stderr
        self.interval=interval;self.clock=clock
        self.lock=threading.RLock();self.stop=threading.Event()
        self.thread=None;self.current=None;self.timings={};self.finished=False
        self.interactive=bool(getattr(self.stream,'isatty',lambda:False)()) and os.environ.get('TERM') not in (None,'dumb')
        self.transient=False;self.output_available=True

    def __enter__(self):
        self.started=self.clock()
        self.begin(self.stages[0])
        self.thread=threading.Thread(target=self._heartbeat,name='diagnostic-progress',daemon=True)
        self.thread.start()
        return self

    def _write(self,message,*,refresh=False):
        if not self.output_available:return
        try:
            if self.interactive:
                if self.transient:self.stream.write('\r\033[2K')
                self.stream.write(message+('' if refresh else '\n'))
                self.transient=refresh
            else:
                self.stream.write(message+'\n')
            self.stream.flush()
        except (OSError,ValueError):
            # A closed SSH display cannot change a verification result.
            self.output_available=False

    def _complete(self,now):
        if self.current is None:return
        elapsed=max(0.,now-self.stage_started)
        self.timings[self.current]=elapsed
        self._write(f'{self.scope} | Finished: {self.current} | took {duration(elapsed)}')

    def begin(self,label):
        with self.lock:
            if self.finished:raise RuntimeError('Diagnostic progress has already finished.')
            index=self.stages.index(label)
            expected=0 if self.current is None else self.stages.index(self.current)+1
            if index!=expected:raise ValueError('Diagnostic stages must run in their declared order.')
            now=self.clock();self._complete(now)
            self.current=label;self.stage_started=now
            self._write(self._line(now))

    def run(self,label,action):
        self.begin(label)
        return action()

    def _eta(self,now):
        remaining=self.stages[self.stages.index(self.current):]
        if any(not self.estimates.get(label) for label in remaining):
            return 'ETA unavailable (no completed matching GPU check yet)'
        estimates={label:statistics.median(self.estimates[label]) for label in remaining}
        elapsed=max(0.,now-self.stage_started)
        if elapsed>=estimates[self.current]:
            return 'ETA recalculating (current stage exceeds previous timing)'
        return 'ETA ~'+duration(sum(estimates.values())-elapsed)+' for this GPU (previous GPU timing)'

    def _line(self,now):
        index=self.stages.index(self.current)+1
        return (f'{self.scope} | Stage {index}/{len(self.stages)}: {self.current} | running'
            f' | elapsed {duration(now-self.started)} | stage {duration(now-self.stage_started)} | {self._eta(now)}')

    def refresh(self):
        with self.lock:
            if not self.finished:self._write(self._line(self.clock()),refresh=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):self.refresh()

    def finish(self,status):
        with self.lock:
            if self.finished:return
            now=self.clock()
            if status=='passed':
                if self.current!=self.stages[-1]:
                    raise ValueError('Cannot complete diagnostics before the final stage.')
                self._complete(now)
                for label,elapsed in self.timings.items():
                    samples=self.estimates.setdefault(label,[])
                    samples.append(elapsed);del samples[:-8]
            self._write(f'{self.scope} | {safe_text(status).upper()} | elapsed {duration(now-self.started)}')
            self.finished=True
        self.stop.set()

    def __exit__(self,kind,error,traceback):
        try:
            if not self.finished:self.finish('interrupted' if kind is KeyboardInterrupt else 'failed')
        finally:
            self.stop.set()
            if self.thread is not None:self.thread.join()
