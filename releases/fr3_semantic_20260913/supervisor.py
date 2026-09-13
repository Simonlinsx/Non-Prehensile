"""Bounded process-group lifecycle, copied from scripts/run_bounded_fr3_experiment.py."""
import os
import signal
import subprocess
import time
from pathlib import Path

def live_group(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text(); fields = stat[stat.rfind(')')+2:].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(path.name))
        except (OSError, ValueError):
            continue
    return members

def bounded_process(command, *, cwd, env, log, timeout_s, grace_s=3):
    if timeout_s <= 0 or grace_s <= 0:
        raise ValueError('Timeouts must be positive')
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    reason = 'exited'
    interrupted = False
    previous = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        try:
            child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            reason = 'wall_timeout'
        except KeyboardInterrupt:
            reason = 'interrupted'; interrupted = True
    finally:
        # A parent shell can exit while a simulator still owns CUDA resources.
        # Cleanup targets the session we created, even after normal parent exit.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not live_group(child.pid):
                    break
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                end = time.monotonic()+grace_s
                while live_group(child.pid) and time.monotonic() < end:
                    child.poll(); time.sleep(.05)
            child.wait(timeout=grace_s)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return dict(reason=reason, interrupted=interrupted, returncode=child.returncode,
                process_wall_s=time.monotonic()-started, pgid=child.pid,
                remaining_live_pids=live_group(child.pid))
