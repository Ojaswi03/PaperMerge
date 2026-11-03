"""Transfer existing GPU open-file descriptions without touching training workers.

No lock pathname replacement, lock removal, ptrace attachment, or worker signal.
pidfd_getfd may be denied by the host security policy; that is a hard stop.
"""
import ctypes
import errno
import fcntl
import os
from pathlib import Path
import platform

from basil_core import live_gpu_scheduler as live


def lease_path(profile):
    return Path('/tmp') / f'papermerge-{os.getuid()}-{profile.gpu_uuid}.lock'


def validate_fd(fd, path):
    actual = os.fstat(fd)
    expected = path.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Transferred GPU lease refers to a different inode.')
    # With a true duplicate this succeeds while all original holders continue.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def duplicate_fd(identity, target_fd):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise OSError(errno.ENOSYS, 'pidfd_getfd is unsupported on this platform.')
    pidfd = os.pidfd_open(identity['pid'])
    try:
        if not live.same_process(identity):
            raise ProcessLookupError('Lease holder changed before duplication.')
        libc = ctypes.CDLL(None, use_errno=True)
        # Linux x86_64 and aarch64 both assign 438 to pidfd_getfd.
        fd = libc.syscall(ctypes.c_long(438), ctypes.c_int(pidfd),
                          ctypes.c_int(target_fd), ctypes.c_uint(0))
        if fd < 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        if not live.same_process(identity):
            os.close(fd)
            raise ProcessLookupError('Lease holder changed during duplication.')
        return fd
    finally:
        os.close(pidfd)


def capture(profile, identities):
    """Return a locked handle, sharing an existing OFD if the GPU is occupied."""
    from scripts.diagnose_fedsharp_activity import lease_fds
    path = lease_path(profile)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle, dict(gpu=profile.gpu, method='free lease acquired')
    except BlockingIOError:
        handle.close()
    except BaseException:
        handle.close()
        raise
    errors = []
    for identity in identities:
        for target in lease_fds(identity, path).get('descriptors', []):
            fd = None
            try:
                fd = duplicate_fd(identity, target)
                validate_fd(fd, path)
                return os.fdopen(fd, 'a'), dict(gpu=profile.gpu,
                    method='existing lease duplicated', holderPid=identity['pid'])
            except OSError as error:
                if fd is not None:
                    os.close(fd)
                errors.append(error.errno)
            except BaseException:
                if fd is not None:
                    os.close(fd)
                raise
    if errno.EPERM in errors or errno.EACCES in errors:
        raise PermissionError('Linux denied duplication of the existing GPU lease. '
                              'No dispatcher or worker was signalled; existing runs continue.')
    raise RuntimeError('Cannot safely duplicate every GPU lease; existing runs continue.')


def inherited_handles(specs, profiles):
    """Validate the child handoff before accepting its inherited descriptors."""
    handles = {}
    try:
        for spec in specs:
            gpu, fd = map(int, spec.split(':'))
            if gpu in handles or gpu not in profiles or fd < 3:
                raise ValueError('Invalid inherited GPU lease mapping.')
            validate_fd(fd, lease_path(profiles[gpu]))
            handles[gpu] = os.fdopen(os.dup(fd), 'a')
        if set(handles) != set(profiles):
            raise ValueError('Handoff must retain every GPU lease on this host.')
        return handles
    except BaseException:
        for handle in handles.values():
            handle.close()
        raise
