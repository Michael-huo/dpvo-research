"""Fresh current replacement: build privately, validate, then publish with rollback."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, ExitStack
import fcntl
from pathlib import Path
import shutil
import tempfile
import uuid


MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx"}


def without_worker_log_paths(value):
    """Copy result metadata without temporary worker-log references.

    Apply when assembling persisted results, before artifact hashing. Runtime
    payloads keep their diagnostic paths; timings and other provenance survive.
    """
    if isinstance(value, Mapping):
        return {key: without_worker_log_paths(item) for key, item in value.items()
                if key != "worker_log"}
    if isinstance(value, list):
        return [without_worker_log_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(without_worker_log_paths(item) for item in value)
    return value


def validate_lightweight_results(root):
    for path in Path(root).rglob("*"):
        if path.is_symlink() or path.suffix.lower() in MODEL_SUFFIXES:
            raise RuntimeError(f"model/symlink must not be published in results: {path}")


@contextmanager
def staged_directory(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as name:
        root = Path(name) / "payload"
        root.mkdir()
        yield root


def publish_checkpoint_tree(staged, destination):
    """Atomically replace a complete model directory with rollback on failure."""
    staged, destination = Path(staged), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not staged.is_dir() or staged.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError("checkpoint publication requires same-filesystem staging")
    if staged.resolve() == destination.resolve() or staged.resolve().is_relative_to(destination.resolve()):
        raise ValueError("checkpoint staging must be separate from destination")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    import os
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if destination.exists():
            destination.rename(backup)
        try:
            staged.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        os.close(fd)


def publish_current_canonical(staged, destination, *, checkpoint_staged=None,
                              checkpoint_destination=None):
    """Rename complete trees; reverse every successful rename on publish failure.

    Readers may briefly see missing paths during the two-directory transaction;
    each tree rename is atomic, and failed publication restores both old trees.
    Advisory parent-directory locks serialize concurrent publishers without lock
    files, stale lock recovery or generation directories.
    """
    pairs = [(Path(staged), Path(destination))]
    if (checkpoint_staged is None) != (checkpoint_destination is None):
        raise ValueError("checkpoint staging and destination must be provided together")
    if checkpoint_staged is not None:
        pairs.append((Path(checkpoint_staged), Path(checkpoint_destination)))
    validate_lightweight_results(staged)
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_dir() or source.stat().st_dev != target.parent.stat().st_dev:
            raise RuntimeError("publish requires staging on the destination filesystem")
        if source.resolve() == target.resolve() or source.resolve().is_relative_to(target.resolve()):
            raise ValueError("staging must be separate from canonical artifacts")
    token = uuid.uuid4().hex
    backups, published = [], []
    with ExitStack() as stack:
        import os
        for parent in sorted({p.parent.resolve() for _, p in pairs}):
            fd = os.open(parent, os.O_RDONLY)
            stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            for source, target in pairs:
                backup = target.with_name(f".{target.name}.backup-{token}")
                if target.exists():
                    target.rename(backup)
                    backups.append((backup, target))
                source.rename(target)
                published.append((target, source))
        except BaseException:
            for target, source in reversed(published):
                target.rename(source)
            for backup, target in reversed(backups):
                backup.rename(target)
            raise
        for backup, _ in backups:
            shutil.rmtree(backup)
