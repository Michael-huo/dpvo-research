"""The sole public command for Experiment 1 reproduction."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .behavior import build_comparison
from .runtime import SEQUENCES, repo_root, resolve_config, run_sanity_gate, run_sequence


def controlled_output_root(root: Path) -> Path:
    parent = (root / "research/results/phase1-dpvo-feasibility").resolve()
    target = (parent / "exp1").resolve()
    if target.parent != parent or target.name != "exp1" or target.is_symlink():
        raise RuntimeError(f"refusing unsafe Experiment 1 output target: {target}")
    return target


def main() -> int:
    root = repo_root()
    output = controlled_output_root(root)
    # This cleanup is intentionally constrained to the fixed output directory.
    if output.exists():
        shutil.rmtree(output)
    staging = Path(tempfile.mkdtemp(prefix=".exp1-staging-", dir=output.parent))
    try:
        sanity = run_sanity_gate(root, resolve_config(sequence="MH_01_easy"))
        sequence_runs: dict[str, Path] = {}
        reference: Path | None = None
        for sequence in SEQUENCES:
            sequence_runs[sequence] = run_sequence(root, sequence, staging / sequence, sanity, reference)
            if sequence == "MH_01_easy":
                reference = sequence_runs[sequence] / "report/metrics.json"
        build_comparison(sequence_runs, staging / "comparison")
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Experiment 1 complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
