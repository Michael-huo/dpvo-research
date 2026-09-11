from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import io
import multiprocessing
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from . import run_h0, run_h1, run_h2
from .protocol import PACKAGE_DIR, REPO_ROOT, FrameIdentity, repo_path


WORKER_MODULES = (
    "research.src.jepa_worker", "research.src.pipeline_worker",
    "research.src.parallel_runtime",
)


def _spawn_import_workers(connection, identity):
    with connection:
        connection.send((
            [importlib.import_module(name).__name__ for name in WORKER_MODULES],
            identity,
        ))


class ArchitectureContractTest(unittest.TestCase):
    def test_functional_package_and_canonical_paths(self) -> None:
        self.assertEqual(PACKAGE_DIR, REPO_ROOT / "research/src")
        old_name = "phase1_feasibility"
        self.assertFalse((PACKAGE_DIR / old_name).exists())
        self.assertIsNone(importlib.util.find_spec(f"research.src.{old_name}"))
        for runner, module in (
            (run_h0, "h0_state"), (run_h1, "h1_interface"),
            (run_h2, "h2_prediction"),
        ):
            config, path = runner.load_config()
            self.assertEqual(
                path, REPO_ROOT / "research/configs" / f"{old_name}_{module[:2]}.yaml",
            )
            self.assertEqual(
                repo_path(config["paths"]["output_root"]),
                REPO_ROOT / "research/results/phase1-feasibility" / module,
            )

    def test_public_cli_dispatch_single_three_and_default_sequences(self) -> None:
        sequences = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]
        for runner in (run_h0, run_h1, run_h2):
            for requested in (None, sequences[:1], sequences):
                with self.subTest(runner=runner.__name__, sequences=requested):
                    args = [] if requested is None else ["--sequences", *requested]
                    with patch.object(sys, "argv", [runner.__name__, *args]), \
                         patch.object(runner, "run", return_value={}) as dispatch, \
                         contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(runner.main(), 0)
                    self.assertEqual(list(dispatch.call_args.args[0]), requested or sequences[:1])
                    dispatch.assert_called_once()

    def test_public_cli_rejects_invalid_sequence_before_dispatch(self) -> None:
        for runner in (run_h0, run_h1, run_h2):
            with self.subTest(runner=runner.__name__):
                with patch.object(sys, "argv", [runner.__name__, "--sequences", "invalid"]), \
                     patch.object(runner, "run") as dispatch, \
                     contextlib.redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                dispatch.assert_not_called()

    def test_public_cli_help_in_fresh_processes(self) -> None:
        for module in ("research.src.run_h0", "research.src.run_h1", "research.src.run_h2"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"], cwd=REPO_ROOT,
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True, text=True, timeout=30, check=True,
                )
                self.assertIn("--sequences", result.stdout)

    def test_workers_import_in_fresh_runtime_environments(self) -> None:
        config, _ = run_h2.load_config()
        interpreters = {sys.executable, config["runtime"]["jepa_python"]}
        code = (
            "import importlib, sys; "
            "modules = [importlib.import_module(n) for n in sys.argv[1:]]; "
            "assert all(m.__package__ == 'research.src' for m in modules)"
        )
        for python in sorted(interpreters):
            with self.subTest(python=python):
                if not Path(python).is_file():
                    self.skipTest(f"optional V-JEPA environment missing: {python}")
                subprocess.run(
                    [python, "-c", code, *WORKER_MODULES], cwd=REPO_ROOT,
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True, text=True, timeout=30, check=True,
                )
                for module in WORKER_MODULES[:2]:
                    subprocess.run(
                        [python, "-m", module, "--help"], cwd=REPO_ROOT,
                        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                        capture_output=True, text=True, timeout=30, check=True,
                    )

    def test_spawn_imports_workers_and_round_trips_frame_identity(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        identity = FrameIdentity("euroc", "machine_hall", "sequence", "MH_01_easy", 0, 0, 1)
        process = context.Process(target=_spawn_import_workers, args=(child, identity))
        try:
            process.start()
            child.close()
            self.assertTrue(parent.poll(30), "spawned worker did not respond")
            modules, received = parent.recv()
            self.assertEqual(modules, list(WORKER_MODULES))
            self.assertEqual(received, identity)
            self.assertEqual(type(received).__module__, "research.src.protocol")
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        finally:
            parent.close()
            child.close()
            if process.is_alive():
                process.terminate()
                process.join(5)

    def test_only_three_public_runners(self) -> None:
        root = Path(run_h0.__file__).parent
        self.assertEqual(
            {path.name for path in root.glob("run_*.py")},
            {"run_h0.py", "run_h1.py", "run_h2.py"},
        )

    def test_new_package_has_no_legacy_import_or_result_dependency(self) -> None:
        root = Path(run_h0.__file__).parent
        legacy_package = "phase1_" + "dpvo_feasibility"
        legacy_results = "phase1-" + "dpvo-feasibility"
        legacy_runner = re.compile(r"run_" + r"exp[1-6]")
        legacy_configs = (
            "phase1_" + "dpvo_feasibility.yaml",
            "phase1_" + "exp2.yaml",
            "phase1_" + "exp3.yaml",
            "exp6_" + "decomposition.yaml",
            "exp6_" + "h1.yaml",
            "exp6_" + "h2.yaml",
        )
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(legacy_package, source, path.name)
            self.assertNotIn(legacy_results, source, path.name)
            self.assertIsNone(legacy_runner.search(source), path.name)
            for config in legacy_configs:
                self.assertNotIn(config, source, path.name)
        h0_source = Path(run_h0.__file__).read_text(encoding="utf-8")
        self.assertNotIn("h1_training", h0_source)
        self.assertNotIn("h2_training", h0_source)
        h2_source = Path(run_h2.__file__).read_text(encoding="utf-8")
        imports = [
            node.module for node in ast.walk(ast.parse(h2_source))
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertFalse(any(value and value.endswith("run_h1") for value in imports))
        self.assertNotIn("h1_interface/INDEX.json", h2_source)

    def test_h2_bridge_path_is_the_only_cross_module_artifact(self) -> None:
        config, _ = run_h2.load_config()
        self.assertEqual(
            config["paths"]["h1_bridge"],
            "research/results/phase1-feasibility/h1_interface/bridge.pt",
        )
        self.assertNotIn("h0", str(config["paths"]))

    def test_upstream_dpvo_trees_match_frozen_baseline(self) -> None:
        values = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD:dpvo", "HEAD:config"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertEqual(values, [
            "4d5118e7ef6588a176827a4e306253d639fe0579",
            "24f73cbfb89d28415175beedf681a256644f22b1",
        ])


if __name__ == "__main__":
    unittest.main()
