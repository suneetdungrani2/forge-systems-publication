"""Checks for the selected independent Tier B replay (study_artifact/tierb/replay.py).

Needs the pinned Tier B environment (MuJoCo 3.15.0, NumPy 2.5.4); skipped where MuJoCo is absent. The physics
tests also need the robot meshes, which stay outside Git: point TIERB_MESHES at them (default: the worktree's
.tierb-inputs/meshes). Run from study_artifact/:  tierb/.venv/bin/python -m unittest tests.test_tierb_replay
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

TIERB = Path(__file__).resolve().parents[1] / "tierb"
PACKAGE = TIERB / "package"
MANIFEST = TIERB / "inputs_manifest.json"
MESHES = Path(os.environ.get("TIERB_MESHES", Path(__file__).resolve().parents[2] / ".tierb-inputs" / "meshes"))

HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None  # the shared analysis environment has no MuJoCo
NO_MUJOCO = "MuJoCo is not installed; use the pinned study_artifact/tierb environment"
replay = None
if HAVE_MUJOCO:
    _spec = importlib.util.spec_from_file_location("tierb_replay", TIERB / "replay.py")
    replay = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(replay)


def _package_only_manifest(path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    manifest["files"] = [entry for entry in manifest["files"] if entry["root"] == "package"]
    path.write_text(json.dumps(manifest))


@unittest.skipUnless(HAVE_MUJOCO, NO_MUJOCO)
class ManifestTests(unittest.TestCase):
    def test_replay_imports_only_the_declared_modules(self):
        lines = (TIERB / "replay.py").read_text().splitlines()
        imports = sorted(line.split()[1] for line in lines if line.startswith(("import ", "from ")))
        self.assertEqual(imports, ["argparse", "hashlib", "json", "mujoco", "numpy"])

    def test_committed_package_matches_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            _package_only_manifest(manifest)
            result = replay.verify_manifest(manifest, str(PACKAGE), str(Path(tmp) / "no-meshes"))
        self.assertEqual(result["files_verified"], len(replay.PACKAGE_FILES))

    def test_tampered_package_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "package"
            shutil.copytree(PACKAGE, copy)
            manifest = Path(tmp) / "manifest.json"
            _package_only_manifest(manifest)
            replay.verify_manifest(manifest, str(copy), str(Path(tmp) / "no-meshes"))
            draw = json.loads((copy / "disturbance.json").read_text())
            draw["draw"]["surface_friction"] += 1e-9
            (copy / "disturbance.json").write_text(json.dumps(draw, indent=2, sort_keys=True) + "\n")
            with self.assertRaises(replay.ManifestError):
                replay.verify_manifest(manifest, str(copy), str(Path(tmp) / "no-meshes"))

    def test_missing_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "package"
            shutil.copytree(PACKAGE, copy)
            (copy / "schedule.npz").unlink()
            manifest = Path(tmp) / "manifest.json"
            _package_only_manifest(manifest)
            with self.assertRaises(replay.ManifestError):
                replay.verify_manifest(manifest, str(copy), str(Path(tmp) / "no-meshes"))

    def test_manifest_cannot_drop_a_required_package_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(MANIFEST.read_text())
            manifest["files"] = [e for e in manifest["files"] if e["path"] != "judge.json"]
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(replay.ManifestError):
                replay.verify_manifest(path, str(PACKAGE), str(MESHES))


@unittest.skipUnless(HAVE_MUJOCO, NO_MUJOCO)
@unittest.skipUnless(MESHES.is_dir(), "robot meshes are not available (set TIERB_MESHES)")
class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replay.verify_manifest(MANIFEST, str(PACKAGE), str(MESHES))
        cls.pkg = replay.load_package(str(PACKAGE), str(MESHES))

    def test_manifest_lists_every_mesh_the_scenes_name(self):
        listed = sorted(e["path"] for e in json.loads(MANIFEST.read_text())["files"] if e["root"] == "meshes")
        self.assertEqual(listed, replay.mesh_names(str(PACKAGE)))

    def test_tampered_mesh_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            meshes = Path(tmp) / "meshes"
            shutil.copytree(MESHES, meshes)
            target = meshes / replay.mesh_names(str(PACKAGE))[0]
            data = bytearray(target.read_bytes())
            data[-1] ^= 0x01
            target.write_bytes(bytes(data))
            with self.assertRaises(replay.ManifestError):
                replay.verify_manifest(MANIFEST, str(PACKAGE), str(meshes))

    def test_failed_original_trial_is_reproduced(self):
        model, run, verdict, deviation = replay.replay_condition(self.pkg, str(PACKAGE), "original")
        self.assertEqual(deviation, 0.0)
        self.assertEqual((verdict["status"], verdict["legacy_mode"]), ("FAIL", "seated_out_of_tolerance"))
        comparison = replay.compare(model, run, str(PACKAGE), "original")
        self.assertTrue(comparison["native_grid_times_identical"] and comparison["state_times_identical"])
        self.assertLessEqual(comparison["housing_position_dev_mm"]["max"], replay.GATE["housing_position_max_abs_dev_mm"])

    def test_wrong_disturbance_changes_the_result(self):
        wrong = dict(self.pkg["disturbance"], draw=dict(self.pkg["disturbance"]["draw"]))
        wrong["draw"]["fixture"] = [wrong["draw"]["fixture"][0] + 0.3, wrong["draw"]["fixture"][1],
                                    wrong["draw"]["fixture"][2]]
        model, run, verdict, deviation = replay.replay_condition(self.pkg, str(PACKAGE), "historical_v4_lead_in", wrong)
        self.assertGreater(deviation, 1e-4)  # the edit no longer matches the recorded disturbance
        comparison = replay.compare(model, run, str(PACKAGE), "historical_v4_lead_in")
        self.assertFalse(comparison["housing_pose_bitwise_identical"])
        self.assertGreater(comparison["housing_position_dev_mm"]["max"], 0.05)

    def test_disturbance_rules_reject_unknown_entities(self):
        wrong = json.loads(json.dumps(self.pkg["disturbance"]))
        wrong["rules"][2]["select"]["body_names"] = ["no_such_body"]
        model = replay.build_model(self.pkg, "original")
        with self.assertRaises(ValueError):
            replay.apply_disturbance(model, wrong)


if __name__ == "__main__":
    unittest.main()
