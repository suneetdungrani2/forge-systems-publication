#!/usr/bin/env python3
"""Selected independent Tier B replay of one archived V6 E2 fixture-insertion trial pair.

Standalone: this file imports only mujoco, numpy, json, hashlib and argparse and contains no Forge code. It reads a
data-only package (compiled MJCF scenes, robot meshes, a per-step actuator target schedule, the pre-settle initial
state, the seeded disturbance draw with its model-edit rules, timing and the success-judge definition), verifies every
input against ``inputs_manifest.json``, re-simulates both fixture conditions with MuJoCo, judges them and compares the
trajectories with the archived recordings. One pair is replayed; this is not a cohort reproduction.

Commands (from ``study_artifact/tierb``)::

    python replay.py manifest --package package --meshes MESH_DIR --out inputs_manifest.json
    python replay.py run --package package --meshes MESH_DIR --manifest inputs_manifest.json --result TIERB_RESULT.json
"""
import argparse
import hashlib
import json

import mujoco
import numpy as np

CONDITIONS = ("original", "historical_v4_lead_in")
STATUSES = ("SELECTED_INDEPENDENT_TIER_B_REPRODUCED", "PARTIAL", "TIER_B_UNAVAILABLE")
# Declared before any replay was run. Decisions must agree; the trajectory must stay well inside the judge's
# resolution: a quarter of the tightest position limit (0.4 mm) and less than the original trial's 0.12 deg yaw
# margin over its 3.3 deg limit.
GATE = {"housing_position_max_abs_dev_mm": 0.1, "final_abs_yaw_dev_deg": 0.1}
WARNINGS = ("mjWARN_BADQACC", "mjWARN_BADQVEL", "mjWARN_BADQPOS")
CONTROL_FRICTION_DELTA = 0.05
CONTROL_QPOS_NUDGE_M = 1e-12  # far below any physical tolerance: probes how a last-bits platform difference would grow


class ManifestError(ValueError):
    """An input is missing or differs from the manifest."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    with open(path, "rb") as stream:
        return stream.read()


def _json(path):
    return json.loads(_read(path))


def _write_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


# ----------------------------------------------------------------------------------------------- inputs and manifest
PACKAGE_FILES = ("pair.json", "timing.json", "disturbance.json", "judge.json", "expected.json", "schedule.npz",
                 "initial_state.npz", "model/scene_original.xml", "model/scene_historical_v4_lead_in.xml",
                 "reference/original_pose.npz", "reference/original_qpos30.npz",
                 "reference/historical_v4_lead_in_pose.npz", "reference/historical_v4_lead_in_qpos30.npz")


def mesh_names(package):
    """Mesh files named by the scene MJCF (mesh assets with a file attribute)."""
    names = set()
    for condition in CONDITIONS:
        spec = mujoco.MjSpec.from_string(_read(f"{package}/model/scene_{condition}.xml").decode())
        names.update(mesh.file for mesh in spec.meshes if mesh.file)
    return sorted(names)


def write_manifest(package, meshes, out):
    entries = [{"root": "package", "path": name, "sha256": _sha(_read(f"{package}/{name}")),
                "bytes": len(_read(f"{package}/{name}"))} for name in PACKAGE_FILES]
    entries += [{"root": "meshes", "path": name, "sha256": _sha(_read(f"{meshes}/{name}")),
                 "bytes": len(_read(f"{meshes}/{name}"))} for name in mesh_names(package)]
    pair = _json(f"{package}/pair.json")
    _write_json(out, {"schema": "forge.v6_1.tierb-inputs-manifest/1", "hash": "sha256",
                      "roots": {"package": "study_artifact/tierb/package (in Git)",
                                "meshes": "robot STL meshes of the compiled scene (outside Git; byte-identical to the archived compiled tree)"},
                      "archive_identity": pair["archive_identity"], "files": entries,
                      "total_bytes": {root: sum(e["bytes"] for e in entries if e["root"] == root) for root in ("package", "meshes")}})


def verify_manifest(manifest_path, package, meshes):
    manifest = _json(manifest_path)
    roots = {"package": package, "meshes": meshes}
    listed = {(e["root"], e["path"]) for e in manifest["files"]}
    required = {("package", name) for name in PACKAGE_FILES}
    if not required <= listed:
        raise ManifestError(f"manifest omits package files: {sorted(required - listed)}")
    for entry in manifest["files"]:
        path = f"{roots[entry['root']]}/{entry['path']}"
        try:
            data = _read(path)
        except OSError as exc:
            raise ManifestError(f"missing input {entry['root']}:{entry['path']}") from exc
        if _sha(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise ManifestError(f"input differs from manifest: {entry['root']}:{entry['path']}")
    return {"files_verified": len(manifest["files"]), "manifest_sha256": _sha(_read(manifest_path))}


def load_package(package, meshes):
    with np.load(f"{package}/schedule.npz") as z:
        schedule = {"ctrl": z["ctrl"].copy(), "actuator_names": [str(x) for x in z["actuator_names"]]}
    with np.load(f"{package}/initial_state.npz") as z:
        initial = {key: z[key].copy() for key in z.files}
    return {"pair": _json(f"{package}/pair.json"), "timing": _json(f"{package}/timing.json"),
            "disturbance": _json(f"{package}/disturbance.json"), "judge": _json(f"{package}/judge.json"),
            "expected": _json(f"{package}/expected.json"), "schedule": schedule, "initial": initial,
            "scenes": {c: _read(f"{package}/model/scene_{c}.xml") for c in CONDITIONS},
            "meshes": {name: _read(f"{meshes}/{name}") for name in mesh_names(package)}}


# ----------------------------------------------------------------------------------------------- model and disturbance
def build_model(pkg, condition):
    assets = {f"meshes/{name}": data for name, data in pkg["meshes"].items()}
    return mujoco.MjModel.from_xml_string(pkg["scenes"][condition].decode(), assets=assets)


def _ids(model, kind, names):
    out = []
    for name in names:
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError(f"model has no {kind} named {name}")
        out.append(index)
    return out


def apply_disturbance(model, disturbance):
    """Apply the draw through the package's declared rules; return the edited values by name."""
    draw = disturbance["draw"]
    base = {"geom_friction0": model.geom_friction[:, 0].copy(), "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(), "body_pos": model.body_pos.copy()}
    edited = {"geom_friction0": {}, "body_mass": {}, "body_inertia": {}, "body_pos": {}, "body_quat": {}}
    for rule in disturbance["rules"]:
        field, op, param = rule["field"], rule["op"], rule["param"]
        if "geom_names" in rule["select"]:
            names = rule["select"]["geom_names"]
            ids = _ids(model, mujoco.mjtObj.mjOBJ_GEOM, names)
        else:
            names = rule["select"]["body_names"]
            ids = _ids(model, mujoco.mjtObj.mjOBJ_BODY, names)
        for name, index in zip(names, ids):
            if field == "geom_friction0":
                value = base[field][index] * draw[param] if op == "scale_base" else float(draw[param])
                if op not in ("scale_base", "set"):
                    raise ValueError(f"unsupported op {op}")
                model.geom_friction[index, 0] = value
                edited[field][name] = float(model.geom_friction[index, 0])
            elif field in ("body_mass", "body_inertia") and op == "scale_base":
                getattr(model, field)[index] = base[field][index] * draw[param]
                value = getattr(model, field)[index]
                edited[field][name] = value.tolist() if np.ndim(value) else float(value)
            elif field == "body_pos" and op == "offset_xy_mm":
                dx, dy = draw[param][0], draw[param][1]
                model.body_pos[index] = base[field][index] + np.array([dx, dy, 0.0]) / 1000.0
                edited[field][name] = model.body_pos[index].tolist()
            elif field == "body_quat" and op == "set_yaw_deg":
                half = float(np.radians(rule["nominal_yaw_deg"] + draw[param][2])) / 2
                model.body_quat[index] = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]
                edited[field][name] = model.body_quat[index].tolist()
            else:
                raise ValueError(f"unsupported rule {field}/{op}")
    return edited


def edit_deviation(edited, expected):
    """Largest absolute difference between rule-applied values and the exporter's recorded edited values."""
    worst = 0.0
    for field, values in expected.items():
        if set(values) != set(edited.get(field, {})):
            raise ValueError(f"disturbance edits {field} touch different entities than recorded")
        for name, value in values.items():
            worst = max(worst, float(np.max(np.abs(np.asarray(edited[field][name]) - np.asarray(value)))))
    return worst


# ----------------------------------------------------------------------------------------------- simulation
def simulate(model, pkg, part_body="obj_housing", other_parts=("obj_insert", "obj_lid")):
    timing, ctrl, initial = pkg["timing"], pkg["schedule"]["ctrl"], pkg["initial"]
    if model.nu != ctrl.shape[1] or [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                                     for i in range(model.nu)] != pkg["schedule"]["actuator_names"]:
        raise ValueError("schedule actuators do not match the model")
    if model.nmocap or model.na or abs(float(model.opt.timestep) - timing["timestep_s"]) > 0:
        raise ValueError("model timing/state layout differs from the package")
    n_settle, total = timing["settle_steps"], timing["total_steps"]
    if ctrl.shape[0] != total:
        raise ValueError("schedule length differs from the declared step count")
    data = mujoco.MjData(model)
    data.qpos[:] = initial["qpos"]
    data.qvel[:] = initial["qvel"]
    data.ctrl[:] = initial["ctrl"]
    data.qacc_warmstart[:] = initial["qacc_warmstart"]
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part_body)
    n = total - n_settle
    pos = np.empty((n, 3))
    quat = np.empty((n, 4))
    yaw = np.empty(n)
    tilt = np.empty(n)
    times = np.empty(n)
    state_t, state_q = [], []
    for k in range(total):
        t = (k - n_settle) * dt
        data.ctrl[:] = ctrl[k]
        mujoco.mj_step(model, data)
        if k >= n_settle:
            i = k - n_settle
            rot = data.xmat[body]
            times[i] = t
            pos[i] = data.xpos[body] * 1000.0
            quat[i] = data.xquat[body]
            yaw[i] = np.degrees(np.arctan2(rot[3], rot[0]))
            tilt[i] = np.degrees(np.arccos(max(-1.0, min(1.0, float(rot[8])))))
            if t >= len(state_t) / 30.0 - 1e-9:
                state_t.append(float(t))
                state_q.append(data.qpos.copy())
    others = {name: (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)] * 1000.0).tolist()
              for name in other_parts}
    warnings = {name: int(data.warning[int(getattr(mujoco.mjtWarning, name))].number) for name in WARNINGS}
    return {"task_time_s": times, "pos_mm": pos, "quat_wxyz": quat, "yaw_deg": yaw, "tilt_deg": tilt,
            "state_t": np.asarray(state_t), "state_qpos": np.asarray(state_q), "other_final_pos_mm": others,
            "warnings": warnings, "finite": bool(np.all(np.isfinite(data.qpos)))}


# ----------------------------------------------------------------------------------------------- success judge
def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def judge(run, judge_spec, condition, task_s, dt):
    target = judge_spec["conditions"][condition]["seat_target_mm_deg"]
    legacy = judge_spec["terminal_legacy_judge"]
    tol = legacy["tolerances"]
    x, y, z = (float(v) for v in run["pos_mm"][-1])
    tilt = float(run["tilt_deg"][-1])
    dz = z - target["z_mm"]
    dxy = float(np.hypot(x - target["x_mm"], y - target["y_mm"]))
    dyaw = _wrap(float(run["yaw_deg"][-1]) - target["yaw_deg"])
    starts = judge_spec["conditions"][condition]["collateral_start_pos_mm"]
    collateral = max(float(np.linalg.norm(np.asarray(run["other_final_pos_mm"][name]) - np.asarray(start)))
                     for name, start in starts.items())
    mode = None
    if not run["finite"] or any(run["warnings"].values()):
        mode = "unstable_simulation"
    elif dxy > legacy["grasp_slip"]["xy_mm_gt"] or z < target["z_mm"] - legacy["grasp_slip"]["z_below_seat_mm_gt"]:
        mode = "grasp_slip"
    elif abs(dz) > tol["abs_z_mm"] + legacy["abs_z_epsilon"] or tilt > tol["jam_tilt_deg"]:
        mode = "ear_out_of_slot" if abs(dyaw) > tol["abs_yaw_deg"] else "jam_on_rim"
    elif dxy > tol["xy_mm"] or abs(dyaw) > tol["abs_yaw_deg"] or tilt > tol["seated_tilt_deg"]:
        mode = "seated_out_of_tolerance"
    elif collateral > legacy["collateral_mm_gt"]:
        mode = "collateral_disturbance"
    hold_spec = judge_spec["finite_hold"]
    start_s, end_s = (task_s + offset for offset in hold_spec["window_s_after_task_end"])
    grid = np.arange(len(run["task_time_s"])) * dt
    if len(grid) != int(round((task_s + 0.5) / dt)) or np.any(np.abs(run["task_time_s"] - grid) > max(1e-9, dt * 1e-6)):
        return {"status": "UNKNOWN", "reason": "pose grid incomplete"}
    window = (grid >= start_s) & (grid <= end_s)
    hold_pos = run["pos_mm"][window]
    worst = {"abs_z_error_mm": float(np.max(np.abs(hold_pos[:, 2] - target["z_mm"]))),
             "xy_error_mm": float(np.max(np.hypot(hold_pos[:, 0] - target["x_mm"], hold_pos[:, 1] - target["y_mm"]))),
             "abs_yaw_error_deg": float(np.max(np.abs(_wrap(run["yaw_deg"][window] - target["yaw_deg"])))),
             "tilt_deg": float(np.max(np.abs(run["tilt_deg"][window])))}
    within = all(worst[key] <= limit for key, limit in hold_spec["limits"].items())
    status = "PASS" if mode is None and within else "FAIL"
    if mode is not None and (not run["finite"] or any(run["warnings"].values())):
        status = "UNKNOWN"
    return {"status": status, "legacy_success": mode is None, "legacy_mode": mode,
            "legacy_metrics": {"z_error_mm": dz, "xy_error_mm": dxy, "yaw_error_deg": dyaw, "tilt_deg": tilt,
                               "collateral_mm": collateral},
            "hold_sample_count": int(np.sum(window)), "hold_worst": worst}


# ----------------------------------------------------------------------------------------------- comparison
def _stats(values):
    values = np.asarray(values, dtype=float)
    return {"max": float(np.max(values)), "rms": float(np.sqrt(np.mean(values ** 2)))}


def _moving_bodies(model):
    return [b for b in range(1, model.nbody) if model.body_weldid[b] != 0]


def _body_positions(model, qpos_rows, bodies):
    data = mujoco.MjData(model)
    out = np.empty((len(qpos_rows), len(bodies), 3))
    for i, qpos in enumerate(qpos_rows):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        out[i] = data.xpos[bodies]
    return out


def compare(model, run, package, condition):
    with np.load(f"{package}/reference/{condition}_pose.npz") as z:
        ref = {key: z[key].copy() for key in z.files}
    with np.load(f"{package}/reference/{condition}_qpos30.npz") as z:
        ref_t, ref_q = z["task_clock_threshold_s"].copy(), z["qpos"].copy()
    if ref["pos_mm"].shape != run["pos_mm"].shape or ref_q.shape != run["state_qpos"].shape:
        return {"comparable": False, "reason": "sample counts differ",
                "archived_pose_samples": int(ref["pos_mm"].shape[0]), "replay_pose_samples": int(run["pos_mm"].shape[0]),
                "archived_states": int(ref_q.shape[0]), "replay_states": int(run["state_qpos"].shape[0])}
    pos_dev = np.linalg.norm(run["pos_mm"] - ref["pos_mm"], axis=1)
    sign = np.where(np.sum(run["quat_wxyz"] * ref["quat_wxyz"], axis=1) < 0, -1.0, 1.0)[:, None]
    chord = np.linalg.norm(run["quat_wxyz"] - sign * ref["quat_wxyz"], axis=1)
    ang_dev = np.degrees(4.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))  # exact zero for identical quaternions
    bodies = _moving_bodies(model)
    body_dev = np.linalg.norm(_body_positions(model, run["state_qpos"], bodies)
                              - _body_positions(model, ref_q, bodies), axis=2) * 1000.0
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    slide = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
    hinge_dev = np.degrees(np.abs(run["state_qpos"][:, model.jnt_qposadr[hinge]] - ref_q[:, model.jnt_qposadr[hinge]]))
    slide_dev = np.abs(run["state_qpos"][:, model.jnt_qposadr[slide]] - ref_q[:, model.jnt_qposadr[slide]]) * 1000.0
    first_dev = np.nonzero(pos_dev > 0)[0]
    return {"comparable": True,
            "native_grid_times_identical": bool(np.array_equal(run["task_time_s"], ref["task_time_s"])),
            "state_times_identical": bool(np.array_equal(run["state_t"], ref_t)),
            "pose_samples": int(pos_dev.size), "qpos_states_30hz": int(ref_q.shape[0]),
            "housing_pose_bitwise_identical": bool(np.array_equal(run["pos_mm"], ref["pos_mm"])
                                                   and np.array_equal(run["quat_wxyz"], ref["quat_wxyz"])),
            "qpos30_bitwise_identical": bool(np.array_equal(run["state_qpos"], ref_q)),
            "housing_position_dev_mm": _stats(pos_dev), "housing_orientation_dev_deg": _stats(ang_dev),
            "housing_yaw_dev_deg": _stats(np.abs(_wrap(run["yaw_deg"] - ref["yaw_deg"]))),
            "housing_tilt_dev_deg": _stats(np.abs(run["tilt_deg"] - ref["tilt_deg"])),
            "first_nonzero_housing_dev_task_time_s": float(run["task_time_s"][first_dev[0]]) if first_dev.size else None,
            "moving_bodies_compared_30hz": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies],
            "moving_body_position_dev_mm_30hz": {"max": float(np.max(body_dev)), "rms": float(np.sqrt(np.mean(body_dev ** 2))),
                                                 "max_by_body": {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): float(np.max(body_dev[:, i]))
                                                                 for i, b in enumerate(bodies)}},
            "hinge_joint_dev_deg_30hz": _stats(hinge_dev), "finger_slide_dev_mm_30hz": _stats(slide_dev),
            "final_housing_pose_delta": {"pos_mm": (run["pos_mm"][-1] - ref["pos_mm"][-1]).tolist(),
                                         "pos_norm_mm": float(pos_dev[-1]),
                                         "yaw_deg": float(_wrap(run["yaw_deg"][-1] - ref["yaw_deg"][-1])),
                                         "tilt_deg": float(run["tilt_deg"][-1] - ref["tilt_deg"][-1]),
                                         "orientation_deg": float(ang_dev[-1])}}


def decision_agreement(verdict, expected):
    archived_worst = expected["hold_acceptance"].get("worst", {})
    return {"composite_status": {"replay": verdict["status"], "archived": expected["status"],
                                 "agree": verdict["status"] == expected["status"]},
            "legacy_success": {"replay": verdict["legacy_success"], "archived": expected["legacy_success"],
                               "agree": verdict["legacy_success"] == expected["legacy_success"]},
            "legacy_mode": {"replay": verdict["legacy_mode"], "archived": expected["legacy_mode"],
                            "agree": verdict["legacy_mode"] == expected["legacy_mode"]},
            "hold_worst_abs_dev": {key: abs(verdict["hold_worst"][key] - value) for key, value in archived_worst.items()},
            "legacy_metric_abs_dev_vs_archived_3dp": {key: abs(round(verdict["legacy_metrics"][key], 3) - expected["legacy_metrics"][key])
                                                      for key in verdict["legacy_metrics"]}}


# ----------------------------------------------------------------------------------------------- environment
def _first_line(path, prefix):
    try:
        with open(path) as stream:
            for line in stream:
                if line.startswith(prefix):
                    return line.split(":", 1)[-1].split("=", 1)[-1].strip().strip('"')
    except OSError:
        pass
    return None


def environment():
    system = getattr(argparse, "_sys", None)
    return {"os": _first_line("/etc/os-release", "PRETTY_NAME="), "kernel": _first_line("/proc/sys/kernel/osrelease", ""),
            "cpu_model": _first_line("/proc/cpuinfo", "model name"),
            "python": system.version.split()[0] if system else None,
            "mujoco": mujoco.__version__, "mujoco_engine": mujoco.mj_versionString(), "numpy": np.__version__}


# ----------------------------------------------------------------------------------------------- main
def replay_condition(pkg, package, condition, disturbance=None):
    model = build_model(pkg, condition)
    disturbance = disturbance or pkg["disturbance"]
    edited = apply_disturbance(model, disturbance)
    deviation = edit_deviation(edited, pkg["disturbance"]["expected_edited_values"][condition])
    run = simulate(model, pkg)
    verdict = judge(run, pkg["judge"], condition, pkg["timing"]["task_duration_s"], float(model.opt.timestep))
    return model, run, verdict, deviation


def _nudge_ensemble(pkg, package, model):
    """Replay each condition with the housing's initial position moved by CONTROL_QPOS_NUDGE_M along x or y."""
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_housing_free")
    out = {"delta_m": CONTROL_QPOS_NUDGE_M, "nudges": [], "archived_status": {}}
    for condition in CONDITIONS:
        out["archived_status"][condition] = pkg["expected"][condition]["status"]
        for axis, sign in (("x", 1.0), ("x", -1.0), ("y", 1.0)):
            initial = dict(pkg["initial"], qpos=pkg["initial"]["qpos"].copy())
            initial["qpos"][model.jnt_qposadr[joint] + "xyz".index(axis)] += sign * CONTROL_QPOS_NUDGE_M
            nudged_model, run, verdict, _ = replay_condition(dict(pkg, initial=initial), package, condition)
            cmp = compare(nudged_model, run, package, condition)
            out["nudges"].append({"condition": condition, "axis": axis, "sign": sign, "status": verdict["status"],
                                  "legacy_mode": verdict["legacy_mode"],
                                  "hold_worst_abs_yaw_error_deg": verdict["hold_worst"]["abs_yaw_error_deg"],
                                  "housing_position_dev_mm": cmp.get("housing_position_dev_mm", {"max": None}),
                                  "final_yaw_delta_deg": cmp.get("final_housing_pose_delta", {}).get("yaw_deg")})
    for condition in CONDITIONS:
        rows = [row for row in out["nudges"] if row["condition"] == condition]
        out[f"{condition}_status_counts"] = {status: sum(row["status"] == status for row in rows)
                                             for status in ("PASS", "FAIL", "UNKNOWN")}
    return out


def run_all(package, meshes, manifest, repeat_check=True):
    verified = verify_manifest(manifest, package, meshes)
    pkg = load_package(package, meshes)
    conditions, runs = {}, {}
    for condition in CONDITIONS:
        model, run, verdict, deviation = replay_condition(pkg, package, condition)
        comparison = compare(model, run, package, condition)
        agreement = decision_agreement(verdict, pkg["expected"][condition])
        entry = {"replay_verdict": verdict, "archived": {k: pkg["expected"][condition][k] for k in
                                                          ("status", "legacy_success", "legacy_mode", "legacy_metrics", "hold_acceptance")},
                 "decision_agreement": agreement, "trajectory_comparison": comparison,
                 "disturbance_rule_vs_recorded_edit_max_abs_dev": deviation,
                 "numerics": {"warnings": run["warnings"], "finite": run["finite"]}}
        if repeat_check:
            _, again, _, _ = replay_condition(pkg, package, condition)
            entry["same_machine_rerun_bitwise_identical"] = bool(
                np.array_equal(again["pos_mm"], run["pos_mm"]) and np.array_equal(again["state_qpos"], run["state_qpos"]))
        conditions[condition] = entry
        runs[condition] = (model, run)
    # Discrimination controls: the same comparison must detect a geometry swap and a small disturbance error.
    original_model, original_run = runs["original"]
    swapped = compare(original_model, runs["historical_v4_lead_in"][1], package, "original")
    wrong = dict(pkg["disturbance"], draw=dict(pkg["disturbance"]["draw"]))
    wrong["draw"]["surface_friction"] = wrong["draw"]["surface_friction"] + CONTROL_FRICTION_DELTA
    wrong_model, wrong_run, wrong_verdict, wrong_edit_dev = replay_condition(pkg, package, "original", wrong)
    wrong_cmp = compare(wrong_model, wrong_run, package, "original")
    keys = ("housing_pose_bitwise_identical", "housing_position_dev_mm", "first_nonzero_housing_dev_task_time_s",
            "final_housing_pose_delta")
    controls = {"original_reference_vs_lead_in_geometry_replay": {k: swapped.get(k) for k in keys},
                "original_with_surface_friction_plus": {"delta": CONTROL_FRICTION_DELTA, "edit_dev": wrong_edit_dev,
                                                        "verdict": {k: wrong_verdict[k] for k in ("status", "legacy_mode")},
                                                        **{k: wrong_cmp.get(k) for k in keys}},
                "initial_housing_position_nudges": _nudge_ensemble(pkg, package, original_model)}
    agree = all(all(c["decision_agreement"][k]["agree"] for k in ("composite_status", "legacy_success", "legacy_mode"))
                for c in conditions.values())
    within = all(c["trajectory_comparison"].get("comparable")
                 and c["trajectory_comparison"]["housing_position_dev_mm"]["max"] <= GATE["housing_position_max_abs_dev_mm"]
                 and abs(c["trajectory_comparison"]["final_housing_pose_delta"]["yaw_deg"]) <= GATE["final_abs_yaw_dev_deg"]
                 for c in conditions.values())
    status = STATUSES[0] if agree and within else STATUSES[1]
    manifest_data = _json(manifest)
    nudges = controls["initial_housing_position_nudges"]
    flips = {c: sum(row["status"] != nudges["archived_status"][c] for row in nudges["nudges"] if row["condition"] == c)
             for c in CONDITIONS}
    spread = {c: max((row["housing_position_dev_mm"]["max"] for row in nudges["nudges"]
                      if row["condition"] == c and row["housing_position_dev_mm"]["max"] is not None), default=float("inf"))
              for c in CONDITIONS}
    timing = pkg["timing"]
    method = {
        "scope": "selected independent Tier B: one identified held-out pair, both fixture conditions; not a cohort reproduction",
        "replayed_open_loop": (f"the full per-step target vector of the {len(pkg['schedule']['actuator_names'])} position actuators for settle, "
                               f"task and hold ({timing['total_steps']} steps of {timing['timestep_s']} s), exported as data. The archived "
                               "controller is a time-indexed schedule fixed before the trial (targets depend only on task time), so "
                               "open-loop replay reproduces its control semantics rather than approximating a state-dependent controller."),
        "closed_loop_inside_replay": ("MuJoCo itself simulates the position servos (kp/kv, force limits), the joint-equality finger "
                                      "couplings, contacts and the implicitfast integrator from the archived compiled MJCF."),
        "reimplemented_in_replay_from_declared_data": ["disturbance application from declared rules (checked against recorded edited values)",
                                                       "stepping, pose-cache and 30 Hz qpos recording rules",
                                                       "terminal legacy judge and finite-hold judge from declared thresholds",
                                                       "trajectory and decision comparison"],
        "exported_by_private_forge_side_tool": ["per-step target schedule", "pre-settle initial state",
                                                "seeded disturbance draw and its recorded model edits",
                                                "seat target and collateral start poses",
                                                "reference trajectories decoded from the archived observation (qpos NPZ copied byte-for-byte)"],
        "input_bytes": manifest_data["total_bytes"]}
    limitations = [
        "One pair (seed {}) of the primary reference-profile held-out split; the other 99 pairs, other splits and the historical_fast profile were not replayed.".format(pkg["pair"]["seed"]),
        "The V6 archive does not record per-step actuator targets or the pre-settle state. They were computed by a private exporter that imports the archived Forge source (verified byte-identical to the archived source manifest) with mj_step replaced by a recorder, so no Forge physics ran. The replay therefore tests reproduction of physics, model, disturbance, recording and judging given Forge's planned targets; it does not re-derive the private planner/controller.",
        "The disturbance draw values come from Forge's seeded sampler via the exporter; the replay applies them but does not re-implement the sampler. The drawn kp_scale is not applied by the archived arm operation and is not applied here; bitwise agreement is consistent with that.",
        "Robot meshes ({} bytes) stay outside Git; they are pinned by SHA-256 and the replay cannot run without them.".format(manifest_data["total_bytes"].get("meshes")),
        "Agreement was observed on one machine pair with the same Python, MuJoCo and NumPy wheels; other CPUs, operating systems, MuJoCo or NumPy versions were not tested.",
        ("The original-fixture outcome is numerically chaotic for this trial: {} of 3 picometre-scale initial housing nudges changed its archived status and housing paths diverged by up to {:.2f} mm, "
         "so per-trial agreement here depends on bit-identical floating-point execution and should not be expected from tolerance-based replays elsewhere. "
         "The lead-in condition kept its status in {} of 3 nudges with at most {:.3f} mm divergence. Three nudges are a probe, not a flip-rate estimate.").format(
            flips["original"], spread["original"], 3 - flips["historical_v4_lead_in"], spread["historical_v4_lead_in"]),
        "Simulation only: nothing here establishes physical credibility of the contact geometry or real-robot behaviour.",
        "The exporter ran with a pre-existing local environment holding the archived MuJoCo/NumPy versions, not a byte-identical rebuild of the archived lock; scene bytes were re-composed and matched to the archive, and the exported task duration equals the archived row."]
    return {"schema": "forge.v6_1.tierb-result/1", "status": status,
            "decisions_agree_both_conditions": agree, "trajectory_gate_met_both_conditions": within,
            "gate_declared_before_replay": GATE, "inputs": verified, "pair": pkg["pair"], "method": method,
            "environment": environment(), "conditions": conditions, "discrimination_controls": controls,
            "limitations": limitations}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="write inputs_manifest.json for a package and its meshes")
    manifest.add_argument("--package", required=True)
    manifest.add_argument("--meshes", required=True)
    manifest.add_argument("--out", required=True)
    run = sub.add_parser("run", help="verify inputs, replay both conditions, judge and compare")
    run.add_argument("--package", required=True)
    run.add_argument("--meshes", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--no-repeat-check", action="store_true")
    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest(args.package, args.meshes, args.out)
        return
    result = run_all(args.package, args.meshes, args.manifest, repeat_check=not args.no_repeat_check)
    _write_json(args.result, result)
    print(json.dumps({"status": result["status"], "decisions_agree": result["decisions_agree_both_conditions"],
                      "trajectory_gate_met": result["trajectory_gate_met_both_conditions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
