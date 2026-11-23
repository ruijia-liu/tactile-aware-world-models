"""Generate one assigned robot route in a fresh FlexiTac simulation process."""
import ast
import json
from pathlib import Path
import sys

import numpy as np

from protocol.locking import source_hashes
from .physics import ShapeMaterials, array_sha, command_schedule, right_contact_shapes, sha
from .planner import NativeMPC, protected_grip_candidates

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config"


def write_json(path, body):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(body, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


def route(index, aperture):
    """Route geometry and initialization are generated independently of materials."""
    rng = np.random.default_rng(88113 + index)
    angle = rng.uniform(0, 2*np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    return {"id": f"route{index:03d}", "gripper": aperture, "speed_m_s": rng.uniform(.043, .053),
            "probe_xy_m": (direction * rng.uniform(.0006, .0009)).tolist(),
            "lift_m": rng.uniform(.028, .034),
            "path_xy_m": [(direction * rng.uniform(.023, .031)).tolist()]}


def material(pair):
    return {"static_friction": pair[0], "dynamic_friction": pair[1]}


def branched_commands(actions, ui, phases, branch, design, finger_slots, closed, opened):
    actions, ui = actions.copy(), ui.copy()
    lift = int(np.flatnonzero(phases == "lift")[0])
    fork = lift + design["decision_steps_after_lift"]
    release = int(np.flatnonzero(phases == "release")[0])
    end_release = int(np.flatnonzero(phases == "retract")[0])
    if not lift < fork < release < end_release:
        raise ValueError("Incorrect command phase ordering")
    target = design["branches"][branch]
    previous = ui[fork-1, 15]
    rate = design["maximum_opening_change_per_step"]
    for t in range(fork, release):
        previous = float(np.clip(target, previous-rate, previous+rate))
        ui[t, 15] = previous
    ui[release:end_release, 15] = np.linspace(previous, 1, end_release-release+1)[1:]
    positions = closed + (opened-closed)*ui[fork:, 15]
    actions[fork:, finger_slots[0]] = positions
    actions[fork:, finger_slots[1]] = -positions
    return actions, ui, lift, fork, release


def tactile_summary(raw):
    pads = raw[2:4].astype(np.float64)
    mass = pads.sum(axis=(1, 2))
    x = np.divide(pads.sum(axis=1) @ np.linspace(0, 1, 32), mass, where=mass>0, out=np.zeros_like(mass))
    y = np.divide(pads.sum(axis=2) @ np.linspace(0, 1, 12), mass, where=mass>0, out=np.zeros_like(mass))
    return np.stack([pads.max(axis=(1, 2)), mass/384, x, y], axis=-1).reshape(8).astype(np.float32)


def control_episode(advance, snapshot, obs, actions, cmd_ui, phases, k, fork, release, design,
                    translation, finger_slots, closed, opened, planner, blackout, *,
                    protect_terminal_hold=False, protect_transport_hold=False):
    """Execute the chosen first four controls before replanning from sensor observations."""
    history = [snapshot(obs)]
    observable = {"plug_xyz": [], "ee_xyz": [], "aperture": [], "commands": []}
    if planner.touch:
        observable["touch"] = []
    decisions = []
    planner.reset()
    goal = None
    for t in range(len(actions)):
        if k <= t < release:
            row = history[-1]
            if t == k:
                goal = row["state"][45:48] + translation
            occluded = blackout and k < t <= fork
            plug_obs = np.full(3, np.nan, dtype=np.float32) if occluded else row["state"][45:48]
            observable["plug_xyz"].append(plug_obs)
            observable["ee_xyz"].append(row["ee_position_w"][1])
            observable["aperture"].append(row["state"][14] - row["state"][15])
            if planner.touch:
                observable["touch"].append(tactile_summary(row["tactile"]))
            if t >= fork and (t-fork)%4==0:
                horizon = min(60, release-t)
                proposals = np.stack([cmd_ui[t:t+horizon, [8, 9, 10, 15]].copy()
                                      for _ in design["branches"]])
                previous = cmd_ui[t-1, 15]
                limits = np.arange(1, horizon+1)*design["maximum_opening_change_per_step"]
                for i, target in enumerate(design["branches"].values()):
                    proposals[i, :, 3] = previous + np.clip(target-previous, -limits, limits)
                length = min(4, horizon)
                eligible = protected_grip_candidates(phases, t, length, design, terminal=protect_terminal_hold,
                                                     transport=protect_transport_hold)
                choice, record = planner.choose(observable, proposals, goal, eligible=eligible)
                selected = proposals[choice, :length, 3]
                cmd_ui[t:t+length, 15] = selected
                positions = closed + selected*(opened-closed)
                actions[t:t+length, finger_slots[0]] = positions
                actions[t:t+length, finger_slots[1]] = -positions
                decisions.append(dict(record, step_from_lift=t-k, object_observed=not occluded))
            observable["commands"].append(cmd_ui[t, [8, 9, 10, 15]].copy())
        elif t == release:
            end = int(np.flatnonzero(phases=="retract")[0])
            cmd_ui[t:end, 15] = np.linspace(cmd_ui[t-1, 15], 1, end-t+1)[1:]
            pos = closed + cmd_ui[t:end, 15]*(opened-closed)
            actions[t:end, finger_slots[0]] = pos
            actions[t:end, finger_slots[1]] = -pos
        history.append(snapshot(advance(actions[t])))
    return history, decisions


def collect(env, cfg, app, report, save_report, check, *, repo, row, output, ensemble_file=None,
            design_path=CONFIG / "design.json", protect_transport_hold=False):
    """Replay one physical prefix at fixed materials and execute each assigned action or policy."""
    import torch

    design = json.loads(design_path.read_bytes())
    config_path = CONFIG / "runtime.json"
    config = json.loads(config_path.read_bytes())
    idx = row["route_index"]
    pk, fk = row["plug_material"], row["pad_material"]
    names = row["pair_action_order"]
    closed_loop = ensemble_file is not None
    expected_names = {"blind", "touch"} if closed_loop else set(design["branches"])
    if (set(names) != expected_names or len(names) != len(expected_names)
            or row["geometry_seed"] != 88113 + idx or row["visibility"] not in {"visible", "mask"}
            or not np.isfinite(row["contact_offset_y_m"])):
        raise ValueError("Invalid action assignment, geometry seed, visibility, or contact perturbation")
    if protect_transport_hold:
        if not closed_loop or row.get("hold_seconds") != 0.35:
            raise ValueError("Protected transport requires the fixed 0.35 s hold")
        config["seconds"]["hold"] = row["hold_seconds"]
    output = Path(output)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    steps_per_s = design["native_hz"]
    if not np.isclose(float(cfg.physics_dt), 1 / steps_per_s):
        raise ValueError("Simulator timestep differs from the experiment design")
    planners = {name: NativeMPC(ensemble_file, name == "touch", 0.) for name in names} if closed_loop else {}
    src_dir = Path(repo)/"Isaacsim_tactile_env"
    sys.path.insert(0, str(src_dir))
    from aloha_ik_controller import AlohaArmIKController, GRIPPER_CLOSED, GRIPPER_OPEN
    upstream = src_dir/"viser_interface.py"
    nodes = [n for n in ast.parse(upstream.read_text("utf8")).body
             if isinstance(n, ast.FunctionDef) and n.name in {"upgrade_ik_to_pose_mode", "compute_pose"}]
    if len(nodes) != 2:
        raise RuntimeError("Missing upstream IK functions")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(upstream), "exec"), namespace)
    ctrls = [AlohaArmIKController(env._robot, cfg.device, arm) for arm in ("left", "right")]
    for ctrl in ctrls:
        namespace["upgrade_ik_to_pose_mode"](ctrl)
    finger_slots = list(map(int, ctrls[1]._gripper_slots))
    if finger_slots != [14, 15]:
        raise ValueError(f"Changed right-finger slots: {finger_slots}")
    shape_links = right_contact_shapes(env._robot, config["right_contact_links"])
    indices = [i for link in shape_links for i in link["shape_indices"]]
    plug = ShapeMaterials(env._plug_obj.root_physx_view, env._sim, cfg.device, "plug")
    pads = ShapeMaterials(env._robot.root_physx_view, env._sim, cfg.device, "right pads", indices)
    source_files = (HERE / "physics.py", design_path, config_path, upstream,
                    src_dir / "aloha_ik_controller.py")
    provenance = {p.name: sha(p) for p in source_files}
    provenance.update(source_hashes())
    if closed_loop:
        provenance["ensemble.npz"] = sha(ensemble_file)
    summary = {"backend": "FlexiTac / Isaac Sim 6 / PhysX", "source_sha256": provenance,
               "episodes": [], "native_hz": steps_per_s}
    report["interactions"] = summary
    save_report()
    neutral = config["reference_material"]
    initial_joint = np.asarray(config["initial_joint_targets"], dtype=np.float32)
    quat = np.asarray(config["right_ee_quaternion_wxyz"], dtype=float)
    quat /= np.linalg.norm(quat)

    def advance(cmd):
        if not app.is_running():
            raise RuntimeError("Simulator stopped")
        obs, _, terminated, truncated, _ = env.step(cmd)
        check()
        if terminated or truncated:
            raise RuntimeError("Unexpected environment termination")
        return obs

    def reset(seed, jitter, fixed=None):
        env.reset(seed=seed)
        robot = env._robot
        robot.write_root_state_to_sim(robot.data.default_root_state.clone())
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        robot.reset()
        for obj in (env._plug_obj, env._socket_obj):
            root_state = obj.data.default_root_state.clone()
            root_state[:, 1] += jitter
            if fixed is not None and obj is env._plug_obj:
                root_state[:, 1] += row["contact_offset_y_m"]
            obj.write_root_state_to_sim(root_state)
            obj.reset()
        for sensor in env._tactile_sensors:
            sensor.reset()
        if fixed is None:
            plug.apply(neutral, write=True)
            pads.apply(neutral, write=True)
        else:
            plug.apply(fixed[0], write=False)
            pads.apply(fixed[1], write=False)
        for _ in range(config["initial_warmup_steps"]):
            advance(initial_joint)

    def inverse(desired):
        arm_values = []
        for i, ctrl in enumerate(ctrls):
            arm = desired[i*8:i*8+8]
            action = namespace["compute_pose"](ctrl, arm[:3], arm[3:7], arm[7])
            arm_values.append(action[i*8:i*8+8])
        return np.concatenate(arm_values).astype(np.float32)

    def snapshot(obs):
        roots = [obj.data.root_state_w[0].detach().cpu().numpy().copy()
                 for obj in (env._robot, env._plug_obj, env._socket_obj)]
        return {"state": np.r_[obs["joint_pos"], obs["joint_vel"], *roots].astype(np.float32),
                "tactile": obs["tactile"].copy(), "time_s": float(env._sim.current_time),
                "ee_position_w": np.stack([ctrl.get_ee_pos() for ctrl in ctrls]).astype(np.float32),
                "ee_quaternion_w": np.stack([ctrl.get_ee_quat() for ctrl in ctrls]).astype(np.float32)}

    family = route(idx, design["reference_gripper"])
    seed = design["seed_base"] + idx
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-.00055, .00055)
    reset(seed, offset)
    left, left_q = ctrls[0].get_ee_pos(), ctrls[0].get_ee_quat()
    grasp = env._plug_obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
    grasp += config["base_right_ee_offset_from_plug_m"]
    ui, phases, segment_ids, _ = command_schedule(config, family, grasp, left, left_q, quat)
    ready_ui = np.r_[left, left_q, 1, grasp+config["approach_delta_m"], quat, 1]
    prefix = []
    for _ in range(config["ready_warmup_steps"]):
        cmd = inverse(ready_ui)
        prefix.append(cmd)
        advance(cmd)
    reference = []
    for planned in ui:
        cmd = inverse(planned)
        advance(cmd)
        reference.append(cmd)
    reference = np.stack(reference)
    candidates = {name: branched_commands(reference, ui, phases, name, design, finger_slots,
                                           GRIPPER_CLOSED, GRIPPER_OPEN) for name in design["branches"]}
    anchor_plan = next(iter(candidates.values()))
    fork = anchor_plan[3]
    if any(not np.array_equal(anchor_plan[0][:fork], candidate[0][:fork]) for candidate in candidates.values()):
        raise AssertionError("Branches do not share the initial commands")
    pmat = material(design["plug_materials"][pk])
    fmat = material(design["pad_materials"][fk])
    plug.apply(pmat, write=True)
    pads.apply(fmat, write=True)
    plug.apply(pmat, write=False)
    pads.apply(fmat, write=False)
    for method in names:
        decisions = []
        arm_meta = {}
        if closed_loop:
            blackout = row["visibility"] == "mask"
            name = f"{method}_mean_{row['visibility']}"
            _, _, k, j, release = anchor_plan
            actions, cmd_ui = reference.copy(), ui.copy()
            arm_meta = {"controller": name, "reads_touch": method == "touch", "risk_weight": 0.,
                        "blackout": blackout, "object_observation": "simulator pose observation mask"}
        else:
            name = method
            actions, cmd_ui, k, j, release = candidates[name]
            if not np.isclose(cmd_ui[release-1, 15], design["branches"][name], rtol=0, atol=1e-7):
                raise ValueError("Scheduled gripper opening does not match assigned action")
        scenario = f"{family['id']}_{pk}_{fk}"
        eid = f"{scenario}_{name}"
        reset(seed, offset, (pmat, fmat))
        for cmd in prefix:
            obs = advance(cmd)
        start = [plug.apply(pmat, write=False), pads.apply(fmat, write=False)]
        translation = np.r_[family["path_xy_m"][-1], family["lift_m"] - config["preprobe_rise_m"]]
        if closed_loop:
            history, decisions = control_episode(advance, snapshot, obs, actions, cmd_ui, phases, k, j,
                release, design, translation, finger_slots, GRIPPER_CLOSED, GRIPPER_OPEN,
                planners[method], blackout, protect_transport_hold=protect_transport_hold)
        else:
            history = [snapshot(obs)]
            for action in actions:
                history.append(snapshot(advance(action)))
        end = [plug.apply(pmat, write=False), pads.apply(fmat, write=False)]
        arrays = {key: np.asarray([v[key] for v in history]) for key in history[0]}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite native sensor values")
        np.testing.assert_allclose(np.diff(arrays["time_s"]), 1/steps_per_s, rtol=1e-5, atol=1e-8)
        path = raw/(eid+".npz")
        np.savez_compressed(path, **arrays, action=actions, ui_action=cmd_ui, phase=phases, segment_id=segment_ids)
        # Target translation is fixed by the route, independent of the material or outcome.
        target = arrays["state"][k, 45:48]+translation
        meta = {"episode_id": eid, "scenario_id": scenario, "family_id": family["id"], "candidate": name,
                "seed": seed, "reset_y_offset_m": offset, "plug_material": pk, "pad_material": fk,
                "physics_dt": 1/steps_per_s, "first_lift_index": k, "decision_index": j,
                "release_index": release, "target_xyz_m": target.tolist(), "route_geometry": family,
                "prefix_sha256": array_sha(actions[:j]), "common_ready_prefix_sha256": array_sha(np.stack(prefix)),
                "action_sha256": array_sha(actions), "npz_sha256": sha(path), "source_sha256": provenance,
                "material_readback_start": start, "material_readback_end": end}
        meta.update(arm_meta, decisions=decisions, material_protocol="fixed_from_reset_after_neutral_reference")
        meta.update({key: row[key] for key in ("visibility", "contact", "contact_offset_y_m",
                    "pair_action_order", "geometry_seed", "manifest_sha256")}, candidate_openings=design["branches"])
        if protect_transport_hold:
            meta.update(protect_transport_hold=True, hold_seconds=config["seconds"]["hold"])
        write_json(raw / (eid+".json"), meta)
        summary["episodes"].append({"episode_id": eid, "sha256": meta["npz_sha256"]})
        save_report()
    summary["passed"] = len(summary["episodes"]) == len(names)
    write_json(output / "collection.json", summary)
    save_report()
