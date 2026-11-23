"""Fit matched dynamics ensembles and lock only after fresh development closed-loop checks."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.dynamics import atomic_output, gradients, initialize, scalers_for, supervised_arrays, tactile_features
from methods.motion import offsets, save_rows, score
from protocol.locking import digest, source_hashes

SEEDS = (11, 23, 47, 59, 71, 83, 97, 101)
ARMS = ("blind", "early")
EPOCHS = 36
BATCH = 192
HORIZON = 20


def write_json(path, value):
    with atomic_output(path) as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf8"))


def checked_episodes(stage, root, design, protocol, design_file, ensemble_sha=None):
    manifest = protocol / (stage + ".json")
    lock = digest(manifest)
    assemblies = []
    files = sorted(root.glob("row-*/raw/*.json"))
    for index, expected in enumerate(json.loads(manifest.read_bytes())):
        route = expected["route_index"]
        report_file = root / f"row-{index:02d}.json"
        report = json.loads(report_file.read_bytes())
        collection = report["interactions"]
        if report["status"] != "PASSED" or not collection["passed"]:
            raise ValueError(f"Collection not completed for route {route}")
        records = {row["episode_id"]: row["sha256"] for row in collection["episodes"]}
        arm_names = expected["pair_action_order"]
        suffix = "mean_" + expected["visibility"]
        if stage != "branches":
            arm_names = [arm + "_" + suffix for arm in arm_names]
        raw = report_file.with_suffix("") / "raw"
        scenario = f"route{route:03d}_{expected['plug_material']}_{expected['pad_material']}"
        if [rec["episode_id"] for rec in collection["episodes"]] != [f"{scenario}_{arm}" for arm in arm_names]:
            raise ValueError(f"Missing or duplicate arms for route {route}")
        samples = []
        for name in arm_names:
            meta = json.loads((raw / f"{scenario}_{name}.json").read_bytes())
            if any(meta[k] != expected[k] for k in (
                    "geometry_seed", "visibility", "contact", "contact_offset_y_m",
                    "plug_material", "pad_material", "pair_action_order")):
                raise ValueError(f"Assignment drift for route {route}")
            if (meta["manifest_sha256"] != lock or meta["source_sha256"]["design.json"] != digest(design_file)
                    or meta["candidate_openings"] != design["branches"]
                    or records[meta["episode_id"]] != meta["npz_sha256"]
                    or (ensemble_sha and meta["source_sha256"]["ensemble.npz"] != ensemble_sha)):
                raise ValueError(f"Unverified collected episode for route {route}")
            row, anchor = score(meta, raw, design)
            if stage != "branches":
                if (meta["risk_weight"] != 0. or meta["blackout"] != (expected["visibility"] == "mask")
                        or meta.get("protect_terminal_hold", False)
                        or not meta.get("protect_transport_hold", False)
                        or meta.get("hold_seconds") != .35 or expected.get("hold_seconds") != .35):
                    raise ValueError("Validation policy, grip protection or timing differs from its stage")
            samples.append((meta, row, anchor, raw))
        for left, right in combinations(samples, 2):
            if any(left[2][key] != right[2][key] for key in ("prefix", "ready_prefix")):
                raise ValueError(f"Commands before the fork differed on route {route}")
        assemblies.append((expected, samples))
    if len(files) != sum(len(samples) for _, samples in assemblies):
        raise ValueError("Collection includes undeclared or repeated records")
    return assemblies


@dataclass
class Trajectory:
    state: np.ndarray
    commands: np.ndarray
    pads: np.ndarray
    visibility: str


def trajectory(meta, raw):
    k, stop = (meta[key] for key in ("first_lift_index", "release_index"))
    with np.load(raw / (meta["episode_id"] + ".npz"), allow_pickle=False) as episode:
        state = episode["state"][k:stop+1]
        ee = episode["ee_position_w"][k:stop+1, 1]
        aperture = state[:, 14] - state[:, 15]
        values = np.column_stack((state[:, 45:48], np.zeros_like(ee), ee, aperture))
        commands = episode["ui_action"][k:stop, [8, 9, 10, 15]]
        pads = tactile_features(episode["tactile"][k:stop+1])
    return Trajectory(values.astype(np.float32), commands.astype(np.float32), pads, meta["visibility"])


def transitions(episodes):
    return tuple(np.concatenate(rows) for rows in zip(*[
        (ep.state[:-1], ep.commands, ep.pads[:-1], np.diff(ep.state, axis=0), ep.pads[1:]) for ep in episodes
    ]))


def validation_batch(episodes):
    if any(len(ep.commands) < 2 * HORIZON for ep in episodes):
        raise ValueError("Episode ends before the validation endpoint")
    return (np.stack([ep.state[:2*HORIZON+1] for ep in episodes]),
            np.stack([ep.pads[:HORIZON+1] for ep in episodes]),
            np.stack([ep.commands[:2*HORIZON] for ep in episodes]),
            np.array([ep.visibility == "visible" for ep in episodes]))


def forecast_errors(model, validation):
    truth, measured_pads, commands, visible = validation
    state, pads = truth[:, 0].copy(), measured_pads[:, 0].copy()
    for t in range(HORIZON):
        state, pads, _ = model.step(state, pads, commands[:, t])
        state[:, 6:] = truth[:, t+1, 6:]
        state[visible, :3] = truth[visible, t+1, :3]
        if model.name == "early":
            pads = measured_pads[:, t+1]
    delta = []
    for t in range(HORIZON, 2*HORIZON):
        state, pads, _ = model.step(state, pads, commands[:, t])
        delta.append(1000. * (state[:, :3].astype(float) - truth[:, t+1, :3]))
    return np.stack(delta, axis=1)


def learn(arm, seed, scaled, rows, validation):
    model = initialize(arm, seed, scaled)
    x, y = supervised_arrays(model, rows)
    names = ["w1", "b1", "w2", "b2", "wo", "bo"]
    m = {k: np.zeros_like(model.p[k]) for k in names}
    v = {k: np.zeros_like(model.p[k]) for k in names}
    generator = np.random.default_rng(seed + 8800)
    best = float("inf")
    step = 0
    curve = []
    for epoch in range(1, EPOCHS + 1):
        order = generator.permutation(len(x))
        loss = 0.
        for start in range(0, len(order), BATCH):
            ids = order[start:start+BATCH]
            batch_loss, g = gradients(model, x[ids], y[ids])
            loss += batch_loss * len(ids)
            clip = min(1., 5. / max(1e-12, sum(float(np.vdot(a, a)) for a in g.values())**.5))
            step += 1
            rate = .001 * np.sqrt(1-.999**step) / (1-.9**step)
            for k in names:
                a = g[k] * clip
                m[k] *= .9
                m[k] += .1 * a
                v[k] *= .999
                v[k] += .001 * a**2
                model.p[k] -= rate * m[k] / (np.sqrt(v[k]) + 1e-8)
        err = forecast_errors(model, validation)[:, -1]
        mse = float(np.mean(np.sum(err**2, axis=-1)))
        if not np.isfinite(mse):
            raise ValueError("Nonfinite validation error")
        curve.append(dict(arm=arm, seed=seed, epoch=epoch, train_loss=loss / len(x), h20_mse_mm2=mse))
        if mse < best:
            selected, best = epoch, mse
            weights = {key: value.copy() for key, value in model.p.items()}
    model.p.update(weights)
    return model, curve, dict(arm=arm, seed=seed, selected_epoch=selected, h20_mse_mm2=best)


def branch_summary(assemblies, design, out):
    metrics, pairs = [], []
    for assignment, samples in assemblies:
        for _, row, _, _ in samples:
            metrics.append(dict(**row, split=assignment["split"], visibility=assignment["visibility"],
                                contact=assignment["contact"]))
        for first, second in combinations(samples, 2):
            gap = offsets(first[2]["j"], second[2]["j"])
            matched = all(gap[key] <= limit for key, limit in design["matched_reset_limits"].items())
            pairs.append(dict(route_index=assignment["route_index"], first=first[1]["arm"], second=second[1]["arm"],
                              matched=matched, slip_disagrees=first[1]["slip_h60"] != second[1]["slip_h60"], **gap))
    save_rows(out / "branches-metrics.csv", metrics)
    save_rows(out / "branches-fork-qc.csv", pairs)
    return dict(physical_episodes=len(metrics), forks=len(pairs), matched_forks=sum(p["matched"] for p in pairs),
                matched_slip_disagreements=sum(p["matched"] and p["slip_disagrees"] for p in pairs),
                per_action={arm: dict(n=sum(r["arm"] == arm for r in metrics),
                                      successes=sum(r["success"] for r in metrics if r["arm"] == arm),
                                      slips_h60=sum(r["slip_h60"] for r in metrics if r["arm"] == arm))
                            for arm in design["branches"]})


def development_errors(cohorts, member_errors, out):
    groups = [samples for assignment, samples in cohorts if assignment["split"] == "select"]
    labels = [meta["episode_id"] for samples in groups for meta, _, _, _ in samples]
    names = [meta["candidate"] for samples in groups for meta, _, _, _ in samples]
    single, gaps = [], []
    for arm, members in member_errors.items():
        errors = np.mean(members, axis=0)
        for i, name in enumerate(labels):
            for horizon in (1, 5, 10, 20):
                residual = errors[i, horizon-1]
                single.append(dict(model=arm, episode_id=name, action=names[i], horizon=horizon,
                                   bias_x_mm=residual[0], bias_y_mm=residual[1], bias_z_mm=residual[2],
                                   squared_error_mm2=float(residual @ residual)))
        for first in range(0, len(labels), 3):
            for i, j in combinations(range(first, first+3), 2):
                for horizon in (1, 5, 10, 20):
                    residual = errors[i, horizon-1] - errors[j, horizon-1]
                    gaps.append(dict(model=arm, first_episode=labels[i], second_episode=labels[j],
                                     horizon=horizon, squared_action_difference_error_mm2=float(residual @ residual)))
    save_rows(out / "selection-motion-errors.csv", single)
    save_rows(out / "selection-action-difference-errors.csv", gaps)


def fit(root, out, design, protocol, design_file):
    cohorts = checked_episodes("branches", root, design, protocol, design_file)
    summary = branch_summary(cohorts, design, out)
    grouped = defaultdict(list)
    for assignment, samples in cohorts:
        grouped[assignment["split"]].extend(trajectory(meta, raw) for meta, _, _, raw in samples)
    if {s: len(eps) for s, eps in grouped.items()} != {"train": 48, "select": 24}:
        raise ValueError("Unexpected branch split")
    rows = transitions(grouped["train"])
    scaled = scalers_for(rows)
    validation = validation_batch(grouped["select"])
    curves, selected = [], []
    by_arm = defaultdict(list)
    member_errors = defaultdict(list)
    for seed in SEEDS:
        for arm in ARMS:
            model, curve, record = learn(arm, seed, scaled, rows, validation)
            by_arm[arm].append(model.p)
            curves.extend(curve)
            selected.append(record)
            member_errors[arm].append(forecast_errors(model, validation))
            print(record, flush=True)
    weights = out / "ensemble.npz"
    bank = {f"{arm}_{key}": np.stack([member[key] for member in members])
            for arm, members in by_arm.items() for key in members[0]}
    with atomic_output(weights) as stream:
        np.savez_compressed(stream, **bank)
    development_errors(cohorts, member_errors, out)
    save_rows(out / "training-curves.csv", curves)
    save_rows(out / "selected-epochs.csv", selected)
    summary.update(status="FITTED", ensemble_sha256=digest(weights), design_sha256=digest(design_file),
                   branches_manifest_sha256=digest(protocol / "branches.json"),
                   development_route_indices=[assignment["route_index"] for assignment, _ in cohorts],
                   epoch_budget=EPOCHS, training_transitions=len(rows[0]), seeds=list(SEEDS),
                   select_h20_ensemble_mse_mm2={arm: float(np.mean(np.sum(np.mean(err, axis=0)[:, -1]**2, axis=-1)))
                                                for arm, err in member_errors.items()})
    write_json(out / "fit.json", summary)


def freeze(root, out, design, protocol, design_file):
    fitted = json.loads((out / "fit.json").read_bytes())
    sha = digest(out / "ensemble.npz")
    if fitted["ensemble_sha256"] != sha or fitted["design_sha256"] != digest(design_file):
        raise ValueError("Weights or native candidate design changed since training")
    cohorts = checked_episodes("validation", root, design, protocol, design_file, sha)
    route_indices = [assignment["route_index"] for assignment, _ in cohorts]
    if set(route_indices) & set(fitted["development_route_indices"]):
        raise ValueError("Selection and development closed loop share routes")
    metrics = [dict(**row, visibility=assignment["visibility"], contact=assignment["contact"])
               for assignment, samples in cohorts for _, row, _, _ in samples]
    save_rows(out / "closed-loop-validation.csv", metrics)
    successes = Counter(r["arm"].split("_")[0] for r in metrics if r["success"])
    reasons = []
    if fitted["matched_slip_disagreements"] == 0:
        reasons.append("No matched development branch showed an action-dependent H60 slip outcome")
    if any(successes[arm] == 0 for arm in ("blind", "touch")):
        reasons.append("At least one method completed zero of eight closed-loop development attempts")
    sources = source_hashes()
    for _, samples in cohorts:
        for meta, _, _, _ in samples:
            if any(meta["source_sha256"].get(name) != version for name, version in sources.items()):
                raise ValueError("Simulator code differs from the validated controller")
    manifest = protocol / "manifest.json"
    reserved = {row["route_index"] for row in json.loads(manifest.read_bytes())["trials"]}
    development = sorted(set(fitted["development_route_indices"]) | set(route_indices))
    if reserved.intersection(development):
        raise ValueError("Confirmation and development share routes")
    report = dict(status="BLOCKED" if reasons else "PASSED", reasons=reasons, ensemble_sha256=sha,
                  development_route_indices=development, design_sha256=digest(design_file),
                  validation_manifest_sha256=digest(protocol / "validation.json"),
                  validation_stage="validation", validation_route_indices=route_indices,
                  confirmation_manifest_sha256=digest(manifest), controller_source_sha256=sources,
                  control_rule=dict(protect_transport_hold=True, hold_seconds=.35,
                                    eligible_openings_during_transport_hold=["tight", "hold"]),
                  success_criteria=design["success"],
                  development_successes_by_method={k: successes[k] for k in ("blind", "touch")},
                  development_trials_per_method=len(cohorts))
    write_json(out / "closed-loop-gate.json", report)
    if reasons:
        raise SystemExit("; ".join(reasons))
    write_json(out / "freeze.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "freeze"))
    parser.add_argument("collection", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise SystemExit("Trajectory validation requires enabled assertions")
    if (args.output / "freeze.json").exists():
        raise SystemExit("A frozen result already exists. Select a new output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    design = json.loads(args.design.read_bytes())
    (fit if args.stage == "fit" else freeze)(args.collection, args.output, design, args.protocol, args.design)
