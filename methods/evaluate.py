"""Score the locked two-controller FlexiTac test once every registered pair is complete."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.dynamics import tactile_features
from methods.motion import offsets, score as score_native
from protocol.locking import check_freeze, digest
from simulation.planner import NativeMPC

METHODS = ("blind", "touch")
CELLS = (("visible", "nominal"), ("mask", "nominal"),
         ("visible", "perturbed"), ("mask", "perturbed"))
SEED = 20261123
DRAWS = 10_000
def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def registered_trials(manifest):
    trials = read(manifest)["trials"]
    require(len(trials) == 100, "Expected exactly 100 paired scenarios")
    for key in ("route_index", "family_id", "scenario_id", "geometry_seed"):
        require(len({t[key] for t in trials}) == 100, f"Repeated {key} across scenarios")
    counts = Counter((t["visibility"], t["contact"]) for t in trials)
    require(counts == dict.fromkeys(CELLS, 25), "Expected 25 scenarios in each of the four cells")
    for t in trials:
        route = t["family_id"]
        require(route == f"route{t['route_index']:03d}", f"Inconsistent route ID {route}")
        require(t["scenario_id"] == f"{route}_{t['plug_material']}_{t['pad_material']}", "Invalid scenario ID")
        offset = abs(t["contact_offset_y_m"])
        require(offset == 0 if t["contact"] == "nominal" else .002 <= offset <= .004,
                f"Unexpected contact offset for {route}")
    return trials


def episode_ids(trial):
    return [f"{trial['scenario_id']}_{method}_mean_{trial['visibility']}" for method in METHODS]


def index_raw(root, expected):
    files = {}
    for path in root.rglob("*.npz"):
        if root != path.parent and path.parent.name != "raw":
            continue
        require(path.stem not in files, f"Duplicate episode {path.stem}")
        files[path.stem] = path
    found = set(files)
    require(found == expected, f"Expected {len(expected)} trials, found {len(found)}. "
            f"Missing {sorted(expected - found)[:4]}, extra {sorted(found - expected)[:4]}")
    return files


def validate_meta(meta, path, trial, method, model_hash, design_hash, manifest_hash, source_hashes):
    for key in ("family_id", "scenario_id", "plug_material", "pad_material"):
        require(meta[key] == trial[key], f"{path.name}: mismatched {key}")
    visibility = trial["visibility"]
    arm = f"{method}_mean_{visibility}"
    require(meta["episode_id"] == path.stem and meta["candidate"] == meta["controller"] == arm
            and meta["reads_touch"] == (method == "touch") and meta["risk_weight"] == 0
            and meta["blackout"] == (visibility == "mask"), f"Invalid controller {path.stem}")
    require(meta["source_sha256"]["ensemble.npz"] == model_hash, f"Incorrect model in {path.name}")
    require(meta["source_sha256"]["design.json"] == design_hash, f"Incorrect design in {path.name}")
    require(all(meta["source_sha256"].get(name) == version for name, version in source_hashes.items()),
            f"Controller source changed in {path.name}")
    require(meta["manifest_sha256"] == manifest_hash, f"Incorrect manifest in {path.name}")
    require(meta.get("protect_transport_hold") and not meta.get("protect_terminal_hold")
            and meta.get("hold_seconds") == .35, f"Incorrect transport and hold policy in {path.name}")
    require(meta["seed"] == trial["reset_seed"], f"Incorrect reset seed in {path.name}")
    for key in ("visibility", "contact", "contact_offset_y_m", "geometry_seed", "pair_action_order"):
        require(meta[key] == trial[key], f"{path.name}: mismatched {key}")


def forecast(models, episode, meta):
    k, j = meta["first_lift_index"], meta["decision_index"]
    require(j == k + 20 and j + 20 <= meta["release_index"], "H20 endpoint outside task")
    with np.load(episode, allow_pickle=False) as archive:
        history_state = archive["state"][k:j+1]
        plug = history_state[:, 45:48].copy()
        if meta["blackout"]:
            plug[1:] = np.nan
        # Both models receive identical real actions. Measurements stop at the decision frame.
        actions = archive["ui_action"][k:j+20, [8, 9, 10, 15]].astype(np.float32)
        history = dict(plug_xyz=plug, ee_xyz=archive["ee_position_w"][k:j+1, 1],
                       aperture=history_state[:, 14] - history_state[:, 15], commands=actions[:j-k])
        pads = tactile_features(archive["tactile"][k:j+1])
        predicted = {}
        for method, model in models.items():
            model.reset()
            model._assimilate(dict(history, touch=pads) if method == "touch" else history)
            state, touch_state = model._state, model._touch
            for cmd in actions[j-k:]:
                state, touch_state = model._step(state, touch_state, cmd)
            predicted[method] = state[:, 0, :3].astype(float).mean(axis=0)
        # Object truth after the joint prediction origin enters scoring only.
        truth = archive["state"][j+20, 45:48].astype(float)
        origin = history_state[0, 45:48].astype(float)
    fields = {}
    for method, position in {"true": truth, **predicted}.items():
        for axis, value in zip("xyz", 1000 * (position - origin)):
            fields[f"{method}_{axis}_mm"] = float(value)
    for method, position in predicted.items():
        residual = 1000 * (position - truth)
        fields[f"{method}_h20_mm2"] = float(residual @ residual)
    require(np.isfinite(list(fields.values())).all(), f"Invalid H20 prediction in {episode.name}")
    return fields


def score_pairs(trials, paths, models, design, hashes):
    episodes, pairs = [], []
    for trial in trials:
        rows, anchors = [], []
        for method, eid in zip(METHODS, episode_ids(trial)):
            path = paths[eid]
            meta = read(path.with_suffix(".json"))
            validate_meta(meta, path, trial, method, *hashes)
            result, anchor = score_native(meta, path.parent, design)
            rows.append(dict(episode=eid, method=method, success=result["success"],
                             **forecast(models, path, meta)))
            anchors.append(anchor)
        one, two = anchors
        require(one["prefix"] == two["prefix"] and one["ready_prefix"] == two["ready_prefix"],
                f"Actions differed before fork in {trial['scenario_id']}")
        gaps = offsets(one["j"], two["j"])
        matched = all(gaps[key] <= limit for key, limit in design["matched_reset_limits"].items())
        identity = {key: trial[key] for key in ("family_id", "scenario_id", "visibility", "contact")}
        pair = dict(identity, matched_reset=matched, **gaps)
        for method, row in zip(METHODS, rows):
            pair[f"{method}_success"] = row["success"]
            pair[f"{method}_h20_mm2"] = float(np.mean([r[f"{method}_h20_mm2"] for r in rows]))
            episodes.append(dict(identity, **row))
        pairs.append(pair)
    return episodes, pairs


def metrics(values):
    # Last dimension is always (blind squared error, touch squared error, blind success, touch success).
    blind, touch, blind_success, touch_success = np.moveaxis(values, -1, 0)
    relative = np.divide(100 * (blind - touch), blind, where=blind > 0,
                         out=np.full(np.shape(blind), np.nan))
    return dict(blind_h20_mm2=blind, touch_h20_mm2=touch, touch_minus_blind_mm2=touch-blind,
                h20_reduction_percent=relative, blind_success_percent=100*blind_success,
                touch_success_percent=100*touch_success, success_gain_pp=100*(touch_success-blind_success))


def estimate(point, samples):
    if not np.isfinite(point):
        return {"estimate": None, "ci95": None}
    limits = None
    if samples is not None and np.isfinite(samples).all():
        limits = np.quantile(samples, [.025, .975]).tolist()
    return {"estimate": float(point), "ci95": limits}


def summarize(pairs, include_interactions=True):
    keys = ("blind_h20_mm2", "touch_h20_mm2", "blind_success", "touch_success")
    groups, boot, joint = {}, {}, []
    rng = np.random.default_rng(SEED)
    for cell in CELLS:
        selected = [row for row in pairs if (row["visibility"], row["contact"]) == cell]
        if not selected:
            continue
        a = np.array([[r[k] for k in keys] for r in selected], dtype=float)
        groups[cell] = a
        boot[cell] = a[rng.integers(len(a), size=(DRAWS, len(a)))].mean(axis=1)
        joint.extend(selected)
    def subset(cells):
        values = np.concatenate([groups[c] for c in cells])
        point = metrics(values.mean(axis=0))
        sample = metrics(np.average([boot[c] for c in cells], axis=0,
                                                       weights=[len(groups[c]) for c in cells]))
        report = {k: estimate(v, sample.get(k)) for k, v in point.items()}
        bd, td = values[:, 2], values[:, 3]
        report.update(scenarios=len(values), blind_successes=int(bd.sum()), touch_successes=int(td.sum()),
                      touch_only_success=int(np.sum(td > bd)), blind_only_success=int(np.sum(bd > td)),
                      success_ties=int(np.sum(td == bd)))
        route_metrics = {"blind_h20_mm2": values[:, 0], "touch_h20_mm2": values[:, 1],
                         "touch_minus_blind_mm2": values[:, 1]-values[:, 0],
                         "blind_success_percent": 100*bd, "touch_success_percent": 100*td,
                         "success_gain_pp": 100*(td-bd)}
        report["across_scenario_sample_sd"] = {
            k: float(v.std(ddof=1)) if len(v) > 1 else None for k, v in route_metrics.items()}
        if np.array_equal(bd, td):
            # A zero-width empirical bootstrap cannot exclude unseen discordant pairs.
            bound = 100 * (1 - .05**(1/len(values)))
            report["zero_discordance_conservative_95_bounds_pp"] = [-bound, bound]
        return report, point, sample

    overall = subset(list(groups))[0]
    slices, interactions = {}, {}
    for cell in groups:
        slices["/".join(cell)] = subset([cell])[0]
    if include_interactions:
        v, m, p, mp = CELLS
        contrasts = {
            "mask_minus_visible": [([m, mp], 1), ([v, p], -1)],
            "perturbed_minus_nominal": [([p, mp], 1), ([v, m], -1)],
            "mask_by_contact_difference_of_differences": [([mp], 1), ([m], -1), ([p], -1), ([v], 1)],
        }
        for c in (v, m, p):
            contrasts[f"mask_perturbed_minus_{'_'.join(c)}"] = [([mp], 1), ([c], -1)]
        for name, terms in contrasts.items():
            components = [(subset(c), sign) for c, sign in terms]
            interactions[name] = {
                key: estimate(sum(sign * x[1][key] for x, sign in components),
                              sum(sign * x[2][key] for x, sign in components))
                for key in ("touch_minus_blind_mm2", "h20_reduction_percent", "success_gain_pp")}
    matched_count = sum(r["matched_reset"] for r in joint)
    return dict(overall=overall, strata=slices, interactions=interactions,
                matched_reset_pairs=matched_count, mismatched_reset_pairs=len(joint)-matched_count)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Episode files or their parent collection directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(__debug__, "Trajectory validation requires enabled assertions")
    require(not args.output.exists() or not any(args.output.iterdir()), "Use a new or empty output directory")
    frozen = check_freeze(args.freeze, args.ensemble, args.design, args.manifest)
    trials = registered_trials(args.manifest)
    hashes = (frozen["ensemble_sha256"], digest(args.design), digest(args.manifest),
              frozen["controller_source_sha256"])
    expected = {eid for trial in trials for eid in episode_ids(trial)}
    files = index_raw(args.raw, expected)
    models = {method: NativeMPC(args.ensemble, method == "touch", 0.) for method in METHODS}
    episodes, pairs = score_pairs(trials, files, models, read(args.design), hashes)
    summary = dict(status="COMPLETE_REGISTERED_TEST", executions=len(episodes), independent_scenarios=len(pairs),
                   ensemble_sha256=hashes[0], design_sha256=hashes[1], manifest_sha256=hashes[2],
                   h20_metric="Mean squared Euclidean error (mm²) at 20 steps, averaging both executions per route. Both models condition on the same realized commands without future sensor readings.",
                   success_metric="Completed transfers for each executing policy. The paired difference uses percentage points.",
                   contrast_units="Positive H20 relative reduction and positive success gain favor tactile input. Interactions between relative reductions are in percentage points.",
                   uncertainty=f"Paired stratified percentile intervals, {DRAWS} draws with seed {SEED}, unadjusted for multiple endpoints.",
                   **summarize(pairs))
    matched = [pair for pair in pairs if pair["matched_reset"]]
    summary["matched_reset_sensitivity"] = summarize(matched, include_interactions=False) if matched else None
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/"episodes.csv", episodes)
    write_csv(args.output/"paired-scenarios.csv", pairs)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(summary["status"], len(pairs), "paired scenarios;", args.output)


if __name__ == "__main__":
    main()
