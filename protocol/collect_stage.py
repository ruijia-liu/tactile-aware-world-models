"""Collect one stage, launching a separate simulator process for each route."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verify(report, rows, manifest_hash, model_hash, raw=None):
    summary = json.loads(report.read_bytes())
    if summary.get("status") != "PASSED" or not summary["interactions"]["passed"]:
        raise ValueError(f"Simulator did not complete the pair or branch set: {report}")
    entries = summary["interactions"]["episodes"]
    scenario = f"route{rows['route_index']:03d}_{rows['plug_material']}_{rows['pad_material']}"
    arms = rows["pair_action_order"]
    if model_hash:
        arms = [f"{method}_mean_{rows['visibility']}" for method in arms]
    if [e["episode_id"] for e in entries] != [f"{scenario}_{arm}" for arm in arms]:
        raise ValueError(f"Incorrect assignment or execution order in {report}")
    for e in entries:
        sidecar = (raw or report.with_suffix("") / "raw") / (e["episode_id"] + ".json")
        meta = json.loads(sidecar.read_bytes())
        with sidecar.with_suffix(".npz").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (digest != e["sha256"] or digest != meta["npz_sha256"] or meta["manifest_sha256"] != manifest_hash
                or meta["geometry_seed"] != rows["geometry_seed"]
                or (model_hash and meta["source_sha256"].get("ensemble.npz") != model_hash)):
            raise ValueError(f"Collection evidence differs from the stage inputs: {sidecar}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("branches", "validation", "confirmation"))
    parser.add_argument("--isaac-python", required=True, type=Path, help="Isaac Sim Python launcher or interpreter")
    parser.add_argument("--repo", required=True, type=Path, help="Upstream FlexiTac source checkout")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--usd-manifest", type=Path, required=True, help="Verified converted FlexiTac asset manifest")
    parser.add_argument("--rows", nargs="+", type=int, help="Zero-based assignments to collect; default: all")
    args = parser.parse_args()
    if (args.ensemble is not None) != (args.stage != "branches") or (args.freeze is not None) != (args.stage == "confirmation"):
        parser.error("Branches need no weights, validation needs weights, and confirmation needs weights and a freeze")
    filename = "manifest.json" if args.stage == "confirmation" else args.stage + ".json"
    manifest = (args.protocol / filename).resolve()
    raw = manifest.read_bytes()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    records = json.loads(raw)
    rows = records["trials"] if args.stage == "confirmation" else records
    ids = range(len(rows)) if args.rows is None else args.rows
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(rows) for i in ids):
        parser.error("Provide distinct row indices inside this stage")
    model_hash = hashlib.sha256(args.ensemble.read_bytes()).hexdigest() if args.ensemble else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempts = output.with_name(output.name + "-attempts")
    attempts.mkdir(parents=True, exist_ok=True)
    for i in ids:
        slot = f"route{rows[i]['route_index']}" if args.stage == "confirmation" else f"row-{i:02d}"
        published = output / (slot + ".json")
        if published.exists():
            verify(published, rows[i], manifest_hash, model_hash)
            continue
        # The raw folder moves first. A verified JSON sidecar marks the complete route.
        destination = published.with_suffix("")
        if destination.exists():
            # Recover an interruption between the two renames, using the same verified report.
            for prior in attempts.glob(f"attempt-{slot}-*/{published.name}"):
                if not prior.with_suffix("").exists():
                    verify(prior, rows[i], manifest_hash, model_hash, destination / "raw")
                    os.rename(prior, published)
                    break
            if published.exists():
                continue
            raise RuntimeError(f"A raw directory at {destination} has no corresponding completed report")
        parent = Path(tempfile.mkdtemp(prefix="attempt-" + slot + "-", dir=attempts))
        report = parent / published.name
        cmd = [str(args.isaac_python.resolve()), "-m", "simulation.collect", "--stage", args.stage,
               "--manifest", str(manifest), "--row", str(i), "--repo", str(args.repo.resolve()),
               "--report", str(report)]
        for flag, path in (("--ensemble", args.ensemble), ("--freeze", args.freeze), ("--usd-manifest", args.usd_manifest)):
            if path is not None:
                cmd.extend([flag, str(path.resolve())])
        print(f"Collecting {args.stage} row {i}", flush=True)
        outcome = subprocess.run(cmd, cwd=ROOT)
        if outcome.returncode and not report.exists():
            raise RuntimeError(f"Simulator exited with code {outcome.returncode} for {slot}. See {parent}")
        verify(report, rows[i], manifest_hash, model_hash)
        if destination.exists():
            raise RuntimeError(f"An unpublished raw directory exists at {destination}; inspect it before retrying")
        os.rename(report.with_suffix(""), destination)
        os.rename(report, published)
        # Additional simulator diagnostics, if produced, stay with the attempt for inspection.
        if not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    main()
