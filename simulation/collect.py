"""Execute one assigned route in a fresh FlexiTac / Isaac Sim 6 process."""
import argparse
import json
from pathlib import Path
import time

from protocol.locking import check_freeze, digest
from .collector import CONFIG, collect, write_json
from .runtime import native_scene


def assignment(stage, manifest, row_index, weights, freeze):
    data = json.loads(manifest.read_bytes())
    if stage == "confirmation":
        if weights is None or freeze is None:
            raise ValueError("Confirmation requires --ensemble and --freeze")
        frozen = check_freeze(freeze, weights, CONFIG / "design.json", manifest)
        rows = data["trials"]
        expected_indices = range(1000, 1100)
    else:
        if freeze is not None or (weights is None) != (stage == "branches"):
            raise ValueError("Branches use fixed actions; validation requires --ensemble")
        rows = data
        expected_indices = range(100, 124) if stage == "branches" else range(400, 408)
    if (not isinstance(rows, list) or len(rows) != len(expected_indices)
            or sorted(r["route_index"] for r in rows) != list(expected_indices)):
        raise ValueError(f"Unexpected {stage} cohort in assignment manifest")
    if not 0 <= row_index < len(rows):
        raise ValueError(f"Row index must be in [0, {len(rows)-1}]")
    row = dict(rows[row_index], manifest_sha256=digest(manifest))
    if stage == "confirmation":
        row["hold_seconds"] = frozen["control_rule"]["hold_seconds"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("branches", "validation", "confirmation"), required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Stage assignment file from protocol.generate")
    parser.add_argument("--row", type=int, required=True, help="Zero-based row index in the manifest")
    parser.add_argument("--repo", type=Path, required=True, help="Pinned upstream FlexiTac checkout")
    parser.add_argument("--usd-manifest", type=Path, required=True, help="Asset checksums from simulation.prepare_assets")
    parser.add_argument("--report", type=Path, required=True,
                        help="New route report, e.g. row-00.json. Raw episodes go into row-00/raw/")
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    row = assignment(args.stage, args.manifest, args.row, args.ensemble, args.freeze)
    path = args.report.resolve()
    if path.suffix != ".json" or path.with_suffix("").exists():
        parser.error("Use a new .json report basename with no existing route directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "STARTING", "stage": args.stage, "route_index": row["route_index"]}
    with path.open("x", encoding="utf8") as f:
        json.dump(report, f)

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(path, report)

    try:
        with native_scene(args.repo, report, save, usd_manifest=args.usd_manifest) as (env, cfg, app, check):
            report["status"] = "COLLECTING"
            save()
            collect(env, cfg, app, report, save, check, repo=args.repo.resolve(), row=row,
                    output=path.with_suffix(""), ensemble_file=args.ensemble,
                    protect_transport_hold=args.stage != "branches")
            if not report["interactions"]["passed"]:
                raise RuntimeError("Incomplete assigned route")
            report["status"] = "PASSED"
            save()
            print(f"Completed {len(report['interactions']['episodes'])} physical trials in {path}", flush=True)
    except Exception as error:
        report.update(status="FAILED", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
