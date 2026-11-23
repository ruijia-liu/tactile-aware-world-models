"""Verify the provenance of a development-validated model before a held-out run."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL_RULE = dict(protect_transport_hold=True, hold_seconds=.35,
                    eligible_openings_during_transport_hold=["tight", "hold"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    # Normalize text line endings across Windows and POSIX checkouts.
    sources = [ROOT / "simulation" / name for name in ("collector.py", "planner.py")]
    return {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources}


def check_freeze(freeze_path, ensemble_path, design_path, manifest_path=None):
    frozen = json.loads(Path(freeze_path).read_bytes())
    design = json.loads(Path(design_path).read_bytes())
    valid = (frozen.get("status") == "PASSED"
             and frozen.get("ensemble_sha256") == digest(ensemble_path)
             and frozen.get("design_sha256") == digest(design_path)
             and frozen.get("controller_source_sha256") == source_hashes()
             and frozen.get("control_rule") == CONTROL_RULE
             and frozen.get("success_criteria") == design["success"]
             and set(frozen.get("validation_route_indices", [])) == set(range(400, 408))
             and all(frozen.get("development_successes_by_method", {}).get(arm, 0) >= 1 for arm in ("blind", "touch")))
    if not valid:
        raise ValueError("Model, controller, success definition, or development gate differs from the freeze")
    if manifest_path is not None:
        if frozen.get("confirmation_manifest_sha256") != digest(manifest_path):
            raise ValueError("Held-out assignment differs from the freeze")
        reserved = {r["route_index"] for r in json.loads(Path(manifest_path).read_bytes())["trials"]}
        dev = set(frozen["development_route_indices"])
        if reserved != set(range(1000, 1100)) or not set(range(400, 408)).issubset(dev) or dev & reserved:
            raise ValueError("Invalid or overlapping held-out cohorts")
    return frozen
