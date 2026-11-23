"""Strict, optional reuse of assets converted by a successful FlexiTac smoke run.

Only the converter alias used by spawn_from_urdf is replaced. Its caller still
passes the original UrdfFileCfg to _spawn_from_usd_file, preserving all physics,
articulation, contact, material, scale and pose settings. Native URDF bindings
are never acquired in cached mode. No upstream source files are modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(repo, path):
    return Path(path).resolve().relative_to(repo).as_posix()


def _resolve(repo, relative):
    path = (repo / relative).resolve()
    path.relative_to(repo)  # Reject traversal and symlink escapes.
    return path


def _config(cfg, repo):
    """Keep every config field except the intentional force-conversion bypass."""
    def normalize(value):
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            # A manifest may move together with its repository. Preserve the
            # exact relative path, not machine-specific absolute prefixes.
            if value == str(repo):
                return "@repo"
            prefix = str(repo) + "/"
            native_prefix = str(repo) + "\\"
            if value.startswith(prefix) or value.startswith(native_prefix):
                return "@repo/" + _relative(repo, value)
        return value
    data = dict(cfg.to_dict())
    data.pop("force_usd_conversion", None)
    data = normalize(data)
    # Reject unsupported/nonfinite values rather than silently losing fields.
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return data, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _snapshot(repo, directories):
    files = {}
    for relative in sorted(set(directories)):
        directory = _resolve(repo, relative)
        if not directory.is_dir():
            raise RuntimeError(f"Missing asset directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name not in {".asset_hash", "config.yaml"}:
                files[_relative(repo, path)] = _sha256(path)
    if not files:
        raise RuntimeError("Empty asset manifest")
    return files


class ConversionRecorder:
    """Record actual converter inputs and outputs before creating the scene."""

    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.entries = []
        self.module = importlib.import_module("isaaclab.sim.converters")
        self.original = self.module.UrdfConverter
        if getattr(self.original, "_flexitac_cache_adapter", False):
            raise RuntimeError("Cannot record fresh conversions in cached mode")

        def recording_converter(cfg):
            configuration, signature = _config(cfg, self.repo)
            loader = self.original(cfg)
            self.entries.append({
                "asset_path": _relative(self.repo, cfg.asset_path),
                "asset_sha256": _sha256(cfg.asset_path),
                "usd_path": _relative(self.repo, loader.usd_path),
                "config": configuration,
                "config_sha256": signature,
                "force_usd_conversion_requested": bool(cfg.force_usd_conversion),
            })
            return loader

        self.wrapper = recording_converter
        self.module.UrdfConverter = recording_converter

    def restore(self):
        if self.module.UrdfConverter is self.wrapper:
            self.module.UrdfConverter = self.original

    def write_manifest(self, path, *, simulator_version, repo_commit):
        """Call only after scene initialization AND all smoke checks succeeded."""
        if len(self.entries) != 3 or len({e["asset_path"] for e in self.entries}) != 3:
            raise RuntimeError("Expected exactly three distinct converted FlexiTac URDF assets")
        directories = ["Isaacsim_tactile_env/assets"]
        directories.extend(str(Path(e["usd_path"]).parent).replace("\\", "/") for e in self.entries)
        hashes = _snapshot(self.repo, directories)
        for entry in self.entries:
            entry["usd_sha256"] = hashes[entry["usd_path"]]
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_simulator_version": str(simulator_version),
            "source_repo_commit": repo_commit,
            "smoke_checks_passed_before_shutdown": True,
            "entries": self.entries,
            "directories": sorted(set(directories)),
            "files_sha256": hashes,
            "ignored_metadata": [".asset_hash", "config.yaml"],
        }
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)
        return {"manifest": str(path), "entries": self.entries,
                "hashed_files": len(hashes), "source_simulator_version": str(simulator_version)}


def apply_cached_usd_adapter(repo, manifest_path):
    """Validate cache and install a converter alias; return JSON-ready evidence.

    Unknown URDFs and any source/config/output changes fail closed. The sole
    ignored config field is force_usd_conversion, intentionally overridden by
    explicit cached mode. Restart the process to restore normal conversion.
    """
    repo = Path(repo).resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("smoke_checks_passed_before_shutdown"):
        raise RuntimeError("Unsupported or unverified USD manifest")
    entries = manifest["entries"]
    if len(entries) != 3 or len({e["asset_path"] for e in entries}) != 3:
        raise RuntimeError("USD manifest must contain three distinct FlexiTac assets")
    if _snapshot(repo, manifest["directories"]) != manifest["files_sha256"]:
        raise RuntimeError("USD manifest source/generated file set or SHA256 mismatch")
    by_asset = {}
    for entry in entries:
        if _sha256(_resolve(repo, entry["asset_path"])) != entry["asset_sha256"]:
            raise RuntimeError("URDF source SHA256 mismatch")
        if _sha256(_resolve(repo, entry["usd_path"])) != entry["usd_sha256"]:
            raise RuntimeError("Converted USD SHA256 mismatch")
        by_asset[entry["asset_path"]] = entry
    module = importlib.import_module("isaaclab.sim.converters")
    if getattr(module.UrdfConverter, "_flexitac_cache_adapter", False):
        raise RuntimeError("A cached USD adapter is already installed")

    class CachedUrdfConverter:
        _flexitac_cache_adapter = True

        def __init__(self, cfg):
            relative = _relative(repo, cfg.asset_path)
            if relative not in by_asset:
                raise RuntimeError(f"URDF is not in the verified cache: {relative}")
            entry = by_asset[relative]
            configuration, signature = _config(cfg, repo)
            if signature != entry["config_sha256"] or configuration != entry["config"]:
                raise RuntimeError(f"Cached URDF conversion/spawn config mismatch: {relative}")
            self.cfg = cfg
            self.usd_path = str(_resolve(repo, entry["usd_path"]))
            self.usd_dir = str(Path(self.usd_path).parent)
            self.usd_file_name = Path(self.usd_path).name
            # Catch changes between installation and individual asset spawn.
            if _sha256(cfg.asset_path) != entry["asset_sha256"] or _sha256(self.usd_path) != entry["usd_sha256"]:
                raise RuntimeError(f"Asset changed after cache validation: {relative}")
            print(f"FLEXITAC_CACHED_USD {relative} -> {self.usd_path}", flush=True)

    module.UrdfConverter = CachedUrdfConverter
    return {"manifest": str(manifest_path), "entries": entries,
            "hashed_files": len(manifest["files_sha256"]),
            "source_simulator_version": manifest["source_simulator_version"],
            "source_repo_commit": manifest["source_repo_commit"],
            "force_usd_conversion_overridden": True}
