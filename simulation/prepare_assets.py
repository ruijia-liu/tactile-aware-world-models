"""Convert and verify the upstream URDF assets using Isaac Sim 5.1's native importer."""
import argparse
from pathlib import Path

from .runtime import native_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--usd-manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.usd_manifest.exists():
        parser.error("Choose an unused asset manifest path")
    with native_scene(args.repo, {}, lambda: None, usd_manifest=args.usd_manifest, prepare_assets=True):
        print(f"Verified converted assets and wrote {args.usd_manifest}", flush=True)


if __name__ == "__main__":
    main()
