"""Verify constitution/pin integrity and paired changes in CI."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION = ROOT / "constitution" / "constitution.yaml"
PIN = ROOT / "constitution" / "constitution.yaml.sha256"


def verify_hash() -> None:
    actual = hashlib.sha256(CONSTITUTION.read_bytes()).hexdigest()
    expected = PIN.read_text(encoding="utf-8").strip().lower()
    if actual != expected:
        raise SystemExit(
            f"constitution hash mismatch: pin={expected}, actual={actual}"
        )


def changed_files(base: str) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Git base ref used to enforce paired changes")
    args = parser.parse_args()

    verify_hash()
    if args.base:
        changed = changed_files(args.base)
        constitution_changed = "constitution/constitution.yaml" in changed
        pin_changed = "constitution/constitution.yaml.sha256" in changed
        if constitution_changed != pin_changed:
            raise SystemExit(
                "constitution.yaml and constitution.yaml.sha256 must change together"
            )
    print("constitution hash and paired-change checks passed")


if __name__ == "__main__":
    main()
