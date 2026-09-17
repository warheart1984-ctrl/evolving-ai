"""Registry of immutable runtime versions with optional persistence.

Distinguishes two kinds of manifests:
- ``released``: promoted through the governance pipeline, addressable by a
  numeric ``release_version`` and eligible to be current.
- ``sandbox``: candidate materialized for evaluation; never current, never
  addressable as a release, and listed only as unreleased work.

Every manifest is immutable: nothing in the registry mutates a stored runtime.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional

from app.governance.models import RuntimeManifest


class RuntimeRegistry:
    """Manages immutable runtime versions. Existing runtimes are never mutated.

    Optionally persists to a StateStore so runtime history survives restarts.
    """

    def __init__(self, persistence=None):
        self._runtimes: Dict[str, RuntimeManifest] = {}
        self._current_version: Optional[str] = None
        self._persist = persistence
        if persistence:
            self._load_from_persistence()
        # v0 bootstrap happens via create_runtime; otherwise no runtime is set.
        if self._current_version is None and "runtime-v0" in self._runtimes:
            self._current_version = "v0"

    def _load_from_persistence(self):
        """Load runtimes and current version from persistent store."""
        rows = self._persist.load_all("runtime")
        for rid, data in rows.items():
            try:
                self._runtimes[rid] = RuntimeManifest(**data)
            except Exception:
                # Handled explicitly (see Phase 6): invalid governance records
                # are surfaced rather than swallowed silently.
                raise ValueError(f"Corrupt runtime record in persistence: {rid}")
        meta = self._persist.load_all("meta")
        if "current_version" in meta:
            self._current_version = meta["current_version"].get("value")

    # --- Helpers ---

    @staticmethod
    def _parse_release_version(version: str) -> Optional[int]:
        """Extract the numeric release version from 'v<N>' or '' for sandbox."""
        if version.startswith("v") and version[1:].isdigit():
            return int(version[1:])
        return None

    def _next_release_version(self) -> int:
        """Smallest positive integer larger than every released version."""
        existing = [
            r.release_version
            for r in self._runtimes.values()
            if r.kind == "released" and r.release_version is not None
        ]
        return (max(existing) + 1) if existing else 1

    # --- Creation ---

    def create_runtime(
        self,
        version: str,
        model_identifier: str,
        constitution_version: str,
        prompts: Dict[str, str] = None,
        tools: Dict[str, str] = None,
        memory: Dict[str, str] = None,
        evaluation: Dict[str, Any] = None,
        created_by: str = "system",
        description: str = "",
        amendment_id: str = None,
    ) -> RuntimeManifest:
        """Create a new *released* immutable runtime version.

        ``version`` must be a release-style ``v<N>`` identifier. The manifest
        is marked ``kind="released"`` and gets the numeric ``release_version``.
        """
        runtime_id = f"runtime-{version}"
        if runtime_id in self._runtimes:
            raise ValueError(f"Runtime {runtime_id} already exists; runtimes are immutable")

        release_version = self._parse_release_version(version)
        if release_version is None or release_version < 0:
            raise ValueError(
                f"Only release-style versions ('v0', 'v1', ...) can be created "
                f"as released runtimes; got '{version}'. Use register_candidate "
                f"for sandbox manifests."
            )

        parent_version = None
        if self.get_current() is not None and self.get_current().version != version:
            parent_version = self.get_current().version

        manifest = RuntimeManifest(
            id=runtime_id,
            version=version,
            release_version=release_version,
            kind="released",
            parent_version=parent_version,
            model_identifier=model_identifier,
            constitution_version=constitution_version,
            prompts=prompts or {},
            tools=tools or {},
            memory=memory or {},
            evaluation=evaluation or {},
            created_by=created_by,
            description=description,
            amendment_id=amendment_id,
        )
        manifest = self._with_hash(manifest)

        self._runtimes[runtime_id] = manifest
        self._current_version = version

        if self._persist:
            self._persist.save("runtime", runtime_id, manifest.model_dump(mode="json"))
            self._persist.save("meta", "current_version", {"value": version})

        return manifest

    def register_candidate(self, manifest: RuntimeManifest) -> RuntimeManifest:
        """Register an immutable SANDBOX manifest without changing current.

        The candidate's ``kind`` is forced to ``sandbox``; it is never
        addressable as a release and can never become current.
        """
        if manifest.kind != "sandbox":
            manifest = manifest.model_copy(update={"kind": "sandbox"})
        candidate = self._with_hash(manifest)

        duplicate = self._runtimes.get(candidate.id)
        if duplicate is not None:
            # Idempotent if it's the exact same immutable candidate.
            if duplicate.manifest_hash == candidate.manifest_hash:
                return duplicate
            raise ValueError(f"Runtime {candidate.id} already exists")

        self._runtimes[candidate.id] = candidate
        if self._persist:
            self._persist.save("runtime", candidate.id, candidate.model_dump(mode="json"))
        return candidate

    # --- Querying ---

    def get_runtime(self, runtime_id: str) -> Optional[RuntimeManifest]:
        """Get a runtime by ID (released or sandbox)."""
        return self._runtimes.get(runtime_id)

    def get_released(self, version: str) -> Optional[RuntimeManifest]:
        """Get a *released* runtime by version (ignores sandbox manifests)."""
        runtime = self._runtimes.get(f"runtime-{version}")
        if runtime is not None and runtime.kind == "released":
            return runtime
        return None

    def has_version(self, version: str) -> bool:
        """Check whether a *released* runtime version exists."""
        return (
            f"runtime-{version}" in self._runtimes
            and self._runtimes[f"runtime-{version}"].kind == "released"
        )

    def list_runtimes(self) -> List[RuntimeManifest]:
        """List all runtimes without crashing, released first then sandbox.

        Released runtimes are sorted by numeric release version; sandbox
        runtimes are sorted by creation time then parent version. Never sorts
        on version strings as integers (which crashes on candidate version
        labels such as 'v0-candidate-x').
        """
        released = [r for r in self._runtimes.values() if r.kind == "released"]
        sandbox = [r for r in self._runtimes.values() if r.kind == "sandbox"]

        released.sort(
            key=lambda r: (r.release_version,)
            if r.release_version is not None
            else (1 << 30, r.version)
        )
        sandbox.sort(key=lambda r: (r.created_at.timestamp(), r.parent_version or "", r.version))
        return released + sandbox

    def list_released(self) -> List[RuntimeManifest]:
        """Released runtimes in numeric release order (current release first)."""
        released = [r for r in self.list_runtimes() if r.kind == "released"]
        released.sort(key=lambda r: -r.release_version if r.release_version is not None else 0)
        return released

    def list_candidates(self) -> List[RuntimeManifest]:
        """Sandbox candidates sorted by creation time."""
        return [r for r in self.list_runtimes() if r.kind == "sandbox"]

    def get_current(self) -> Optional[RuntimeManifest]:
        """Get the current (latest) *released* runtime version."""
        if self._current_version is None:
            released = [
                r for r in self._runtimes.values()
                if r.kind == "released" and r.release_version is not None
            ]
            if not released:
                return None
            latest = max(released, key=lambda r: r.release_version)
            self._current_version = latest.version
            return latest
        runtime = self._runtimes.get(f"runtime-{self._current_version}")
        if runtime is not None and runtime.kind == "released":
            return runtime
        # current_version pointed at a sandbox (shouldn't happen); fall through.
        released = [
            r for r in self._runtimes.values()
            if r.kind == "released" and r.release_version is not None
        ]
        if not released:
            return None
        latest = max(released, key=lambda r: r.release_version)
        self._current_version = latest.version
        return latest

    def rollback_to(self, target_version: str) -> Optional[RuntimeManifest]:
        """Rollback the current pointer to a previous *released* runtime version."""
        runtime = self.get_released(target_version)
        if runtime:
            self._current_version = target_version
            if self._persist:
                self._persist.save("meta", "current_version", {"value": target_version})
        return runtime

    # --- Hashing ---

    @staticmethod
    def _with_hash(manifest: RuntimeManifest) -> RuntimeManifest:
        # created_at is recorded metadata (not content): it is excluded so a
        # candidate re-materialized at approval time hashes identically (P5).
        data = manifest.model_dump(mode="json", exclude={"id", "manifest_hash", "created_at"})
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return manifest.model_copy(update={"manifest_hash": hashlib.sha256(canonical.encode()).hexdigest()})

    def get_amendment_chain(self, runtime_id: str) -> List[RuntimeManifest]:
        """Get the lineage of released runtimes leading to a runtime, oldest first."""
        chain = []
        current = self._runtimes.get(runtime_id) or self.get_released(runtime_id)
        seen = set()
        while current and current.id not in seen:
            seen.add(current.id)
            chain.append(current)
            if current.parent_version:
                current = self._runtimes.get(f"runtime-{current.parent_version}")
            else:
                break
        return chain[::-1]