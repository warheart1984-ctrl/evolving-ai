"""Registry of immutable runtime versions with optional persistence."""
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
        self._current_version: str = "v0"
        self._persist = persistence
        if persistence:
            self._load_from_persistence()

    def _load_from_persistence(self):
        """Load runtimes and current version from persistent store."""
        rows = self._persist.load_all("runtime")
        for rid, data in rows.items():
            try:
                self._runtimes[rid] = RuntimeManifest(**data)
            except Exception:
                pass
        meta = self._persist.load_all("meta")
        if "current_version" in meta:
            self._current_version = meta["current_version"].get("value", self._current_version)

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
        """Create a new immutable runtime version."""
        runtime_id = f"runtime-{version}"
        if runtime_id in self._runtimes:
            raise ValueError(f"Runtime {runtime_id} already exists; runtimes are immutable")

        parent_version = self._current_version if version != self._current_version else None

        manifest = RuntimeManifest(
            id=runtime_id,
            version=version,
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

    @staticmethod
    def _with_hash(manifest: RuntimeManifest) -> RuntimeManifest:
        data = manifest.model_dump(mode="json", exclude={"id", "manifest_hash"})
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return manifest.model_copy(update={"manifest_hash": hashlib.sha256(canonical.encode()).hexdigest()})

    def register_candidate(self, manifest: RuntimeManifest) -> RuntimeManifest:
        """Register an immutable SANDBOX manifest without changing current."""
        candidate = self._with_hash(manifest)
        if candidate.id in self._runtimes:
            raise ValueError(f"Runtime {candidate.id} already exists")
        self._runtimes[candidate.id] = candidate
        if self._persist:
            self._persist.save("runtime", candidate.id, candidate.model_dump(mode="json"))
        return candidate

    def get_runtime(self, runtime_id: str) -> Optional[RuntimeManifest]:
        """Get a runtime by ID."""
        return self._runtimes.get(runtime_id)

    def has_version(self, version: str) -> bool:
        """Check whether a runtime version exists."""
        return f"runtime-{version}" in self._runtimes

    def list_runtimes(self) -> List[RuntimeManifest]:
        """List all runtimes in chronological order."""
        return sorted(self._runtimes.values(), key=lambda r: int(r.version.lstrip("v")))

    def get_current(self) -> Optional[RuntimeManifest]:
        """Get the current (latest) runtime version."""
        return self._runtimes.get(f"runtime-{self._current_version}")

    def rollback_to(self, target_version: str) -> Optional[RuntimeManifest]:
        """Rollback the current pointer to a previous version."""
        runtime = self._runtimes.get(f"runtime-{target_version}")
        if runtime:
            self._current_version = target_version
            if self._persist:
                self._persist.save("meta", "current_version", {"value": target_version})
        return runtime

    def get_amendment_chain(self, runtime_id: str) -> List[RuntimeManifest]:
        """Get the lineage of runtimes leading to a runtime, oldest first."""
        chain = []
        current = self._runtimes.get(runtime_id)
        seen = set()
        while current and current.id not in seen:
            seen.add(current.id)
            chain.append(current)
            if current.parent_version:
                current = self._runtimes.get(f"runtime-{current.parent_version}")
            else:
                break
        return chain[::-1]
