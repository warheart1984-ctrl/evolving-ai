"""Registry of immutable runtime versions."""
from typing import Any, Dict, List, Optional

from app.governance.models import RuntimeManifest


class RuntimeRegistry:
    """Manages immutable runtime versions. Existing runtimes are never mutated."""

    def __init__(self):
        self._runtimes: Dict[str, RuntimeManifest] = {}
        self._current_version: str = "v0"

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

        self._runtimes[runtime_id] = manifest
        self._current_version = version
        return manifest

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
