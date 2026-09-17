"""P2 constitution as root of trust: hash-pinned boot, no runtime writes."""
import hashlib

import pytest

from constitution.constitution import Constitution, ConstitutionIntegrityError


def _sha256_of_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class TestHashPinAtBoot:
    def test_matching_pin_loads_constitution(self, tmp_path):
        """Boot with a correct pin file succeeds."""
        constitution_path = tmp_path / "constitution.yaml"
        constitution_path.write_text("version: v1\n", encoding="utf-8")
        pin = tmp_path / "constitution.yaml.sha256"
        pin.write_text(_sha256_of_file(str(constitution_path)), encoding="utf-8")

        c = Constitution.from_file(str(constitution_path), pin_path=str(pin))
        assert c.version == "v1"

    def test_tampered_constitution_refuses_boot(self, tmp_path):
        """A hash mismatch must refuse startup with ConstitutionIntegrityError."""
        constitution_path = tmp_path / "constitution.yaml"
        constitution_path.write_text("version: v1\n", encoding="utf-8")
        pin = tmp_path / "constitution.yaml.sha256"
        pin.write_text("0" * 64, encoding="utf-8")  # wrong pin

        with pytest.raises(ConstitutionIntegrityError):
            Constitution.from_file(str(constitution_path), pin_path=str(pin))

    def test_missing_pin_file_refuses_boot(self, tmp_path):
        """No pin file means integrity can't be verified — refuse to boot."""
        constitution_path = tmp_path / "constitution.yaml"
        constitution_path.write_text("version: v1\n", encoding="utf-8")

        with pytest.raises(ConstitutionIntegrityError):
            Constitution.from_file(str(constitution_path), pin_path=str(tmp_path / "nope.sha256"))

    def test_real_repo_pin_matches_yaml(self):
        """The committed pin must match the committed constitution.yaml."""
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(root, "constitution", "constitution.yaml")
        pin_path = os.path.join(root, "constitution", "constitution.yaml.sha256")
        c = Constitution.from_file(yaml_path, pin_path=pin_path)
        assert c.content_hash == _sha256_of_file(yaml_path)


class TestNoRuntimeWrite:
    def test_constitution_is_frozen_in_memory(self):
        """The in-memory Constitution cannot be mutated by any code path."""
        c = Constitution()
        with pytest.raises(Exception):
            c.version = "v9-rogue"
        with pytest.raises(Exception):
            c.promotion_gates = {}
        assert c.version == "v1"

    def test_runtime_addresses_constitution_by_version_only(self, tmp_path):
        """Promotions never modify or re-load constitution.yaml from the process."""
        import os
        from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
        from app.governance.models import Evaluation, Evidence

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(root, "constitution", "constitution.yaml")
        pin_path = yaml_path + ".sha256"
        before_hash = _sha256_of_file(yaml_path)

        constitution = Constitution.from_file(yaml_path, pin_path=pin_path)
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m",
                                constitution_version=constitution.version,
                                created_by="system", description="v0")
        governor = Governor(registry, constitution)

        evaluation = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0", candidate_runtime="v1",
            correctness=0.95, instruction_following=0.92, robustness=0.9, safety=1.0,
            regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="r", runtime_version="v1")],
        )
        amendment = Amendment(
            id="prop-1", parent_version="v0", target=TargetType.PROMPT,
            description="Improve prompt", rationale="Regression",
            proposed_diff={"prompts": {"system": "new"}},
            proposer="steward", reviewer="human",
            evaluation=evaluation, status=AmendmentStatus.REVIEW,
        )
        # P5: bind approval to the exact candidate manifest being evaluated.
        amendment.evaluation.candidate_manifest_hash = governor.build_candidate(amendment).manifest_hash
        result = governor.approve_amendment(amendment, evidence_ids=["ev-1"])
        assert result.success is True

        after_hash = _sha256_of_file(yaml_path)
        assert after_hash == before_hash, "constitution.yaml was written by the runtime"

        # Promoted runtime references the SAME constitution version; pin still verifies.
        new_runtime = registry.get_runtime(result.new_runtime_id)
        assert new_runtime.constitution_version == constitution.version
        assert _sha256_of_file(yaml_path) == before_hash