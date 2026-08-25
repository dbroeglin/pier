"""Tests for VAL-003 (BLK-1: context_overlay_id) and VAL-004 (BLK-2: pier_trial_id).

Covers:
- Round-trip serialization of context_overlay_id through TrialConfig and TrialResult
- Backward-compatibility: legacy JSON without context_overlay_id loads as None
- Metadata mirroring: present overlay_id copied into AgentContext.metadata
- Metadata merge: existing metadata keys are preserved during mirroring
- Root-only provenance: pier_trial_id stamped on root trajectory, not subagents
- Per-trial bijection: each TrialResult receives a unique UUID
- _maybe_populate_agent_context integration boundary (UUID/overlay/metadata/sequencing)
- _stamp_trajectory_trial_id write-failure emits observable warning
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pytest

from pier.agents.installed.base import BaseInstalledAgent
from pier.models.agent.context import AgentContext
from pier.models.job.lock import TrialLock, _build_lock_trial
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from pier.models.trial.result import AgentInfo, TrialResult
from pier.trial.trial import (
    Trial,
    _mirror_overlay_to_metadata,
    _stamp_trajectory_trial_id,
)
from pier.utils.trajectory_utils import format_trajectory_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_config(task_dir: Path) -> TaskConfig:
    return TaskConfig(path=task_dir)


def _trial_config(
    task_dir: Path,
    *,
    context_overlay_id: str | None = None,
) -> TrialConfig:
    return TrialConfig(
        task=_task_config(task_dir),
        context_overlay_id=context_overlay_id,
    )


def _trial_result(config: TrialConfig, task_dir: Path) -> TrialResult:
    return TrialResult(
        task_name=task_dir.name,
        trial_name=config.trial_name,
        trial_uri=(task_dir / config.trial_name).resolve().as_uri(),
        task_id=LocalTaskId(path=task_dir),
        task_checksum="deadbeef" * 8,
        config=config,
        agent_info=AgentInfo(name="copilot-cli", version="1.0"),
    )


def _minimal_trajectory_dict(*, with_subagents: bool = False) -> dict:
    """Produce a raw dict that validates against the Trajectory schema."""
    root: dict = {
        "schema_version": "ATIF-v1.7",
        "session_id": "test-session",
        "agent": {"name": "copilot-cli", "version": "1.0"},
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Hello",
            }
        ],
    }
    if with_subagents:
        root["subagent_trajectories"] = [
            {
                "schema_version": "ATIF-v1.7",
                "trajectory_id": "subagent-abc",
                "agent": {"name": "copilot-cli:subagent", "version": "1.0"},
                "steps": [
                    {
                        "step_id": 1,
                        "source": "user",
                        "message": "Subagent task",
                    }
                ],
            }
        ]
    return root


# ---------------------------------------------------------------------------
# VAL-003: context_overlay_id field — round-trip and backward-compatibility
# ---------------------------------------------------------------------------


def test_trial_config_default_overlay_id_is_none(tmp_path: Path) -> None:
    config = _trial_config(tmp_path)
    assert config.context_overlay_id is None


def test_trial_config_context_overlay_id_round_trips(tmp_path: Path) -> None:
    config = _trial_config(tmp_path, context_overlay_id="baseline-v1")
    data = json.loads(config.model_dump_json())
    restored = TrialConfig.model_validate(data)
    assert restored.context_overlay_id == "baseline-v1"


def test_trial_config_none_overlay_id_round_trips(tmp_path: Path) -> None:
    config = _trial_config(tmp_path, context_overlay_id=None)
    data = json.loads(config.model_dump_json())
    restored = TrialConfig.model_validate(data)
    assert restored.context_overlay_id is None


def test_trial_config_legacy_json_loads_without_context_overlay_id(
    tmp_path: Path,
) -> None:
    config = _trial_config(tmp_path)
    data = json.loads(config.model_dump_json())
    # Simulate legacy JSON that predates the field
    data.pop("context_overlay_id", None)
    assert "context_overlay_id" not in data
    restored = TrialConfig.model_validate(data)
    assert restored.context_overlay_id is None


def test_trial_result_config_round_trips_overlay_id(tmp_path: Path) -> None:
    config = _trial_config(tmp_path, context_overlay_id="overlay-A-v2")
    result = _trial_result(config, tmp_path)
    data = json.loads(result.model_dump_json())
    restored = TrialResult.model_validate(data)
    assert restored.config.context_overlay_id == "overlay-A-v2"


def test_trial_lock_context_overlay_id_round_trips(tmp_path: Path) -> None:
    config = _trial_config(tmp_path, context_overlay_id="treatment-v3")
    lock = _build_lock_trial(config)
    assert lock.context_overlay_id == "treatment-v3"
    data = json.loads(lock.model_dump_json())
    restored = TrialLock.model_validate(data)
    assert restored.context_overlay_id == "treatment-v3"


def test_trial_lock_legacy_json_without_overlay_id(tmp_path: Path) -> None:
    config = _trial_config(tmp_path)
    lock = _build_lock_trial(config)
    data = json.loads(lock.model_dump_json())
    data.pop("context_overlay_id", None)
    restored = TrialLock.model_validate(data)
    assert restored.context_overlay_id is None


# ---------------------------------------------------------------------------
# VAL-003: metadata mirroring
# ---------------------------------------------------------------------------


def test_mirror_overlay_id_adds_to_empty_metadata() -> None:
    ctx = AgentContext()
    _mirror_overlay_to_metadata(ctx, "baseline-v1")
    assert ctx.metadata == {"context_overlay_id": "baseline-v1"}


def test_mirror_overlay_id_preserves_existing_metadata_keys() -> None:
    ctx = AgentContext(metadata={"copilot_aiu": 1.5, "other_key": "value"})
    _mirror_overlay_to_metadata(ctx, "treatment-v1")
    assert ctx.metadata is not None
    assert ctx.metadata["copilot_aiu"] == 1.5
    assert ctx.metadata["other_key"] == "value"
    assert ctx.metadata["context_overlay_id"] == "treatment-v1"


def test_mirror_overlay_id_noop_when_none_leaves_metadata_unchanged() -> None:
    ctx = AgentContext(metadata={"existing": "data"})
    _mirror_overlay_to_metadata(ctx, None)
    assert ctx.metadata == {"existing": "data"}


def test_mirror_overlay_id_noop_when_none_and_no_metadata() -> None:
    ctx = AgentContext()
    _mirror_overlay_to_metadata(ctx, None)
    assert ctx.metadata is None


# ---------------------------------------------------------------------------
# VAL-004: pier_trial_id stamping — root-only provenance
# ---------------------------------------------------------------------------


def test_stamp_trajectory_trial_id_injects_into_extra(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    data = _minimal_trajectory_dict()
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")

    _stamp_trajectory_trial_id(trajectory_path, "00000000-1111-2222-3333-444444444444")

    result = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert result["extra"]["pier_trial_id"] == "00000000-1111-2222-3333-444444444444"


def test_stamp_trajectory_trial_id_preserves_existing_extra_keys(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    data = _minimal_trajectory_dict()
    data["extra"] = {"copilot_aiu": 2.5, "summarization_count": 0}
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")

    _stamp_trajectory_trial_id(trajectory_path, "test-uuid")

    result = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert result["extra"]["copilot_aiu"] == 2.5
    assert result["extra"]["summarization_count"] == 0
    assert result["extra"]["pier_trial_id"] == "test-uuid"


def test_stamp_trajectory_trial_id_root_only_leaves_subagents_unstamped(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    data = _minimal_trajectory_dict(with_subagents=True)
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")

    _stamp_trajectory_trial_id(trajectory_path, "root-trial-uuid")

    result = json.loads(trajectory_path.read_text(encoding="utf-8"))
    # Root trajectory has pier_trial_id
    assert result["extra"]["pier_trial_id"] == "root-trial-uuid"
    # Subagent trajectories do NOT carry pier_trial_id
    for sub in result.get("subagent_trajectories") or []:
        assert "pier_trial_id" not in (sub.get("extra") or {})


def test_stamp_trajectory_trial_id_noop_when_file_absent(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "nonexistent.json"
    # Must not raise; file must remain absent
    _stamp_trajectory_trial_id(trajectory_path, "some-uuid")
    assert not trajectory_path.exists()


def test_stamp_trajectory_trial_id_overwrites_previous_id(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    data = _minimal_trajectory_dict()
    data["extra"] = {"pier_trial_id": "old-uuid"}
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")

    _stamp_trajectory_trial_id(trajectory_path, "new-uuid")

    result = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert result["extra"]["pier_trial_id"] == "new-uuid"


# ---------------------------------------------------------------------------
# Per-trial bijection: each TrialResult receives a distinct UUID
# ---------------------------------------------------------------------------


def test_trial_result_ids_are_distinct_per_instance(tmp_path: Path) -> None:
    config = _trial_config(tmp_path)
    r1 = _trial_result(config, tmp_path)
    r2 = _trial_result(config, tmp_path)
    assert r1.id != r2.id


def test_trial_result_id_is_valid_uuid(tmp_path: Path) -> None:
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    assert isinstance(result.id, UUID)
    # Verify round-trip through string (canonical form used as pier_trial_id)
    assert str(result.id) == str(UUID(str(result.id)))


def test_ten_trial_results_all_have_unique_ids(tmp_path: Path) -> None:
    config = _trial_config(tmp_path)
    ids = {_trial_result(config, tmp_path).id for _ in range(10)}
    assert len(ids) == 10


# ---------------------------------------------------------------------------
# _maybe_populate_agent_context integration boundary (VAL-003/VAL-004)
# ---------------------------------------------------------------------------


class _StubInstalledAgent(BaseInstalledAgent):
    """Minimal concrete BaseInstalledAgent for boundary testing."""

    @staticmethod
    def name() -> str:
        return "stub"

    def version(self) -> str | None:
        return "0.0"

    async def setup(self, environment) -> None:  # type: ignore[override]
        pass

    async def run(self, instruction, environment, context) -> None:  # type: ignore[override]
        pass

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    def install_spec(self):  # type: ignore[override]
        return None


def _stub_agent(tmp_path: Path) -> _StubInstalledAgent:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return _StubInstalledAgent(logs_dir=logs_dir)


def _fake_trial_for_populate(
    tmp_path: Path,
    *,
    result: TrialResult | None,
    context_overlay_id: str | None = None,
) -> SimpleNamespace:
    """Build a minimal SimpleNamespace that satisfies _maybe_populate_agent_context."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        _agent=_stub_agent(tmp_path),
        _result=result,
        _trial_paths=SimpleNamespace(agent_dir=agent_dir),
        config=SimpleNamespace(context_overlay_id=context_overlay_id),
    )


def _write_trajectory(agent_dir: Path, data: dict) -> Path:
    trajectory_path = agent_dir / "trajectory.json"
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")
    return trajectory_path


def test_maybe_populate_stamps_exact_result_id_into_trajectory(
    tmp_path: Path,
) -> None:
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(tmp_path, result=result)

    data = _minimal_trajectory_dict()
    _write_trajectory(trial._trial_paths.agent_dir, data)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    stamped = json.loads(
        (trial._trial_paths.agent_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    stamped_id = stamped["extra"]["pier_trial_id"]
    assert stamped_id == str(result.id)
    assert UUID(stamped_id) == result.id


def test_maybe_populate_mirrors_context_overlay_id_when_present(
    tmp_path: Path,
) -> None:
    config = _trial_config(tmp_path, context_overlay_id="treatment-v2")
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(
        tmp_path, result=result, context_overlay_id="treatment-v2"
    )

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert agent_result.metadata is not None
    assert agent_result.metadata["context_overlay_id"] == "treatment-v2"


def test_maybe_populate_no_overlay_key_when_context_overlay_id_is_none(
    tmp_path: Path,
) -> None:
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(tmp_path, result=result, context_overlay_id=None)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert agent_result.metadata is None or "context_overlay_id" not in (
        agent_result.metadata or {}
    )


def test_maybe_populate_preserves_unrelated_metadata(tmp_path: Path) -> None:
    config = _trial_config(tmp_path, context_overlay_id="ov-1")
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(
        tmp_path, result=result, context_overlay_id="ov-1"
    )

    agent_result = AgentContext(metadata={"copilot_aiu": 3.7, "other": "kept"})
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert agent_result.metadata is not None
    assert agent_result.metadata["copilot_aiu"] == 3.7
    assert agent_result.metadata["other"] == "kept"
    assert agent_result.metadata["context_overlay_id"] == "ov-1"


def test_maybe_populate_stamp_occurs_before_step_count_reads_trajectory(
    tmp_path: Path,
) -> None:
    """Stamping writes pier_trial_id before step-count reads the same file."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(tmp_path, result=result)

    data = _minimal_trajectory_dict()
    data["steps"] = [
        {"step_id": 1, "source": "user", "message": "hi"},
        {"step_id": 2, "source": "agent", "message": "done"},
    ]
    _write_trajectory(trial._trial_paths.agent_dir, data)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    stamped = json.loads(
        (trial._trial_paths.agent_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    # Stamped before step-count read, so both are set from the same file content
    assert "pier_trial_id" in stamped["extra"]
    assert agent_result.n_agent_steps == 1  # one agent step in the trajectory


def test_maybe_populate_result_none_guard_skips_stamping(tmp_path: Path) -> None:
    """When _result is None the stamp must not be attempted and no error raised."""
    config = _trial_config(tmp_path)
    trial = _fake_trial_for_populate(tmp_path, result=None)

    data = _minimal_trajectory_dict()
    traj_path = _write_trajectory(trial._trial_paths.agent_dir, data)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    after = json.loads(traj_path.read_text(encoding="utf-8"))
    assert "pier_trial_id" not in (after.get("extra") or {})


def test_maybe_populate_agent_result_none_guard_returns_early(
    tmp_path: Path,
) -> None:
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(tmp_path, result=result)

    # Must not raise even with no trajectory file present
    Trial._maybe_populate_agent_context(trial, None)


def test_maybe_populate_non_installed_agent_guard_returns_early(
    tmp_path: Path,
) -> None:
    """When _agent is not a BaseInstalledAgent the method must be a no-op."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(tmp_path, result=result)
    trial._agent = object()  # not a BaseInstalledAgent

    data = _minimal_trajectory_dict()
    traj_path = _write_trajectory(trial._trial_paths.agent_dir, data)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    after = json.loads(traj_path.read_text(encoding="utf-8"))
    assert "pier_trial_id" not in (after.get("extra") or {})
    assert agent_result.metadata is None


def test_maybe_populate_called_after_agent_timeout_still_stamps(
    tmp_path: Path,
) -> None:
    """Simulates the failure/timeout path: _result is already set, agent_result populated."""
    config = _trial_config(tmp_path, context_overlay_id="baseline")
    result = _trial_result(config, tmp_path)
    trial = _fake_trial_for_populate(
        tmp_path, result=result, context_overlay_id="baseline"
    )

    data = _minimal_trajectory_dict()
    _write_trajectory(trial._trial_paths.agent_dir, data)

    # Simulate partial context collected before timeout
    agent_result = AgentContext(n_input_tokens=500, n_output_tokens=100)
    Trial._maybe_populate_agent_context(trial, agent_result)

    stamped = json.loads(
        (trial._trial_paths.agent_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    assert stamped["extra"]["pier_trial_id"] == str(result.id)
    assert agent_result.metadata is not None
    assert agent_result.metadata["context_overlay_id"] == "baseline"
    # Existing token counts survive
    assert agent_result.n_input_tokens == 500
    assert agent_result.n_output_tokens == 100


# ---------------------------------------------------------------------------
# Write-failure: observable warning (not silent pass)
# ---------------------------------------------------------------------------


def test_stamp_trajectory_trial_id_write_failure_emits_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A write OSError must produce a structured warning, not a silent pass."""
    trajectory_path = tmp_path / "trajectory.json"
    data = _minimal_trajectory_dict()
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="pier"):
        with patch.object(
            Path,
            "write_text",
            side_effect=OSError("disk full"),
        ):
            _stamp_trajectory_trial_id(trajectory_path, "fail-uuid")

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("pier_trial_id" in m or str(trajectory_path) in m for m in warning_messages), (
        f"Expected a warning mentioning the path or pier_trial_id; got: {warning_messages}"
    )


def test_stamp_trajectory_trial_id_write_failure_does_not_raise(
    tmp_path: Path,
) -> None:
    """A write OSError must not propagate — the trial must remain runnable."""
    trajectory_path = tmp_path / "trajectory.json"
    data = _minimal_trajectory_dict()
    trajectory_path.write_text(format_trajectory_json(data), encoding="utf-8")

    with patch.object(Path, "write_text", side_effect=OSError("read-only fs")):
        # Must not raise
        _stamp_trajectory_trial_id(trajectory_path, "fail-uuid")
