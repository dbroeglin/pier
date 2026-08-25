"""Tests for VAL-003 (BLK-1: context_overlay_id) and VAL-004 (BLK-2: pier_trial_id).

Covers:
- Round-trip serialization of context_overlay_id through TrialConfig and TrialResult
- Backward-compatibility: legacy JSON without context_overlay_id loads as None
- Metadata mirroring: present overlay_id copied into AgentContext.metadata
- Metadata merge: existing metadata keys are preserved during mirroring
- Root-only provenance: pier_trial_id stamped on root trajectory, not subagents
- Per-trial bijection: each TrialResult receives a unique UUID
"""

import json
from pathlib import Path
from uuid import UUID

import pytest

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
from pier.trial.trial import _mirror_overlay_to_metadata, _stamp_trajectory_trial_id
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
