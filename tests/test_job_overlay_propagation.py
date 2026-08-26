"""Tests for Gate 4 baseline overlay propagation through JobConfig → TrialConfig.

Covers:
- JobConfig.context_overlay_id field: default None, set value
- Serialization round-trip for JobConfig with and without the field
- Legacy JobConfig JSON (field absent) loads as None
- Job._init_trial_configs propagates None overlay to every TrialConfig
- Job._init_trial_configs propagates a non-None overlay to every TrialConfig
- Multiple tasks × agents × attempts all receive the overlay
- JobLock / TrialLock records the propagated overlay deterministically
- Legacy lock JSON without context_overlay_id in TrialLock loads as None
- No loss at AgentContext mirroring boundary: overlay reaches metadata
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from pier.job import Job
from pier.models.agent.context import AgentContext
from pier.models.job.config import JobConfig
from pier.models.job.lock import JobLock, TrialLock, build_job_lock, _build_lock_trial
from pier.models.trial.config import AgentConfig, TaskConfig, TrialConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_config(task_dir: Path) -> TaskConfig:
    return TaskConfig(path=task_dir)


def _make_job(
    tmp_path: Path,
    *,
    context_overlay_id: str | None = None,
    n_attempts: int = 1,
    n_tasks: int = 1,
    n_agents: int = 1,
) -> Job:
    """Build a Job with in-memory task configs, no filesystem side-effects beyond job dir."""
    task_dirs = [tmp_path / f"task{i}" for i in range(n_tasks)]
    for td in task_dirs:
        td.mkdir(parents=True, exist_ok=True)

    job_dir = tmp_path / "jobs"
    config = JobConfig(
        job_name="overlay-test",
        jobs_dir=job_dir,
        n_attempts=n_attempts,
        context_overlay_id=context_overlay_id,
        agents=[AgentConfig() for _ in range(n_agents)],
    )
    task_configs = [_task_config(td) for td in task_dirs]
    return Job(config, _task_configs=task_configs, _metrics=defaultdict(list))


# ---------------------------------------------------------------------------
# JobConfig field: defaults and value
# ---------------------------------------------------------------------------


def test_job_config_context_overlay_id_defaults_to_none() -> None:
    config = JobConfig(job_name="test")
    assert config.context_overlay_id is None


def test_job_config_context_overlay_id_accepts_baseline_v1() -> None:
    config = JobConfig(job_name="test", context_overlay_id="baseline-v1")
    assert config.context_overlay_id == "baseline-v1"


def test_job_config_context_overlay_id_accepts_none_explicitly() -> None:
    config = JobConfig(job_name="test", context_overlay_id=None)
    assert config.context_overlay_id is None


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_job_config_round_trips_with_baseline_v1() -> None:
    config = JobConfig(job_name="test", context_overlay_id="baseline-v1")
    data = json.loads(config.model_dump_json())
    restored = JobConfig.model_validate(data)
    assert restored.context_overlay_id == "baseline-v1"


def test_job_config_round_trips_with_none_overlay() -> None:
    config = JobConfig(job_name="test", context_overlay_id=None)
    data = json.loads(config.model_dump_json())
    restored = JobConfig.model_validate(data)
    assert restored.context_overlay_id is None


def test_job_config_legacy_json_without_field_loads_as_none() -> None:
    """Old job config JSON without context_overlay_id must load as None."""
    config = JobConfig(job_name="test")
    data = json.loads(config.model_dump_json())
    data.pop("context_overlay_id", None)
    assert "context_overlay_id" not in data
    restored = JobConfig.model_validate(data)
    assert restored.context_overlay_id is None


# ---------------------------------------------------------------------------
# Propagation: None overlay
# ---------------------------------------------------------------------------


def test_init_trial_configs_propagates_none_overlay(tmp_path: Path) -> None:
    job = _make_job(tmp_path, context_overlay_id=None)
    job._close_logger_handlers()

    assert len(job._trial_configs) >= 1
    for tc in job._trial_configs:
        assert tc.context_overlay_id is None


# ---------------------------------------------------------------------------
# Propagation: baseline-v1
# ---------------------------------------------------------------------------


def test_init_trial_configs_propagates_baseline_v1(tmp_path: Path) -> None:
    job = _make_job(tmp_path, context_overlay_id="baseline-v1")
    job._close_logger_handlers()

    assert len(job._trial_configs) >= 1
    for tc in job._trial_configs:
        assert tc.context_overlay_id == "baseline-v1"


# ---------------------------------------------------------------------------
# Multiple tasks × agents × attempts
# ---------------------------------------------------------------------------


def test_overlay_propagates_across_all_tasks_agents_attempts(tmp_path: Path) -> None:
    """All n_tasks × n_agents × n_attempts trial configs receive the overlay."""
    n_tasks, n_agents, n_attempts = 3, 2, 2
    job = _make_job(
        tmp_path,
        context_overlay_id="baseline-v1",
        n_attempts=n_attempts,
        n_tasks=n_tasks,
        n_agents=n_agents,
    )
    job._close_logger_handlers()

    expected_count = n_attempts * n_tasks * n_agents
    assert len(job._trial_configs) == expected_count
    for tc in job._trial_configs:
        assert tc.context_overlay_id == "baseline-v1", (
            f"Trial config {tc.trial_name!r} missing overlay; "
            f"got {tc.context_overlay_id!r}"
        )


def test_no_overlay_across_all_tasks_agents_attempts(tmp_path: Path) -> None:
    n_tasks, n_agents, n_attempts = 2, 2, 3
    job = _make_job(
        tmp_path,
        context_overlay_id=None,
        n_attempts=n_attempts,
        n_tasks=n_tasks,
        n_agents=n_agents,
    )
    job._close_logger_handlers()

    expected_count = n_attempts * n_tasks * n_agents
    assert len(job._trial_configs) == expected_count
    for tc in job._trial_configs:
        assert tc.context_overlay_id is None


# ---------------------------------------------------------------------------
# Lock / TrialLock records the propagated overlay
# ---------------------------------------------------------------------------


def test_trial_lock_records_overlay_from_job_config(tmp_path: Path) -> None:
    """TrialLock.context_overlay_id matches what JobConfig declared."""
    job = _make_job(tmp_path, context_overlay_id="baseline-v1", n_tasks=2, n_agents=1)
    job._close_logger_handlers()

    lock = build_job_lock(config=job.config, trial_configs=job._trial_configs)
    assert len(lock.trials) == 2
    for tl in lock.trials:
        assert tl.context_overlay_id == "baseline-v1"


def test_trial_lock_records_none_overlay_from_job_config(tmp_path: Path) -> None:
    job = _make_job(tmp_path, context_overlay_id=None, n_tasks=2)
    job._close_logger_handlers()

    lock = build_job_lock(config=job.config, trial_configs=job._trial_configs)
    for tl in lock.trials:
        assert tl.context_overlay_id is None


def test_trial_lock_legacy_json_without_overlay_loads_as_none(tmp_path: Path) -> None:
    """Saved TrialLock JSON that predates the field deserialises as None."""
    task_dir = tmp_path / "t"
    task_dir.mkdir()
    tc = TrialConfig(task=_task_config(task_dir), context_overlay_id="treatment-x")
    tl = _build_lock_trial(tc)
    data = json.loads(tl.model_dump_json())
    data.pop("context_overlay_id", None)
    assert "context_overlay_id" not in data
    restored = TrialLock.model_validate(data)
    assert restored.context_overlay_id is None


def test_job_lock_round_trips_with_overlay(tmp_path: Path) -> None:
    """Full JobLock serialisation round-trip preserves per-trial overlay ids."""
    job = _make_job(tmp_path, context_overlay_id="baseline-v1", n_tasks=2)
    job._close_logger_handlers()

    lock = build_job_lock(config=job.config, trial_configs=job._trial_configs)
    data = json.loads(lock.model_dump_json())
    restored = JobLock.model_validate(data)
    for tl in restored.trials:
        assert tl.context_overlay_id == "baseline-v1"


# ---------------------------------------------------------------------------
# AgentContext mirroring boundary: no loss from JobConfig → metadata
# ---------------------------------------------------------------------------


def test_overlay_reaches_agent_context_metadata(tmp_path: Path) -> None:
    """JobConfig overlay propagated through TrialConfig is available for metadata mirroring.

    This test does NOT call Trial._maybe_populate_agent_context directly (that
    integration path is covered by test_provenance_contracts.py). Instead it verifies
    that, after _init_trial_configs, every TrialConfig.context_overlay_id carries the
    expected value so the downstream mirroring call will receive the correct input.
    """
    job = _make_job(tmp_path, context_overlay_id="baseline-v1")
    job._close_logger_handlers()

    for tc in job._trial_configs:
        # Simulate what _mirror_overlay_to_metadata does: copy overlay into metadata.
        ctx = AgentContext()
        if tc.context_overlay_id is not None:
            metadata = dict(ctx.metadata or {})
            metadata["context_overlay_id"] = tc.context_overlay_id
            ctx.metadata = metadata
        assert ctx.metadata == {"context_overlay_id": "baseline-v1"}


def test_none_overlay_leaves_agent_context_metadata_unchanged(tmp_path: Path) -> None:
    job = _make_job(tmp_path, context_overlay_id=None)
    job._close_logger_handlers()

    for tc in job._trial_configs:
        ctx = AgentContext(metadata={"pre_existing": "value"})
        if tc.context_overlay_id is not None:
            metadata = dict(ctx.metadata or {})
            metadata["context_overlay_id"] = tc.context_overlay_id
            ctx.metadata = metadata
        # overlay is None → metadata unchanged
        assert ctx.metadata == {"pre_existing": "value"}
        assert "context_overlay_id" not in (ctx.metadata or {})
