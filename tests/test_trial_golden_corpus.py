import asyncio
import functools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pier.agents.installed.base import NonZeroAgentExitCodeError
from pier.environments.base import ExecResult
from pier.models.task.config import TaskOS
from pier.models.trial.config import TaskConfig as TrialTaskConfig
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TrialConfig,
    VerifierConfig,
)
from pier.models.trial.paths import EnvironmentPaths
from pier.models.trial.result import AgentInfo
from pier.trial.trial import Trial


def run_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _task_with_artifacts(
    tmp_path: Path,
    *,
    artifacts: list[str] | None = None,
) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        f"artifacts = {repr(artifacts or [])}\n"
        "[agent]\ntimeout_sec = 10.0\n"
        "[verifier]\ntimeout_sec = 10.0\n"
        "[environment]\n",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Do nothing.\n", encoding="utf-8")
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n", encoding="utf-8"
    )
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n", encoding="utf-8"
    )
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    return task_dir


def _make_env(
    *,
    missing_patch: bool = False,
    invalid_patch: bool = False,
) -> AsyncMock:
    env = AsyncMock()
    env.default_user = None
    env.capabilities = SimpleNamespace(mounted=True)
    env.task_os = TaskOS.LINUX
    env.env_paths = EnvironmentPaths()
    env.exec.return_value = ExecResult(stdout="/workspace", stderr="", return_code=0)
    env.is_dir.return_value = False
    env.run_healthcheck.return_value = None
    env.start.return_value = None
    env.stop.return_value = None
    env.reset_dirs.return_value = None
    env.empty_dirs.return_value = None
    env.upload_dir.return_value = None
    env.upload_file.return_value = None
    env.prepare_logs_for_host.return_value = None

    async def download_file(source_path, target_path):
        source = str(source_path)
        if source.endswith("/model.patch") and missing_patch:
            raise FileNotFoundError(source)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.endswith("/model.patch") and invalid_patch:
            target.write_text("this-is-not-a-unified-diff\n", encoding="utf-8")
        else:
            target.write_text(source, encoding="utf-8")

    async def download_dir(source_dir, target_dir):
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "artifact.txt").write_text(str(source_dir), encoding="utf-8")

    env.download_file.side_effect = download_file
    env.download_dir.side_effect = download_dir
    env.download_dir_with_exclusions.side_effect = download_dir
    return env


def _mock_agent(*, run_side_effect=None) -> MagicMock:
    run_mock = AsyncMock()
    if run_side_effect is not None:
        run_mock.side_effect = run_side_effect
    return MagicMock(
        name=lambda: "oracle",
        version=lambda: "1.0",
        SUPPORTS_WINDOWS=True,
        setup=AsyncMock(),
        run=run_mock,
        to_agent_info=lambda: AgentInfo(name="oracle", version="1.0"),
        install_spec=lambda: None,
        network_allowlist=lambda: None,
    )


async def _run_trial(
    *,
    task_dir: Path,
    trials_dir: Path,
    env: AsyncMock,
    reward_setup,
    run_side_effect=None,
    verifier_disable: bool = False,
):
    config = TrialConfig(
        task=TrialTaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(type="docker", delete=False),
        verifier=VerifierConfig(disable=verifier_disable),
    )

    with (
        patch(
            "pier.trial.trial.EnvironmentFactory.create_environment_from_config",
            return_value=env,
        ),
        patch(
            "pier.trial.execution.AgentFactory.create_agent_from_config",
            return_value=_mock_agent(run_side_effect=run_side_effect),
        ),
    ):
        trial = await Trial.create(config)
        trial._trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        reward_setup(trial._trial_paths)
        await trial.run()
        return trial


@run_async
async def test_gc_2_trial_accepts_reward_zero(tmp_path: Path):
    task_dir = _task_with_artifacts(tmp_path)
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    trial = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(),
        reward_setup=lambda paths: paths.reward_text_path.write_text("0", encoding="utf-8"),
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards == {"reward": 0.0}
    assert trial.result.exception_info is None


@run_async
async def test_gc_3_nonzero_agent_exit_sets_exception_info(tmp_path: Path):
    task_dir = _task_with_artifacts(tmp_path)
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    trial = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(),
        reward_setup=lambda _paths: None,
        run_side_effect=NonZeroAgentExitCodeError("copilot auth failed"),
        verifier_disable=True,
    )

    assert trial.result.exception_info is not None
    assert trial.result.exception_info.exception_type == "NonZeroAgentExitCodeError"
    assert trial.result.verifier_result is None


@run_async
async def test_gc_7_missing_patch_still_allows_reward_zero_verdict(tmp_path: Path):
    task_dir = _task_with_artifacts(tmp_path, artifacts=["/logs/artifacts/model.patch"])
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    trial = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(missing_patch=True),
        reward_setup=lambda paths: paths.reward_text_path.write_text("0", encoding="utf-8"),
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards == {"reward": 0.0}
    assert trial.result.exception_info is None

    manifest = json.loads(
        (trial._trial_paths.artifacts_dir / "manifest.json").read_text()
    )
    entry = next(
        item for item in manifest if item["source"] == "/logs/artifacts/model.patch"
    )
    assert entry["status"] == "failed"


@run_async
async def test_gc_8_invalid_patch_can_be_graded_as_reward_zero(tmp_path: Path):
    task_dir = _task_with_artifacts(tmp_path, artifacts=["/logs/artifacts/model.patch"])
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    trial = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(invalid_patch=True),
        reward_setup=lambda paths: paths.reward_text_path.write_text("0", encoding="utf-8"),
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards == {"reward": 0.0}
    assert trial.result.exception_info is None
    assert (trial._trial_paths.artifacts_dir / "model.patch").read_text(encoding="utf-8").startswith(
        "this-is-not-a-unified-diff"
    )


@pytest.mark.parametrize(
    ("reward_setup", "expected_exception"),
    [
        (
            lambda paths: paths.reward_json_path.write_text("not json", encoding="utf-8"),
            "VerifierOutputParseError",
        ),
        (lambda paths: paths.reward_text_path.write_text("", encoding="utf-8"), "RewardFileEmptyError"),
        (lambda _paths: None, "RewardFileNotFoundError"),
    ],
)
@run_async
async def test_gc_9_verifier_reward_file_failures_surface_on_trial(
    tmp_path: Path,
    reward_setup,
    expected_exception: str,
):
    task_dir = _task_with_artifacts(tmp_path)
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    trial = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(),
        reward_setup=reward_setup,
    )

    assert trial.result.verifier_result is None
    assert trial.result.exception_info is not None
    assert trial.result.exception_info.exception_type == expected_exception


@run_async
async def test_gc_11_repeated_task_keeps_checksum_but_changes_trial_ids(tmp_path: Path):
    task_dir = _task_with_artifacts(tmp_path)
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    first = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(),
        reward_setup=lambda paths: paths.reward_text_path.write_text("1", encoding="utf-8"),
    )
    second = await _run_trial(
        task_dir=task_dir,
        trials_dir=trials_dir,
        env=_make_env(),
        reward_setup=lambda paths: paths.reward_text_path.write_text("0", encoding="utf-8"),
    )

    assert first.result.task_checksum == second.result.task_checksum
    assert first.result.id != second.result.id
    # Copilot runs are stochastic; repeated-task checks must not require equal rewards.
    assert first.result.verifier_result is not None
    assert second.result.verifier_result is not None
    assert first.result.verifier_result.rewards["reward"] == 1.0
    assert second.result.verifier_result.rewards["reward"] == 0.0
