"""Tests for additive Docker mount behavior.

Verifies that explicit ``environment.mounts`` augments — rather than replaces —
the default evidence mounts (verifier / agent / artifacts).

Gate: every path through mounts_json (None, [], non-empty) must leave the
default log mounts intact, and non-empty custom mounts must be appended after
the defaults.
"""

from pathlib import Path

from pier.environments.docker.docker import DockerEnvironment
from pier.models.task.config import EnvironmentConfig, TaskOS
from pier.models.trial.config import ServiceVolumeConfig
from pier.models.trial.paths import EnvironmentPaths, TrialPaths


def _make_env(
    tmp_path: Path,
    mounts_json: list[ServiceVolumeConfig] | None = None,
) -> DockerEnvironment:
    """Construct a minimal DockerEnvironment backed by a scratch Dockerfile."""
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM scratch\n")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return DockerEnvironment(
        environment_dir=env_dir,
        environment_name="test-env",
        session_id="test-session",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(os=TaskOS.LINUX),
        mounts_json=mounts_json,
    )


def _default_targets() -> set[str]:
    paths = EnvironmentPaths()
    return {
        str(paths.verifier_dir),
        str(paths.agent_dir),
        str(paths.artifacts_dir),
    }


def test_docker_mounts_none_yields_only_defaults(tmp_path: Path) -> None:
    """mounts_json=None → only the three default evidence mounts, nothing extra."""
    env = _make_env(tmp_path, mounts_json=None)
    targets = {m["target"] for m in env._mounts_json}
    assert targets == _default_targets()


def test_docker_mounts_empty_list_preserves_defaults(tmp_path: Path) -> None:
    """mounts_json=[] → default evidence mounts preserved, no extras added."""
    env = _make_env(tmp_path, mounts_json=[])
    targets = {m["target"] for m in env._mounts_json}
    assert targets == _default_targets()


def test_docker_mounts_custom_mount_appended_to_defaults(tmp_path: Path) -> None:
    """Custom mounts are appended after defaults, never replacing them."""
    custom: ServiceVolumeConfig = {
        "type": "bind",
        "source": "/host/overlay",
        "target": "/container/overlay",
    }
    env = _make_env(tmp_path, mounts_json=[custom])
    targets = {m["target"] for m in env._mounts_json}
    assert targets == _default_targets() | {"/container/overlay"}
    assert env._mounts_json[-1] == custom
