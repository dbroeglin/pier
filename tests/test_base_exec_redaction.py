"""Regression tests for DEF-1: _exec must never log raw credential values.

The test exercises the actual logging.LogRecord emitted by BaseInstalledAgent._exec,
verifying that sensitive env values are redacted at the sink while safe diagnostics
(command, non-sensitive env keys) remain useful.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from pier.agents.installed.base import BaseInstalledAgent, _redact_env_for_log
from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import EnvironmentCapabilities
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig, TaskOS
from pier.models.trial.paths import TrialPaths


# ---------------------------------------------------------------------------
# Minimal concrete fixtures
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Logging handler that accumulates all emitted LogRecord objects."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _StubAgent(BaseInstalledAgent):
    """Minimal concrete subclass of BaseInstalledAgent for _exec testing."""

    @staticmethod
    def name() -> str:
        return "stub-agent"

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name="stub-agent",
            steps=[InstallStep(run="echo stub")],
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        pass


class _RecordingEnvironment(BaseEnvironment):
    """Minimal concrete environment that records exec calls."""

    def __init__(self, tmp_path: Path) -> None:
        trial_paths = TrialPaths(tmp_path / "trial")
        trial_paths.mkdir()
        super().__init__(
            environment_dir=tmp_path,
            environment_name="test",
            session_id="session",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(os=TaskOS.LINUX),
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def upload_file(self, source_path, target_path) -> None:
        pass

    async def upload_dir(self, source_dir, target_dir) -> None:
        pass

    async def download_file(self, source_path, target_path) -> None:
        pass

    async def download_dir(self, source_dir, target_dir) -> None:
        pass

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        return ExecResult(return_code=0, stdout="ok", stderr="")


def _make_agent_with_handler(
    tmp_path: Path, handler: _ListHandler
) -> _StubAgent:
    """Build a _StubAgent whose log records flow into *handler*."""
    capture_logger = logging.getLogger(f"test.exec_redaction.{id(handler)}")
    capture_logger.setLevel(logging.DEBUG)
    capture_logger.propagate = False
    capture_logger.addHandler(handler)
    return _StubAgent(logs_dir=tmp_path, logger=capture_logger)


# ---------------------------------------------------------------------------
# Unit tests for _redact_env_for_log helper
# ---------------------------------------------------------------------------


def test_redact_env_for_log_strips_token_key():
    raw = {"COPILOT_GITHUB_TOKEN": "ghp_SUPER_SECRET_abc123", "PATH": "/usr/bin"}
    result = _redact_env_for_log(raw)
    assert "ghp_SUPER_SECRET_abc123" not in result["COPILOT_GITHUB_TOKEN"]
    assert result["PATH"] == "/usr/bin"


def test_redact_env_for_log_covers_gh_token_and_github_token():
    secret = "ghp_SUPER_SECRET_abc123"
    raw = {"GH_TOKEN": secret, "GITHUB_TOKEN": secret}
    result = _redact_env_for_log(raw)
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        assert secret not in result[key], f"Secret leaked in {key}"


def test_redact_env_for_log_preserves_safe_values():
    raw = {
        "COPILOT_CLI_MODE": "agent",
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/agent",
    }
    result = _redact_env_for_log(raw)
    assert result == raw


def test_redact_env_for_log_redacts_generic_sensitive_keys():
    secret = "top_secret_value_xyz"
    raw = {
        "MY_API_KEY": secret,
        "DB_PASSWORD": secret,
        "AUTH_TOKEN": secret,
        "AWS_SECRET_ACCESS_KEY": secret,
    }
    result = _redact_env_for_log(raw)
    for key in raw:
        assert secret not in result[key], f"Secret leaked in {key}"


def test_redact_env_for_log_preserves_zero_for_numeric_zero_string():
    raw = {"SAFE_COUNT": "0", "TIMEOUT": "30"}
    result = _redact_env_for_log(raw)
    assert result == raw


# ---------------------------------------------------------------------------
# Integration tests: verify the actual LogRecord emitted by _exec
# ---------------------------------------------------------------------------

_SENTINEL = "ghp_SUPER_SECRET_TOKEN_abc123xyz"  # unique sentinel that must never appear in logs


def test_exec_log_record_does_not_contain_copilot_github_token(tmp_path: Path):
    """DEF-1 primary: COPILOT_GITHUB_TOKEN must be redacted in the Running-command record."""
    handler = _ListHandler()
    agent = _make_agent_with_handler(tmp_path, handler)
    env = {
        "COPILOT_GITHUB_TOKEN": _SENTINEL,
        "PATH": "/usr/bin",
        "COPILOT_CLI_MODE": "agent",
    }

    asyncio.run(agent._exec(_RecordingEnvironment(tmp_path), "echo hi", env=env))

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running, "Expected a 'Running command' debug record from _exec"
    record = running[0]

    env_logged = record.env
    assert isinstance(env_logged, dict), "record.env must be a dict"
    assert _SENTINEL not in str(env_logged), (
        f"Raw credential leaked in record.env: {env_logged}"
    )
    # Safe values must survive redaction
    assert env_logged.get("PATH") == "/usr/bin"
    assert env_logged.get("COPILOT_CLI_MODE") == "agent"


def test_exec_log_record_does_not_contain_gh_token(tmp_path: Path):
    """DEF-1: GH_TOKEN must be redacted."""
    handler = _ListHandler()
    agent = _make_agent_with_handler(tmp_path, handler)
    env = {"GH_TOKEN": _SENTINEL, "HOME": "/home/agent"}

    asyncio.run(agent._exec(_RecordingEnvironment(tmp_path), "ls", env=env))

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running
    assert _SENTINEL not in str(running[0].env)
    assert running[0].env.get("HOME") == "/home/agent"


def test_exec_log_record_does_not_contain_github_token(tmp_path: Path):
    """DEF-1: GITHUB_TOKEN must be redacted."""
    handler = _ListHandler()
    agent = _make_agent_with_handler(tmp_path, handler)
    env = {"GITHUB_TOKEN": _SENTINEL, "COPILOT_CLI_EFFORT": "low"}

    asyncio.run(agent._exec(_RecordingEnvironment(tmp_path), "ls", env=env))

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running
    assert _SENTINEL not in str(running[0].env)
    assert running[0].env.get("COPILOT_CLI_EFFORT") == "low"


def test_exec_log_record_redacts_all_three_credential_keys_simultaneously(tmp_path: Path):
    """DEF-1: All three canonical credential keys must be redacted together."""
    handler = _ListHandler()
    agent = _make_agent_with_handler(tmp_path, handler)
    env = {
        "COPILOT_GITHUB_TOKEN": _SENTINEL,
        "GH_TOKEN": _SENTINEL,
        "GITHUB_TOKEN": _SENTINEL,
        "PATH": "/usr/bin",
    }

    asyncio.run(agent._exec(_RecordingEnvironment(tmp_path), "echo hi", env=env))

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running
    record = running[0]
    env_logged = record.env
    assert _SENTINEL not in str(env_logged)
    assert env_logged.get("PATH") == "/usr/bin"
    # Redacted values must still be present (non-empty placeholder)
    for key in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        assert key in env_logged
        assert env_logged[key]  # not empty


def test_exec_with_extra_env_also_redacts_merged_credentials(tmp_path: Path):
    """DEF-1: credentials injected via extra_env (merged in _exec) must also be redacted."""
    handler = _ListHandler()
    capture_logger = logging.getLogger(f"test.exec_redaction.extra.{id(handler)}")
    capture_logger.setLevel(logging.DEBUG)
    capture_logger.propagate = False
    capture_logger.addHandler(handler)
    agent = _StubAgent(
        logs_dir=tmp_path,
        logger=capture_logger,
        extra_env={"COPILOT_GITHUB_TOKEN": _SENTINEL},
    )

    asyncio.run(
        agent._exec(
            _RecordingEnvironment(tmp_path),
            "echo hi",
            env={"PATH": "/usr/bin"},
        )
    )

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running
    env_logged = running[0].env
    assert _SENTINEL not in str(env_logged)
    assert env_logged.get("PATH") == "/usr/bin"


def test_exec_with_no_env_logs_empty_dict(tmp_path: Path):
    """_exec with env=None must log an empty dict without errors."""
    handler = _ListHandler()
    agent = _make_agent_with_handler(tmp_path, handler)

    asyncio.run(agent._exec(_RecordingEnvironment(tmp_path), "echo hi", env=None))

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running
    assert running[0].env == {}
