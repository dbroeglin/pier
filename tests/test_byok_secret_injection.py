"""Gate 2b regression suite: COPILOT_PROVIDER_API_KEY secret-safe injection proof.

Purpose
-------
Prove, at the source level, that the ``COPILOT_PROVIDER_API_KEY`` carrier variable
(used in Plan 002 as a short-lived Entra bearer token) cannot leak into any
Pier-managed serialized surface when it is delivered via the documented
``--env-file`` / ``AgentConfig.env`` template route.

Surfaces covered
----------------
1. config.json  – AgentConfig.env field_serializer  (templatize_sensitive_env)
2. lock.json    – CLI invocation sanitizer           (sanitize_cli_invocation)
3. result.json  – same TrialConfig serializer, verified via round-trip
4. debug logs   – _exec log path                     (_redact_env_for_log)
5. subprocess forwarding – resolve_env_vars + build_process_env
6. Verifier isolation    – VerifierConfig.env is independent of AgentConfig.env

Claims not made by these tests
-------------------------------
- Foundry connectivity, Entra auth, or Azure cost
- Runtime Docker container behavior (no containers are started)
- Actual credential values (all values are placeholder strings)
"""

import asyncio
import logging
import os
from pathlib import Path

import pytest

from pier.agents.installed.base import _redact_env_for_log
from pier.agents.installed.copilot_cli import CopilotCli
from pier.models.job.lock import sanitize_cli_invocation
from pier.models.trial.config import AgentConfig, VerifierConfig
from pier.utils.env import (
    is_sensitive_env_key,
    resolve_env_vars,
    templatize_sensitive_env,
)

# ---------------------------------------------------------------------------
# Placeholder constants — never real credentials
# ---------------------------------------------------------------------------

_PLACEHOLDER_API_KEY = "PLACEHOLDER_ENTRA_BEARER_FOR_GATE2B_TESTS_ONLY"
_PLACEHOLDER_BASE_URL = "https://my-resource.services.ai.azure.com/mai/v1"
_PLACEHOLDER_GH_TOKEN = "ghp_PLACEHOLDER_FOR_GATE2B_TESTS_ONLY_ABCDEF"

_BYOK_TEMPLATE_ENV: dict[str, str] = {
    "COPILOT_PROVIDER_TYPE": "openai",
    "COPILOT_PROVIDER_BASE_URL": _PLACEHOLDER_BASE_URL,
    "COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY}",
    "COPILOT_MODEL": "MAI-Thinking-1",
}

_BYOK_LITERAL_ENV: dict[str, str] = {
    "COPILOT_PROVIDER_TYPE": "openai",
    "COPILOT_PROVIDER_BASE_URL": _PLACEHOLDER_BASE_URL,
    "COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY,
    "COPILOT_MODEL": "MAI-Thinking-1",
}


# ===========================================================================
# Part 1: templatize_sensitive_env — the core serialization guard
# ===========================================================================


def test_copilot_provider_api_key_is_detected_as_sensitive():
    """KEY suffix triggers is_sensitive_env_key for COPILOT_PROVIDER_API_KEY."""
    assert is_sensitive_env_key("COPILOT_PROVIDER_API_KEY")


def test_templatize_preserves_template_value():
    """When the value is already a template, it is not further transformed."""
    env = {"COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY}"}
    result = templatize_sensitive_env(env)
    assert result["COPILOT_PROVIDER_API_KEY"] == "${COPILOT_PROVIDER_API_KEY}"


def test_templatize_converts_literal_to_template_when_in_os_environ(monkeypatch):
    """Literal secret value → ${VAR} template when os.environ holds same value."""
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", _PLACEHOLDER_API_KEY)
    env = {"COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY}
    result = templatize_sensitive_env(env)
    assert result["COPILOT_PROVIDER_API_KEY"] == "${COPILOT_PROVIDER_API_KEY}"
    assert _PLACEHOLDER_API_KEY not in result["COPILOT_PROVIDER_API_KEY"]


def test_templatize_redacts_literal_not_in_os_environ():
    """Literal secret value not in os.environ → redacted placeholder, not raw value."""
    # Ensure key is NOT in os.environ for this test
    env = {"COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY}
    result = templatize_sensitive_env(env)
    assert _PLACEHOLDER_API_KEY not in result["COPILOT_PROVIDER_API_KEY"]
    # Redacted form must contain at least one '*'
    assert "*" in result["COPILOT_PROVIDER_API_KEY"]


def test_templatize_preserves_non_sensitive_byok_vars():
    """Non-secret BYOK vars (TYPE, BASE_URL, MODEL) pass through unchanged."""
    env = {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": _PLACEHOLDER_BASE_URL,
        "COPILOT_MODEL": "MAI-Thinking-1",
    }
    result = templatize_sensitive_env(env)
    assert result == env


# ===========================================================================
# Part 2: AgentConfig.env serialization → config.json / result.json safety
# ===========================================================================


def test_agent_config_serializes_template_env_without_resolution():
    """AgentConfig.env with template value → preserved as template in model_dump."""
    config = AgentConfig(env={"COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY}"})
    serialized = config.model_dump()
    assert serialized["env"]["COPILOT_PROVIDER_API_KEY"] == "${COPILOT_PROVIDER_API_KEY}"


def test_agent_config_serializes_literal_to_safe_form_when_in_environ(monkeypatch):
    """Literal API key in AgentConfig.env → templatized (not raw) in model_dump."""
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", _PLACEHOLDER_API_KEY)
    config = AgentConfig(env={"COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY})
    serialized = config.model_dump()
    assert _PLACEHOLDER_API_KEY not in str(serialized["env"])


def test_agent_config_json_serializes_template_env(tmp_path):
    """model_dump_json round-trip confirms no raw API key value in JSON output."""
    config = AgentConfig(
        name="copilot-cli",
        env={
            "COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY}",
            "COPILOT_PROVIDER_TYPE": "openai",
        },
    )
    json_str = config.model_dump_json()
    assert _PLACEHOLDER_API_KEY not in json_str
    assert "${COPILOT_PROVIDER_API_KEY}" in json_str


def test_verifier_config_env_is_separate_from_agent_env():
    """VerifierConfig.env is an independent field; agent secrets do not flow into it."""
    agent_config = AgentConfig(
        env={"COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY}"}
    )
    verifier_config = VerifierConfig()
    assert "COPILOT_PROVIDER_API_KEY" not in verifier_config.env
    assert "COPILOT_PROVIDER_API_KEY" not in verifier_config.model_dump()["env"]
    # Agent env is separate
    assert "COPILOT_PROVIDER_API_KEY" in agent_config.env


# ===========================================================================
# Part 3: lock.json CLI invocation sanitization
# ===========================================================================


def test_sanitize_cli_invocation_redacts_api_key_literal(monkeypatch):
    """--agent-env COPILOT_PROVIDER_API_KEY=<literal> → templatized in lock.json."""
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", _PLACEHOLDER_API_KEY)
    argv = [
        "pier", "run", "--config", "smoke.yaml",
        "--agent-env", f"COPILOT_PROVIDER_API_KEY={_PLACEHOLDER_API_KEY}",
    ]
    sanitized = sanitize_cli_invocation(argv)
    sanitized_str = " ".join(sanitized)
    assert _PLACEHOLDER_API_KEY not in sanitized_str
    # Must still be templatized (not dropped entirely)
    assert "COPILOT_PROVIDER_API_KEY" in sanitized_str


def test_sanitize_cli_invocation_preserves_template_arg(monkeypatch):
    """--agent-env COPILOT_PROVIDER_API_KEY=${...} → template preserved in lock.json."""
    argv = [
        "pier", "run",
        "--agent-env", "COPILOT_PROVIDER_API_KEY=${COPILOT_PROVIDER_API_KEY}",
    ]
    sanitized = sanitize_cli_invocation(argv)
    assert "${COPILOT_PROVIDER_API_KEY}" in " ".join(sanitized)


def test_sanitize_cli_invocation_preserves_non_sensitive_args():
    """Non-sensitive args are not altered."""
    argv = ["pier", "run", "--job-name", "plan002-smoke-01", "--agent-env", "SAFE_VAR=value"]
    sanitized = sanitize_cli_invocation(argv)
    assert "SAFE_VAR=value" in sanitized


@pytest.mark.parametrize("flag", ["--agent-env", "--ae"])
def test_sanitize_cli_invocation_handles_both_agent_env_flags(flag, monkeypatch):
    """Both --agent-env and --ae aliases are sanitized."""
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", _PLACEHOLDER_API_KEY)
    argv = ["pier", "run", flag, f"COPILOT_PROVIDER_API_KEY={_PLACEHOLDER_API_KEY}"]
    sanitized = sanitize_cli_invocation(argv)
    assert _PLACEHOLDER_API_KEY not in " ".join(sanitized)


# ===========================================================================
# Part 4: debug log redaction (_redact_env_for_log)
# ===========================================================================


def test_redact_env_for_log_redacts_copilot_provider_api_key():
    """_redact_env_for_log must not pass COPILOT_PROVIDER_API_KEY value through."""
    raw = {
        "COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY,
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_GITHUB_TOKEN": _PLACEHOLDER_GH_TOKEN,
    }
    result = _redact_env_for_log(raw)
    assert _PLACEHOLDER_API_KEY not in result["COPILOT_PROVIDER_API_KEY"]
    assert _PLACEHOLDER_GH_TOKEN not in result["COPILOT_GITHUB_TOKEN"]
    # Non-secret values survive
    assert result["COPILOT_PROVIDER_TYPE"] == "openai"


def test_redact_env_for_log_redacts_all_byok_sensitive_keys():
    """All BYOK vars with KEY/TOKEN in name are redacted in log output."""
    sentinel = "SUPER_SECRET_SENTINEL_FOR_GATE2B"
    raw = {
        "COPILOT_PROVIDER_API_KEY": sentinel,
        "COPILOT_GITHUB_TOKEN": sentinel,
        "GH_TOKEN": sentinel,
        "GITHUB_TOKEN": sentinel,
    }
    result = _redact_env_for_log(raw)
    for key in raw:
        assert sentinel not in result[key], f"Secret leaked in {key}"
    # Non-secret BYOK vars pass through
    non_secret = {"COPILOT_PROVIDER_TYPE": "openai", "COPILOT_MODEL": "MAI-Thinking-1"}
    merged = _redact_env_for_log({**raw, **non_secret})
    assert merged["COPILOT_PROVIDER_TYPE"] == "openai"
    assert merged["COPILOT_MODEL"] == "MAI-Thinking-1"


# ===========================================================================
# Part 5: subprocess forwarding via resolve_env_vars + build_process_env
# ===========================================================================


def test_resolve_env_vars_resolves_api_key_template(monkeypatch):
    """Template ${COPILOT_PROVIDER_API_KEY} resolves to actual value from os.environ."""
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", _PLACEHOLDER_API_KEY)
    env_dict = {"COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY}"}
    resolved = resolve_env_vars(env_dict)
    assert resolved["COPILOT_PROVIDER_API_KEY"] == _PLACEHOLDER_API_KEY


def test_resolve_env_vars_raises_on_missing_template_var():
    """resolve_env_vars raises ValueError for unset template vars (fail-closed)."""
    env_dict = {"COPILOT_PROVIDER_API_KEY": "${COPILOT_PROVIDER_API_KEY_UNSET_XYZ}"}
    with pytest.raises(ValueError, match="COPILOT_PROVIDER_API_KEY_UNSET_XYZ"):
        resolve_env_vars(env_dict)


def test_build_process_env_includes_byok_api_key_from_extra_env(tmp_path, monkeypatch):
    """COPILOT_PROVIDER_API_KEY in extra_env reaches build_process_env output (subprocess env)."""
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", _PLACEHOLDER_API_KEY)
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={
            "COPILOT_GITHUB_TOKEN": _PLACEHOLDER_GH_TOKEN,
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_BASE_URL": _PLACEHOLDER_BASE_URL,
            "COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY,
            "COPILOT_MODEL": "MAI-Thinking-1",
        },
    )
    process_env = agent.build_process_env({"COPILOT_GITHUB_TOKEN": _PLACEHOLDER_GH_TOKEN})
    # The resolved value must be present for the subprocess to use
    assert process_env.get("COPILOT_PROVIDER_API_KEY") == _PLACEHOLDER_API_KEY
    # Non-secret companion vars also present
    assert process_env.get("COPILOT_PROVIDER_TYPE") == "openai"
    assert process_env.get("COPILOT_PROVIDER_BASE_URL") == _PLACEHOLDER_BASE_URL
    assert process_env.get("COPILOT_MODEL") == "MAI-Thinking-1"


def test_build_process_env_api_key_not_present_without_extra_env(tmp_path):
    """Without extra_env (no --env-file forwarding), API key does not appear in process env."""
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={"COPILOT_GITHUB_TOKEN": _PLACEHOLDER_GH_TOKEN},
    )
    process_env = agent.build_process_env({"COPILOT_GITHUB_TOKEN": _PLACEHOLDER_GH_TOKEN})
    assert "COPILOT_PROVIDER_API_KEY" not in process_env


# ===========================================================================
# Part 6: _exec log path does not leak API key
# ===========================================================================
# Reuses the recording-environment pattern from test_base_exec_redaction.py

from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import EnvironmentCapabilities
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig, TaskOS
from pier.models.trial.paths import TrialPaths


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _StubAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "stub-byok-test"

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(agent_name="stub-byok-test", steps=[InstallStep(run="echo")])

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        pass


class _RecordingEnvironment(BaseEnvironment):
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


def test_exec_log_does_not_leak_copilot_provider_api_key(tmp_path: Path):
    """COPILOT_PROVIDER_API_KEY must be redacted in _exec log records."""
    handler = _ListHandler()
    logger = logging.getLogger(f"test.byok.exec_log.{id(handler)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)

    agent = _StubAgent(logs_dir=tmp_path, logger=logger)
    env = {
        "COPILOT_GITHUB_TOKEN": _PLACEHOLDER_GH_TOKEN,
        "COPILOT_PROVIDER_API_KEY": _PLACEHOLDER_API_KEY,
        "COPILOT_PROVIDER_TYPE": "openai",
    }
    asyncio.run(agent._exec(_RecordingEnvironment(tmp_path), "echo hi", env=env))

    running = [r for r in handler.records if "Running command" in r.getMessage()]
    assert running, "Expected a 'Running command' debug record"
    env_logged = running[0].env
    assert _PLACEHOLDER_API_KEY not in str(env_logged), (
        f"COPILOT_PROVIDER_API_KEY leaked in log: {env_logged}"
    )
    assert _PLACEHOLDER_GH_TOKEN not in str(env_logged), (
        f"COPILOT_GITHUB_TOKEN leaked in log: {env_logged}"
    )
    # Non-secret provenance vars survive
    assert env_logged.get("COPILOT_PROVIDER_TYPE") == "openai"
