"""Tests for Gate 4 metadata: cost provenance, model provenance, token-accounting provenance.

Covers (as enumerated in Gate 4 task):
- Gate 3 numeric sample: AIU 100.27622 → 1.0027622 USD public-list equivalent
- Zero AI credits: _nano_aiu_to_aiu(0) → 0.0 (genuine zero, not None)
- Missing AIU: no cost_usd populated
- Rate provenance: extra fields contain rate, basis, source URL, as-of
- Serialization: FinalMetrics round-trips with cost fields
- Cost aggregation: cost flows through populate_context_from_final_metrics
- Observed model only (no requested model): copilot_observed_models set, no copilot_requested_model
- Requested-vs-observed mismatch: copilot_requested_model present when they differ
- Multiple root models (model changes): full ordered list preserved
- Subagent isolation: root observed_models ≠ subagent observed_models
- Trial boundary refresh: agent_info.model_info updated via _maybe_populate_agent_context
- tokenDetails canonical semantics: source="tokenDetails", cache-inclusive input
- Cache-write capture: cache_write_tokens additive in extra and metadata
- modelMetrics fallback/delta: mm_* fields present when tokenDetails is canonical
- Legacy compatibility: sessions without model/AIU fields remain backward-compatible
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pier.agents.installed.copilot_cli import (
    CopilotCli,
    _GITHUB_AI_CREDIT_RATE_AS_OF,
    _GITHUB_AI_CREDIT_RATE_BASIS,
    _GITHUB_AI_CREDIT_RATE_SOURCE,
    _GITHUB_AI_CREDIT_RATE_USD,
    _nano_aiu_to_aiu,
)
from pier.agents.installed.base import BaseInstalledAgent
from pier.models.agent.context import AgentContext
from pier.models.trajectories.final_metrics import FinalMetrics
from pier.models.trial.config import (
    TaskConfig,
    TrialConfig,
)
from pier.models.trial.result import AgentInfo, TrialResult
from pier.models.task.id import LocalTaskId
from pier.trial.trial import Trial
from pier.utils.trajectory_metrics import populate_context_from_final_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path, *, model_name: str | None = None) -> CopilotCli:
    return CopilotCli(logs_dir=tmp_path, model_name=model_name)


def _write_events(tmp_path: Path, events: list[dict], session_id: str = "s1") -> Path:
    events_path = (
        tmp_path / "copilot-home" / "session-state" / session_id / "events.jsonl"
    )
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return events_path


def _shutdown_event(
    *,
    nano_aiu: float | None = None,
    token_details: dict | None = None,
    model_metrics: dict | None = None,
) -> dict:
    data: dict = {}
    if nano_aiu is not None:
        data["totalNanoAiu"] = nano_aiu
    if token_details is not None:
        data["tokenDetails"] = token_details
    if model_metrics is not None:
        data["modelMetrics"] = model_metrics
    return {"type": "session.shutdown", "data": data}


def _trial_config(task_dir: Path) -> TrialConfig:
    return TrialConfig(task=TaskConfig(path=task_dir))


def _trial_result(config: TrialConfig, task_dir: Path) -> TrialResult:
    return TrialResult(
        task_name=task_dir.name,
        trial_name=config.trial_name,
        trial_uri=(task_dir / config.trial_name).resolve().as_uri(),
        task_id=LocalTaskId(path=task_dir),
        task_checksum="deadbeef" * 8,
        config=config,
        agent_info=AgentInfo(name="copilot-cli", version="1.0.80"),
    )


class _StubCopilotAgent(BaseInstalledAgent):
    """Minimal CopilotCli-like agent that populates metadata with model info."""

    def __init__(
        self,
        logs_dir: Path,
        *,
        model_name: str | None = None,
        observed_models: list[str] | None = None,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name)
        self._observed_models = observed_models or []

    @staticmethod
    def name() -> str:
        return "copilot-cli"

    def version(self) -> str | None:
        return "1.0.80"

    async def setup(self, environment) -> None:  # type: ignore[override]
        pass

    async def run(self, instruction, environment, context) -> None:  # type: ignore[override]
        pass

    def populate_context_post_run(self, context: AgentContext) -> None:
        metadata = dict(context.metadata or {})
        if self._observed_models:
            metadata["copilot_observed_models"] = self._observed_models
            metadata["copilot_model_provenance_source"] = "session_events"
        context.metadata = metadata or None

    def install_spec(self):  # type: ignore[override]
        return None


def _fake_trial(
    tmp_path: Path,
    *,
    result: TrialResult | None,
    agent: BaseInstalledAgent | None = None,
) -> SimpleNamespace:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        _agent=agent or _StubCopilotAgent(tmp_path),
        _result=result,
        _trial_paths=SimpleNamespace(agent_dir=agent_dir),
        config=SimpleNamespace(context_overlay_id=None),
        _agent_logs_download_ok=True,
    )


# ---------------------------------------------------------------------------
# A. _nano_aiu_to_aiu correctness
# ---------------------------------------------------------------------------


def test_nano_aiu_to_aiu_zero_returns_zero_not_none() -> None:
    """Genuine zero nanoAiu (e.g. free tier) must yield 0.0, not None."""
    assert _nano_aiu_to_aiu(0) == 0.0
    assert _nano_aiu_to_aiu(0.0) == 0.0


def test_nano_aiu_to_aiu_none_returns_none() -> None:
    assert _nano_aiu_to_aiu(None) is None


def test_nano_aiu_to_aiu_gate3_sample() -> None:
    """Gate 3 authoritative sample: 100.27622 AI credits from 100276220000 nanoAiu."""
    nano = 100.27622 * 1_000_000_000
    aiu = _nano_aiu_to_aiu(nano)
    assert aiu is not None
    assert abs(aiu - 100.27622) < 1e-4


# ---------------------------------------------------------------------------
# A. Public-list cost — rate constants
# ---------------------------------------------------------------------------


def test_github_ai_credit_rate_is_one_cent() -> None:
    """The public-list rate must be exactly $0.01 per AI credit."""
    assert _GITHUB_AI_CREDIT_RATE_USD == 0.01


def test_github_ai_credit_rate_provenance_fields_are_set() -> None:
    assert _GITHUB_AI_CREDIT_RATE_BASIS == "github-public-ai-credit-list-price"
    assert _GITHUB_AI_CREDIT_RATE_AS_OF == "2026-08-26"
    assert "docs.github.com" in _GITHUB_AI_CREDIT_RATE_SOURCE


# ---------------------------------------------------------------------------
# A. Gate 3 numeric sample: 100.27622 AI credits → 1.0027622 USD
# ---------------------------------------------------------------------------


def test_gate3_cost_from_aiu_sample(tmp_path: Path) -> None:
    """Gate 3 authoritative: AIU 100.27622 → total_cost_usd 1.0027622."""
    agent = _make_agent(tmp_path)
    events = [
        {
            "type": "user.message",
            "data": {"content": "Fix it"},
        },
        {
            "type": "assistant.message",
            "data": {"messageId": "m1", "content": "Done", "outputTokens": 5},
        },
        _shutdown_event(
            nano_aiu=100_276_220_000,  # 100.27622 AI credits
            model_metrics={
                "claude-sonnet-5": {
                    "usage": {
                        "inputTokens": 1816949,
                        "cacheReadTokens": 1621541,
                        "outputTokens": 18996,
                    }
                }
            },
        ),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(1.0027622)
    extra = trajectory.final_metrics.extra
    assert extra is not None
    assert extra["copilot_aiu"] == pytest.approx(100.27622, rel=1e-5)
    assert extra["copilot_cost_rate_usd_per_credit"] == 0.01
    assert extra["copilot_cost_basis"] == "github-public-ai-credit-list-price"
    assert extra["copilot_cost_as_of"] == "2026-08-26"
    assert "docs.github.com" in extra["copilot_cost_source"]


def test_gate3_cost_arithmetic_is_exact() -> None:
    """100.27622 × 0.01 == 1.0027622 with float rounding."""
    assert round(100.27622 * 0.01, 10) == 1.0027622


# ---------------------------------------------------------------------------
# A. Missing AIU → no cost_usd
# ---------------------------------------------------------------------------


def test_cost_usd_is_none_when_aiu_is_missing(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.message",
            "data": {"messageId": "m1", "content": "Done", "outputTokens": 5},
        },
        # No totalNanoAiu in shutdown
        {"type": "session.shutdown", "data": {}},
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_cost_usd is None
    extra = trajectory.final_metrics.extra or {}
    assert "copilot_cost_rate_usd_per_credit" not in extra
    assert "copilot_cost_basis" not in extra


def test_cost_usd_is_none_when_no_shutdown_and_no_checkpoint(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.usage",
            "data": {"apiCallId": "c1", "inputTokens": 100, "outputTokens": 5},
        },
        {
            "type": "assistant.message",
            "data": {"apiCallId": "c1", "content": "Done", "outputTokens": 5},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_cost_usd is None


# ---------------------------------------------------------------------------
# A. Zero AIU → cost 0.0 (not None)
# ---------------------------------------------------------------------------


def test_zero_aiu_produces_zero_cost(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Hi"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=0),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_cost_usd == 0.0
    extra = trajectory.final_metrics.extra
    assert extra is not None
    assert extra["copilot_aiu"] == 0.0
    # Provenance fields must be present even for zero cost
    assert extra["copilot_cost_rate_usd_per_credit"] == 0.01


# ---------------------------------------------------------------------------
# A. Serialization: FinalMetrics round-trips with cost fields
# ---------------------------------------------------------------------------


def test_final_metrics_serializes_total_cost_usd() -> None:
    metrics = FinalMetrics(
        total_prompt_tokens=100,
        total_completion_tokens=5,
        total_cost_usd=1.0027622,
        extra={"copilot_aiu": 100.27622},
    )
    data = json.loads(metrics.model_dump_json())
    restored = FinalMetrics.model_validate(data)
    assert restored.total_cost_usd == pytest.approx(1.0027622)
    assert restored.extra["copilot_aiu"] == pytest.approx(100.27622)


def test_final_metrics_total_cost_usd_none_round_trips() -> None:
    metrics = FinalMetrics(total_prompt_tokens=100, total_cost_usd=None)
    data = json.loads(metrics.model_dump_json())
    restored = FinalMetrics.model_validate(data)
    assert restored.total_cost_usd is None


# ---------------------------------------------------------------------------
# A. Cost aggregation: flows through populate_context_from_final_metrics
# ---------------------------------------------------------------------------


def test_cost_flows_to_agent_context_cost_usd() -> None:
    metrics = FinalMetrics(total_cost_usd=1.0027622)
    context = AgentContext()
    populate_context_from_final_metrics(context, metrics)
    assert context.cost_usd == pytest.approx(1.0027622)


def test_populate_context_cost_usd_none_when_metrics_has_no_cost() -> None:
    metrics = FinalMetrics(total_prompt_tokens=100)
    context = AgentContext()
    populate_context_from_final_metrics(context, metrics)
    assert context.cost_usd is None


# ---------------------------------------------------------------------------
# A. Cost metadata propagated via populate_context_post_run
# ---------------------------------------------------------------------------


def test_populate_context_post_run_propagates_cost_provenance(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Fix it"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "Done"}},
        _shutdown_event(nano_aiu=100_000_000),  # 0.1 AI credits → $0.001
    ]
    _write_events(tmp_path, events)

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.cost_usd == pytest.approx(0.001)
    metadata = context.metadata or {}
    assert metadata["copilot_cost_rate_usd_per_credit"] == 0.01
    assert metadata["copilot_cost_basis"] == "github-public-ai-credit-list-price"
    assert "copilot_cost_source" in metadata
    assert metadata["copilot_cost_as_of"] == "2026-08-26"


# ---------------------------------------------------------------------------
# B. Observed model only — session.start.selectedModel
# ---------------------------------------------------------------------------


def test_observed_model_from_session_start(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "session.start", "data": {"sessionId": "s1", "selectedModel": "claude-sonnet-5"}},
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["copilot_observed_models"] == ["claude-sonnet-5"]
    assert extra["copilot_model_provenance_source"] == "session_events"
    # No requested model — agent has no model_name configured
    assert "copilot_requested_model" not in extra


def test_observed_model_from_assistant_message(tmp_path: Path) -> None:
    """Model observed from assistant.message when no session.start is present."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.message",
            "data": {"messageId": "m1", "model": "gpt-5.5", "content": "OK"},
        },
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert "gpt-5.5" in extra["copilot_observed_models"]


def test_no_observed_models_when_events_carry_none(tmp_path: Path) -> None:
    """When no model field appears in events, copilot_observed_models is absent."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert "copilot_observed_models" not in extra
    assert "copilot_model_provenance_source" not in extra


# ---------------------------------------------------------------------------
# B. Requested vs observed mismatch
# ---------------------------------------------------------------------------


def test_requested_model_recorded_when_it_differs_from_observed(tmp_path: Path) -> None:
    """If --model foo but server used bar, preserve both."""
    agent = CopilotCli(logs_dir=tmp_path, command_model_name="gpt-5.4")
    events = [
        {"type": "session.start", "data": {"sessionId": "s1", "selectedModel": "claude-sonnet-5"}},
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["copilot_observed_models"] == ["claude-sonnet-5"]
    assert extra["copilot_requested_model"] == "gpt-5.4"


def test_no_requested_model_field_when_matches_observed(tmp_path: Path) -> None:
    """If requested model matches the observed one, copilot_requested_model is absent."""
    agent = CopilotCli(logs_dir=tmp_path, command_model_name="claude-sonnet-5")
    events = [
        {"type": "session.start", "data": {"sessionId": "s1", "selectedModel": "claude-sonnet-5"}},
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert "copilot_requested_model" not in extra


# ---------------------------------------------------------------------------
# B. Multiple root models (model changes)
# ---------------------------------------------------------------------------


def test_multiple_root_models_captured_in_order(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "session.start", "data": {"sessionId": "s1", "selectedModel": "model-A"}},
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "session.model_change", "data": {"newModel": "model-B"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["copilot_observed_models"] == ["model-A", "model-B"]


def test_model_deduplication_in_observed_list(tmp_path: Path) -> None:
    """Same model repeated (e.g. multiple assistant messages) is listed once."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "session.start", "data": {"sessionId": "s1", "selectedModel": "model-X"}},
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "model": "model-X", "content": "OK"}},
        {"type": "assistant.message", "data": {"messageId": "m2", "model": "model-X", "content": "More"}},
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["copilot_observed_models"] == ["model-X"]


# ---------------------------------------------------------------------------
# B. Subagent isolation — root observed_models ≠ subagent observed_models
# ---------------------------------------------------------------------------


def test_subagent_model_does_not_appear_in_root_observed_models(tmp_path: Path) -> None:
    """Models from subagent events must not contaminate root observed_models."""
    agent = _make_agent(tmp_path)
    root_session_start = {
        "type": "session.start",
        "data": {"sessionId": "root", "selectedModel": "root-model"},
    }
    subagent_id = "sub-001"
    events = [
        root_session_start,
        {"type": "user.message", "data": {"content": "Delegate"}},
        {
            "type": "assistant.message",
            "data": {
                "messageId": "m1",
                "content": "OK",
                "toolRequests": [{"toolCallId": subagent_id, "name": "delegate", "arguments": {}}],
            },
        },
        # Subagent start event
        {
            "type": "subagent.started",
            "agentId": subagent_id,
            "data": {"agentName": "sub", "model": "subagent-model"},
        },
        # Subagent's session.start (carries agentId tag)
        {
            "type": "session.start",
            "agentId": subagent_id,
            "data": {"sessionId": "subroot", "selectedModel": "subagent-model"},
        },
        {
            "type": "assistant.message",
            "agentId": subagent_id,
            "data": {"messageId": "sm1", "model": "subagent-model", "content": "Done"},
        },
        {
            "type": "subagent.completed",
            "agentId": subagent_id,
            "data": {"agentName": "sub", "status": "completed"},
        },
        _shutdown_event(nano_aiu=1_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    # Root observed_models must only contain the root model
    assert extra.get("copilot_observed_models") == ["root-model"]
    # Subagent model must not bleed into root
    assert "subagent-model" not in extra.get("copilot_observed_models", [])


# ---------------------------------------------------------------------------
# B. Trial boundary refresh: _maybe_populate_agent_context updates agent_info
# ---------------------------------------------------------------------------


def test_maybe_populate_agent_context_sets_model_info(tmp_path: Path) -> None:
    """After populate_context_post_run, agent_info.model_info must be set."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    assert result.agent_info.model_info is None

    stub_agent = _StubCopilotAgent(tmp_path, observed_models=["claude-sonnet-5"])
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert result.agent_info.model_info is not None
    assert result.agent_info.model_info.name == "claude-sonnet-5"
    assert result.agent_info.model_info.provider is None


def test_maybe_populate_agent_context_preserves_cli_version(tmp_path: Path) -> None:
    """AgentInfo.version (CLI version) must not be changed by model refresh."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    original_version = result.agent_info.version

    stub_agent = _StubCopilotAgent(tmp_path, observed_models=["claude-sonnet-5"])
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert result.agent_info.version == original_version


def test_maybe_populate_agent_context_no_model_when_no_observed(tmp_path: Path) -> None:
    """When no observed models in metadata, model_info stays None."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)

    stub_agent = _StubCopilotAgent(tmp_path, observed_models=[])  # empty
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert result.agent_info.model_info is None


def test_maybe_populate_agent_context_result_none_guard(tmp_path: Path) -> None:
    """When _result is None, no model_info update attempted."""
    stub_agent = _StubCopilotAgent(tmp_path, observed_models=["some-model"])
    trial = _fake_trial(tmp_path, result=None, agent=stub_agent)

    # Must not raise
    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)


def test_trial_boundary_model_info_reaches_result_json(tmp_path: Path) -> None:
    """model_info set by _maybe_populate_agent_context must be present in result dump."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    stub_agent = _StubCopilotAgent(tmp_path, observed_models=["claude-sonnet-5"])
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    # Simulate what _cleanup_and_finalize does — serialize result
    dumped = json.loads(result.model_dump_json())
    model_info = dumped["agent_info"]["model_info"]
    assert model_info is not None
    assert model_info["name"] == "claude-sonnet-5"
    assert model_info["provider"] is None


def test_populate_context_post_run_stores_observed_models_in_metadata(
    tmp_path: Path,
) -> None:
    """populate_context_post_run must put copilot_observed_models in metadata."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "session.start", "data": {"sessionId": "s1", "selectedModel": "model-Z"}},
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=1_000_000),
    ]
    _write_events(tmp_path, events)

    context = AgentContext()
    agent.populate_context_post_run(context)

    metadata = context.metadata or {}
    assert metadata.get("copilot_observed_models") == ["model-Z"]
    assert metadata.get("copilot_model_provenance_source") == "session_events"


# ---------------------------------------------------------------------------
# C. tokenDetails canonical semantics
# ---------------------------------------------------------------------------


def test_token_accounting_source_is_token_details(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            nano_aiu=1_000_000,
            token_details={
                "input": {"tokenCount": 1000},
                "cache_read": {"tokenCount": 200},
                "cache_write": {"tokenCount": 100},
                "output": {"tokenCount": 50},
            },
        ),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["token_accounting_source"] == "tokenDetails"
    assert extra["input_includes_cache"] is True
    # Canonical total = 1000 + 200 + 100 = 1300
    assert trajectory.final_metrics.total_prompt_tokens == 1300
    assert trajectory.final_metrics.total_cached_tokens == 200
    assert extra["cache_write_tokens"] == 100


def test_token_accounting_source_is_model_metrics(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            nano_aiu=1_000_000,
            model_metrics={
                "gpt-5.4": {
                    "usage": {
                        "inputTokens": 100,
                        "cacheReadTokens": 20,
                        "cacheWriteTokens": 10,
                        "outputTokens": 5,
                    }
                }
            },
        ),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["token_accounting_source"] == "modelMetrics"
    assert extra["input_includes_cache"] is True


def test_token_accounting_source_is_assistant_usage(tmp_path: Path) -> None:
    """No-shutdown path: reconciled assistant.usage events."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.usage",
            "data": {
                "apiCallId": "c1",
                "inputTokens": 100,
                "cacheReadTokens": 20,
                "cacheWriteTokens": 5,
                "outputTokens": 10,
            },
        },
        {
            "type": "assistant.message",
            "data": {"apiCallId": "c1", "content": "OK"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["token_accounting_source"] == "assistant_usage"
    assert extra["input_includes_cache"] is True


# ---------------------------------------------------------------------------
# C. Cache-write capture
# ---------------------------------------------------------------------------


def test_cache_write_tokens_in_extra_from_token_details(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            token_details={
                "input": {"tokenCount": 500},
                "cache_read": {"tokenCount": 100},
                "cache_write": {"tokenCount": 50},
                "output": {"tokenCount": 25},
            },
        ),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["cache_write_tokens"] == 50
    # cache_read is in total_cached_tokens (n_cache_tokens compatibility)
    assert trajectory.final_metrics.total_cached_tokens == 100


def test_cache_write_tokens_in_metadata_via_populate_context_post_run(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            nano_aiu=1_000_000,
            token_details={
                "input": {"tokenCount": 500},
                "cache_read": {"tokenCount": 100},
                "cache_write": {"tokenCount": 42},
                "output": {"tokenCount": 25},
            },
        ),
    ]
    _write_events(tmp_path, events)

    context = AgentContext()
    agent.populate_context_post_run(context)

    metadata = context.metadata or {}
    assert metadata["cache_write_tokens"] == 42
    # n_cache_tokens = cache_read (backward compat)
    assert context.n_cache_tokens == 100


def test_cache_write_from_assistant_usage_events(tmp_path: Path) -> None:
    """cache_write_tokens captured from assistant.usage in no-shutdown path."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.usage",
            "data": {
                "apiCallId": "c1",
                "inputTokens": 100,
                "cacheReadTokens": 20,
                "cacheWriteTokens": 15,
                "outputTokens": 10,
            },
        },
        {
            "type": "assistant.message",
            "data": {"apiCallId": "c1", "content": "OK"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["cache_write_tokens"] == 15
    # cache_read is carried in total_cached_tokens
    assert trajectory.final_metrics.total_cached_tokens == 20


# ---------------------------------------------------------------------------
# C. modelMetrics fallback/delta (alternative totals when tokenDetails canonical)
# ---------------------------------------------------------------------------


def test_model_metrics_alternative_recorded_when_token_details_wins(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            nano_aiu=1_000_000,
            token_details={
                "input": {"tokenCount": 1000},
                "cache_read": {"tokenCount": 200},
                "cache_write": {"tokenCount": 100},
                "output": {"tokenCount": 50},
            },
            model_metrics={
                "model-x": {
                    "usage": {
                        "inputTokens": 800,
                        "cacheReadTokens": 160,
                        "cacheWriteTokens": 80,
                        "outputTokens": 40,
                    }
                }
            },
        ),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    # Canonical source
    assert extra["token_accounting_source"] == "tokenDetails"
    assert trajectory.final_metrics.total_prompt_tokens == 1300  # tokenDetails wins
    # modelMetrics alternative preserved for delta visibility
    assert extra["copilot_mm_input_tokens"] == 800
    assert extra["copilot_mm_cache_read_tokens"] == 160
    assert extra["copilot_mm_cache_write_tokens"] == 80
    assert extra["copilot_mm_output_tokens"] == 40


def test_model_metrics_alternative_absent_when_only_model_metrics(
    tmp_path: Path,
) -> None:
    """When only modelMetrics (no tokenDetails), no mm_* alternative fields."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            model_metrics={
                "model-x": {
                    "usage": {
                        "inputTokens": 100,
                        "cacheReadTokens": 20,
                        "outputTokens": 5,
                    }
                }
            },
        ),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["token_accounting_source"] == "modelMetrics"
    assert "copilot_mm_input_tokens" not in extra


# ---------------------------------------------------------------------------
# C. input_includes_cache and provenance documented
# ---------------------------------------------------------------------------


def test_input_includes_cache_not_set_without_token_source(tmp_path: Path) -> None:
    """When no billing source available, input_includes_cache not asserted."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.message",
            "data": {"messageId": "m1", "content": "OK", "outputTokens": 5},
        },
        # No shutdown, no usage events — message input only
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    # source is "assistant_message" — input_includes_cache semantics unclear, not set
    assert "input_includes_cache" not in extra


def test_token_accounting_source_in_metadata(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(
            nano_aiu=1_000_000,
            token_details={
                "input": {"tokenCount": 100},
                "cache_read": {"tokenCount": 20},
                "cache_write": {"tokenCount": 10},
                "output": {"tokenCount": 5},
            },
        ),
    ]
    _write_events(tmp_path, events)

    context = AgentContext()
    agent.populate_context_post_run(context)

    metadata = context.metadata or {}
    assert metadata["token_accounting_source"] == "tokenDetails"
    assert metadata["input_includes_cache"] is True


# ---------------------------------------------------------------------------
# Legacy compatibility — no regressions
# ---------------------------------------------------------------------------


def test_legacy_session_without_aiu_or_model(tmp_path: Path) -> None:
    """Sessions lacking AIU and model evidence must still produce a valid trajectory."""
    agent = _make_agent(tmp_path, model_name="gpt-5.4")
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {
            "type": "assistant.message",
            "data": {"messageId": "m1", "content": "Done", "outputTokens": 5},
        },
        {"type": "session.shutdown", "data": {}},
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_cost_usd is None
    extra = trajectory.final_metrics.extra or {}
    assert "copilot_aiu" not in extra
    assert "copilot_cost_rate_usd_per_credit" not in extra
    assert "copilot_observed_models" not in extra


def test_copilot_aiu_key_preserved_for_backward_compat(tmp_path: Path) -> None:
    """copilot_aiu must still be present when AIU is available."""
    agent = _make_agent(tmp_path)
    events = [
        {"type": "user.message", "data": {"content": "Go"}},
        {"type": "assistant.message", "data": {"messageId": "m1", "content": "OK"}},
        _shutdown_event(nano_aiu=500_000_000),
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    extra = trajectory.final_metrics.extra or {}
    assert extra["copilot_aiu"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# B. Provider preservation — focused Fact-Checker boundary tests
# ---------------------------------------------------------------------------


def test_model_provider_preserved_when_observed_matches_configured_bare_name(
    tmp_path: Path,
) -> None:
    """requested 'github/claude-sonnet-5', observed 'claude-sonnet-5' → provider='github'."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    stub_agent = _StubCopilotAgent(
        tmp_path,
        model_name="github/claude-sonnet-5",
        observed_models=["claude-sonnet-5"],
    )
    # Simulate what Trial.run() does after _setup_agent(): record configured model info.
    result.agent_info = stub_agent.to_agent_info()
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert result.agent_info.model_info is not None
    assert result.agent_info.model_info.name == "claude-sonnet-5"
    # Explicit provider from 'github/claude-sonnet-5' spec is preserved
    assert result.agent_info.model_info.provider == "github"
    # CLI version must not be touched
    assert result.agent_info.version == "1.0.80"


def test_model_provider_not_transferred_when_observed_differs_from_configured(
    tmp_path: Path,
) -> None:
    """requested 'github/gpt-5.4', observed 'claude-sonnet-5' → provider=None, mismatch retained."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    stub_agent = _StubCopilotAgent(
        tmp_path,
        model_name="github/gpt-5.4",
        observed_models=["claude-sonnet-5"],  # server selected a different model
    )
    # Simulate to_agent_info() → model_info.name="gpt-5.4", provider="github"
    result.agent_info = stub_agent.to_agent_info()
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    # Simulate metadata already populated (is_empty=False to skip populate_context_post_run)
    agent_result = AgentContext(
        metadata={
            "copilot_observed_models": ["claude-sonnet-5"],
            "copilot_requested_model": "gpt-5.4",  # mismatch marker from task B
        }
    )
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert result.agent_info.model_info is not None
    assert result.agent_info.model_info.name == "claude-sonnet-5"
    # 'github' provider from 'gpt-5.4' spec must NOT be applied to a different model
    assert result.agent_info.model_info.provider is None
    # Mismatch metadata preserved unchanged
    assert (agent_result.metadata or {})["copilot_requested_model"] == "gpt-5.4"


def test_model_provider_is_none_without_provider_prefix_in_configured_spec(
    tmp_path: Path,
) -> None:
    """No requested provider → provider=None, even when observed name matches configured."""
    config = _trial_config(tmp_path)
    result = _trial_result(config, tmp_path)
    stub_agent = _StubCopilotAgent(
        tmp_path,
        model_name="claude-sonnet-5",  # no 'provider/' prefix
        observed_models=["claude-sonnet-5"],
    )
    result.agent_info = stub_agent.to_agent_info()
    trial = _fake_trial(tmp_path, result=result, agent=stub_agent)

    agent_result = AgentContext()
    Trial._maybe_populate_agent_context(trial, agent_result)

    assert result.agent_info.model_info is not None
    assert result.agent_info.model_info.name == "claude-sonnet-5"
    assert result.agent_info.model_info.provider is None
