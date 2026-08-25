from pathlib import Path

from atif_assertions import assert_one_step_per_api_call, assert_valid_atif
from pier.agents.installed.copilot_cli import CopilotCli, _read_jsonl

_FIXTURES = Path(__file__).parent / "fixtures" / "copilot_cli"


def _convert_fixture(name: str, tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / f"{name}.jsonl")
    assert events, f"fixture {name} is empty"
    return agent._convert_events_to_trajectory(events, metrics_events=events)


def test_val_006_session_truncation_reports_pre_truncation_peak(tmp_path: Path):
    trajectory = _convert_fixture("session_truncation", tmp_path)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.final_metrics is not None
    # The truncation event carries the largest pre-limit context window and must
    # win over both shutdown.currentTokens and compaction peaks.
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 7700
    # Only successful compactions count as summarizations.
    assert trajectory.final_metrics.extra["summarization_count"] == 1
    assert [step.source for step in trajectory.steps] == ["user", "agent", "system", "agent"]


def test_gc_3_session_error_fixture_preserves_failure_shape(tmp_path: Path):
    trajectory = _convert_fixture("session_error", tmp_path)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["session_errors"] == [
        {
            "errorType": "AuthError",
            "message": "401 Unauthorized",
            "code": "E401",
            "statusCode": 401,
            "source": "network",
            "fatal": True,
        }
    ]


def test_gc_6_unknown_event_fixture_is_forward_compatible(tmp_path: Path):
    trajectory = _convert_fixture("session_unknown_event", tmp_path)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    # Unknown events are ignored; known events still form the expected steps.
    assert [step.source for step in trajectory.steps] == ["user", "agent"]
