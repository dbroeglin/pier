"""Tests for Copilot CLI raw artifact retention and validation (BLK-3 / VAL-005).

Covers:
- present-valid: file exists with parseable JSONL → "valid" + SHA-256
- absent-not-produced: file absent, download OK → "not-produced"
- download-failure: file absent, download failed → "download-failure"
- corrupt JSONL: file present but no parseable lines → "corrupt" + SHA-256
- multiple native session paths: mtime / session-id disambiguation
- checksum stability: identical content → identical SHA-256
- non-Copilot agents: notify_download_complete is a no-op on base class
"""

import hashlib
import json
from pathlib import Path

from pier.agents.installed.base import BaseInstalledAgent
from pier.agents.installed.copilot_cli import (
    CopilotCli,
    RawArtifactResult,
    RawArtifactStatus,
    _sha256_hex,
    _validate_raw_artifact,
    validate_copilot_raw_artifacts,
)
from pier.models.agent.context import AgentContext


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


def _minimal_event() -> dict:
    return {
        "type": "assistant.message",
        "timestamp": "2026-01-01T00:00:01.000Z",
        "data": {"apiCallId": "api-1", "content": "hello"},
    }


# ── _sha256_hex ───────────────────────────────────────────────────────────────


def test_sha256_hex_matches_standard_library(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert _sha256_hex(p) == expected


def test_sha256_hex_is_stable_across_calls(tmp_path: Path) -> None:
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(_minimal_event()) + "\n", encoding="utf-8")
    assert _sha256_hex(p) == _sha256_hex(p)


# ── _validate_raw_artifact ────────────────────────────────────────────────────


def test_validate_present_valid(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    _write_events(p, [_minimal_event()])

    status, sha256 = _validate_raw_artifact(p, download_ok=None)

    assert status == "valid"
    assert sha256 == _sha256_hex(p)


def test_validate_present_corrupt_all_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text("not json\nalso not json\n", encoding="utf-8")

    status, sha256 = _validate_raw_artifact(p, download_ok=None)

    assert status == "corrupt"
    assert sha256 == _sha256_hex(p)


def test_validate_present_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text("", encoding="utf-8")

    status, sha256 = _validate_raw_artifact(p, download_ok=None)

    assert status == "corrupt"
    assert sha256 is not None  # empty file still has a deterministic hash


def test_validate_present_mixed_lines_counts_as_valid(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text(
        "not json\n" + json.dumps(_minimal_event()) + "\n",
        encoding="utf-8",
    )

    status, sha256 = _validate_raw_artifact(p, download_ok=None)

    assert status == "valid"
    assert sha256 is not None


def test_validate_absent_no_download_context(tmp_path: Path) -> None:
    status, sha256 = _validate_raw_artifact(
        tmp_path / "nonexistent.jsonl", download_ok=None
    )
    assert status is None
    assert sha256 is None


def test_validate_absent_download_ok_is_not_produced(tmp_path: Path) -> None:
    status, sha256 = _validate_raw_artifact(
        tmp_path / "nonexistent.jsonl", download_ok=True
    )
    assert status == "not-produced"
    assert sha256 is None


def test_validate_absent_download_failed_is_download_failure(tmp_path: Path) -> None:
    status, sha256 = _validate_raw_artifact(
        tmp_path / "nonexistent.jsonl", download_ok=False
    )
    assert status == "download-failure"
    assert sha256 is None


def test_validate_none_path_with_download_failure(tmp_path: Path) -> None:
    status, sha256 = _validate_raw_artifact(None, download_ok=False)
    assert status == "download-failure"
    assert sha256 is None


def test_validate_none_path_no_download_context() -> None:
    status, sha256 = _validate_raw_artifact(None, download_ok=None)
    assert status is None
    assert sha256 is None


# ── validate_copilot_raw_artifacts ────────────────────────────────────────────


def test_validate_both_files_present_and_valid(tmp_path: Path) -> None:
    jsonl = tmp_path / "copilot-cli.jsonl"
    _write_events(jsonl, [_minimal_event()])

    session_dir = tmp_path / "copilot-home" / "session-state" / "sess-1"
    events = session_dir / "events.jsonl"
    _write_events(events, [_minimal_event()])

    result = validate_copilot_raw_artifacts(tmp_path, session_id="sess-1")

    assert result.jsonl_status == "valid"
    assert result.jsonl_sha256 == _sha256_hex(jsonl)
    assert result.events_status == "valid"
    assert result.events_sha256 == _sha256_hex(events)


def test_validate_both_absent_no_download_context(tmp_path: Path) -> None:
    result = validate_copilot_raw_artifacts(tmp_path)

    assert result.jsonl_status is None
    assert result.jsonl_sha256 is None
    assert result.events_status is None
    assert result.events_sha256 is None


def test_validate_both_absent_download_ok(tmp_path: Path) -> None:
    result = validate_copilot_raw_artifacts(tmp_path, download_ok=True)

    assert result.jsonl_status == "not-produced"
    assert result.events_status == "not-produced"


def test_validate_both_absent_download_failed(tmp_path: Path) -> None:
    result = validate_copilot_raw_artifacts(tmp_path, download_ok=False)

    assert result.jsonl_status == "download-failure"
    assert result.events_status == "download-failure"


def test_validate_jsonl_corrupt_events_valid(tmp_path: Path) -> None:
    jsonl = tmp_path / "copilot-cli.jsonl"
    jsonl.write_text("{{{{garbage\n", encoding="utf-8")

    session_dir = tmp_path / "copilot-home" / "session-state" / "sess-1"
    events = session_dir / "events.jsonl"
    _write_events(events, [_minimal_event()])

    result = validate_copilot_raw_artifacts(tmp_path, session_id="sess-1")

    assert result.jsonl_status == "corrupt"
    assert result.jsonl_sha256 is not None  # hash of corrupt file
    assert result.events_status == "valid"
    assert result.events_sha256 is not None


def test_validate_jsonl_present_events_absent_download_ok(tmp_path: Path) -> None:
    jsonl = tmp_path / "copilot-cli.jsonl"
    _write_events(jsonl, [_minimal_event()])

    result = validate_copilot_raw_artifacts(tmp_path, download_ok=True)

    assert result.jsonl_status == "valid"
    assert result.events_status == "not-produced"


def test_validate_jsonl_absent_events_present(tmp_path: Path) -> None:
    session_dir = tmp_path / "copilot-home" / "session-state" / "sess-1"
    _write_events(session_dir / "events.jsonl", [_minimal_event()])

    result = validate_copilot_raw_artifacts(tmp_path, session_id="sess-1", download_ok=True)

    assert result.jsonl_status == "not-produced"
    assert result.events_status == "valid"


# ── Multiple native session paths (find_copilot_session_events via validate) ─


def test_validate_picks_session_by_id_over_newer_mtime(tmp_path: Path) -> None:
    """Explicit session_id match takes priority over mtime ordering."""
    import os

    root = tmp_path / "copilot-home" / "session-state"
    wanted = root / "wanted" / "events.jsonl"
    _write_events(wanted, [_minimal_event()])

    newer = root / "newer" / "events.jsonl"
    _write_events(newer, [_minimal_event()])
    # Make "newer" look newer so mtime fallback would pick it
    os.utime(newer, (2_000_000_000, 2_000_000_000))

    result = validate_copilot_raw_artifacts(tmp_path, session_id="wanted")

    assert result.events_status == "valid"
    assert result.events_sha256 == _sha256_hex(wanted)


def test_validate_falls_back_to_mtime_when_no_session_id(tmp_path: Path) -> None:
    import os

    root = tmp_path / "copilot-home" / "session-state"
    older = root / "older" / "events.jsonl"
    _write_events(older, [{"type": "old"}])
    os.utime(older, (1_000_000_000, 1_000_000_000))

    newer = root / "newer" / "events.jsonl"
    _write_events(newer, [_minimal_event()])
    os.utime(newer, (2_000_000_000, 2_000_000_000))

    result = validate_copilot_raw_artifacts(tmp_path)

    # Should pick the newer file
    assert result.events_status == "valid"
    assert result.events_sha256 == _sha256_hex(newer)


# ── Checksum stability ────────────────────────────────────────────────────────


def test_checksum_is_stable_across_identical_files(tmp_path: Path) -> None:
    content = json.dumps(_minimal_event()) + "\n"
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(content, encoding="utf-8")
    p2.write_text(content, encoding="utf-8")

    assert _sha256_hex(p1) == _sha256_hex(p2)


def test_checksum_differs_for_different_content(tmp_path: Path) -> None:
    e1 = {"type": "assistant.message", "data": {"content": "A"}}
    e2 = {"type": "assistant.message", "data": {"content": "B"}}
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(e1) + "\n", encoding="utf-8")
    p2.write_text(json.dumps(e2) + "\n", encoding="utf-8")

    assert _sha256_hex(p1) != _sha256_hex(p2)


def test_repeated_validate_call_returns_same_sha256(tmp_path: Path) -> None:
    jsonl = tmp_path / "copilot-cli.jsonl"
    _write_events(jsonl, [_minimal_event()])

    r1 = validate_copilot_raw_artifacts(tmp_path)
    r2 = validate_copilot_raw_artifacts(tmp_path)

    assert r1.jsonl_sha256 == r2.jsonl_sha256
    assert r1.jsonl_sha256 is not None


# ── RawArtifactResult.to_metadata_dict ───────────────────────────────────────


def test_to_metadata_dict_valid_includes_sha256() -> None:
    result = RawArtifactResult(
        jsonl_status="valid",
        jsonl_sha256="abc123",
        events_status="valid",
        events_sha256="def456",
    )
    d = result.to_metadata_dict()
    assert d["copilot_cli_jsonl_status"] == "valid"
    assert d["copilot_cli_jsonl_sha256"] == "abc123"
    assert d["copilot_events_jsonl_status"] == "valid"
    assert d["copilot_events_jsonl_sha256"] == "def456"


def test_to_metadata_dict_not_produced_no_sha256() -> None:
    result = RawArtifactResult(
        jsonl_status="not-produced",
        jsonl_sha256=None,
        events_status="download-failure",
        events_sha256=None,
    )
    d = result.to_metadata_dict()
    assert d["copilot_cli_jsonl_status"] == "not-produced"
    assert "copilot_cli_jsonl_sha256" not in d
    assert d["copilot_events_jsonl_status"] == "download-failure"
    assert "copilot_events_jsonl_sha256" not in d


def test_to_metadata_dict_none_status_omitted() -> None:
    result = RawArtifactResult(
        jsonl_status=None,
        jsonl_sha256=None,
        events_status=None,
        events_sha256=None,
    )
    assert result.to_metadata_dict() == {}


def test_to_metadata_dict_corrupt_includes_sha256() -> None:
    result = RawArtifactResult(
        jsonl_status="corrupt",
        jsonl_sha256="deadbeef",
        events_status="corrupt",
        events_sha256="cafebabe",
    )
    d = result.to_metadata_dict()
    assert d["copilot_cli_jsonl_status"] == "corrupt"
    assert d["copilot_cli_jsonl_sha256"] == "deadbeef"
    assert d["copilot_events_jsonl_status"] == "corrupt"
    assert d["copilot_events_jsonl_sha256"] == "cafebabe"


# ── CopilotCli.populate_context_post_run integration ─────────────────────────


def test_populate_context_records_valid_status_when_files_present(
    tmp_path: Path,
) -> None:
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    agent.notify_download_complete(ok=True)

    jsonl = tmp_path / "copilot-cli.jsonl"
    _write_events(jsonl, [_minimal_event()])
    session_dir = tmp_path / "copilot-home" / "session-state" / "sess-1"
    events_file = session_dir / "events.jsonl"
    _write_events(events_file, [_minimal_event()])

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["copilot_cli_jsonl_status"] == "valid"
    assert "copilot_cli_jsonl_sha256" in context.metadata
    assert context.metadata["copilot_events_jsonl_status"] == "valid"
    assert "copilot_events_jsonl_sha256" in context.metadata


def test_populate_context_records_not_produced_when_files_absent_and_download_ok(
    tmp_path: Path,
) -> None:
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    agent.notify_download_complete(ok=True)

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["copilot_cli_jsonl_status"] == "not-produced"
    assert context.metadata["copilot_events_jsonl_status"] == "not-produced"
    assert "copilot_cli_jsonl_sha256" not in context.metadata


def test_populate_context_records_download_failure_when_download_failed(
    tmp_path: Path,
) -> None:
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    agent.notify_download_complete(ok=False)

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["copilot_cli_jsonl_status"] == "download-failure"
    assert context.metadata["copilot_events_jsonl_status"] == "download-failure"


def test_populate_context_records_corrupt_for_unreadable_jsonl(
    tmp_path: Path,
) -> None:
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    agent.notify_download_complete(ok=True)

    # Write a file with content that is valid bytes but zero parseable JSON objects
    (tmp_path / "copilot-cli.jsonl").write_text(
        "not json at all\n{{broken\n", encoding="utf-8"
    )

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["copilot_cli_jsonl_status"] == "corrupt"
    # SHA-256 is still recorded for corrupt files
    assert "copilot_cli_jsonl_sha256" in context.metadata


def test_populate_context_distinguishes_corrupt_jsonl_from_not_produced_events(
    tmp_path: Path,
) -> None:
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    agent.notify_download_complete(ok=True)

    # JSONL is corrupt; no native events dir written
    (tmp_path / "copilot-cli.jsonl").write_text(
        "total garbage\n", encoding="utf-8"
    )

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["copilot_cli_jsonl_status"] == "corrupt"
    assert context.metadata["copilot_events_jsonl_status"] == "not-produced"


# ── Non-Copilot agents are unaffected ─────────────────────────────────────────


def test_notify_download_complete_is_a_noop_on_base(tmp_path: Path) -> None:
    """BaseInstalledAgent.notify_download_complete must not raise."""
    # Use a minimal concrete subclass to exercise the default implementation
    class _MinimalAgent(BaseInstalledAgent):
        @staticmethod
        def name() -> str:
            return "minimal"

        def version(self) -> str | None:
            return None

        def install_spec(self):
            return None

        async def setup(self, environment) -> None:
            pass

        async def run(self, instruction, environment, context) -> None:
            pass

        def populate_context_post_run(self, context) -> None:
            pass

    agent = _MinimalAgent(logs_dir=tmp_path)
    # Must not raise and must not modify any agent state (no _download_ok attribute)
    agent.notify_download_complete(ok=False)
    agent.notify_download_complete(ok=True)
    assert not hasattr(agent, "_download_ok")
