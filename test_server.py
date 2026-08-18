"""Offline unit tests for the pure logic in server.py.

These use temp fixtures and never invoke agy, so they cost no AI Pro quota and
can run anywhere (including CI). For the live end-to-end check, see
test_smoke.py instead.

    pytest test_server.py
"""

import json
import os
import subprocess
import time

import pytest

import server

# --------------------------------------------------------------------------
# _normalize_workspace
# --------------------------------------------------------------------------


def test_normalize_workspace_none_returns_cwd():
    assert server._normalize_workspace(None) == os.getcwd()


def test_normalize_workspace_relative_is_absolutised(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert server._normalize_workspace("sub/dir") == os.path.abspath("sub/dir")


# --------------------------------------------------------------------------
# _read_last_conv_id
# --------------------------------------------------------------------------


@pytest.fixture
def last_conv_file(tmp_path, monkeypatch):
    f = tmp_path / "last_conversations.json"
    monkeypatch.setattr(server, "LAST_CONVERSATIONS", f)
    return f


def test_read_last_conv_id_exact_match(last_conv_file):
    last_conv_file.write_text(json.dumps({"C:\\proj": "conv-1"}), encoding="utf-8")
    assert server._read_last_conv_id("C:\\proj") == "conv-1"


def test_read_last_conv_id_is_case_insensitive(last_conv_file):
    last_conv_file.write_text(json.dumps({"C:\\Proj": "conv-2"}), encoding="utf-8")
    assert server._read_last_conv_id("c:\\proj") == "conv-2"


def test_read_last_conv_id_missing_file_returns_none(last_conv_file):
    assert server._read_last_conv_id("anything") is None


def test_read_last_conv_id_malformed_json_returns_none(last_conv_file):
    last_conv_file.write_text("{not valid json", encoding="utf-8")
    assert server._read_last_conv_id("x") is None


def test_read_last_conv_id_absent_key_returns_none(last_conv_file):
    last_conv_file.write_text(json.dumps({"other": "c"}), encoding="utf-8")
    assert server._read_last_conv_id("missing") is None


# --------------------------------------------------------------------------
# _find_newest_conv_after
# --------------------------------------------------------------------------


@pytest.fixture
def brain_dir(tmp_path, monkeypatch):
    d = tmp_path / "brain"
    d.mkdir()
    monkeypatch.setattr(server, "BRAIN_DIR", d)
    return d


def test_find_newest_conv_after_missing_brain_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "BRAIN_DIR", tmp_path / "does-not-exist")
    assert server._find_newest_conv_after(time.time()) is None


def test_find_newest_conv_after_picks_newest_dir(brain_dir):
    start = time.time()
    old = brain_dir / "old-conv"
    old.mkdir()
    new = brain_dir / "new-conv"
    new.mkdir()
    os.utime(old, (start - 100, start - 100))
    os.utime(new, (start + 5, start + 5))
    assert server._find_newest_conv_after(start) == "new-conv"


def test_find_newest_conv_after_ignores_plain_files(brain_dir):
    start = time.time()
    f = brain_dir / "a-file"
    f.write_text("x", encoding="utf-8")
    os.utime(f, (start + 5, start + 5))
    assert server._find_newest_conv_after(start) is None


def test_find_newest_conv_after_skips_dirs_older_than_start(brain_dir):
    start = time.time()
    stale = brain_dir / "stale"
    stale.mkdir()
    os.utime(stale, (start - 100, start - 100))
    assert server._find_newest_conv_after(start) is None


# --------------------------------------------------------------------------
# _read_response
# --------------------------------------------------------------------------


def _entry(type_, content=None, status="DONE", source="MODEL", tool_calls=None):
    e = {"source": source, "status": status, "type": type_}
    if content is not None:
        e["content"] = content
    if tool_calls is not None:
        e["tool_calls"] = tool_calls
    return json.dumps(e)


def _write_transcript(brain_dir, conv_id, lines):
    logs = brain_dir / conv_id / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "transcript.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _view_file_call(path, start=None, end=None):
    args = {
        "AbsolutePath": f'"{path}"',
        "toolAction": '"Viewing file"',
        "toolSummary": '"View file"',
    }
    if start is not None:
        args["StartLine"] = str(start)
    if end is not None:
        args["EndLine"] = str(end)
    return [{"name": "view_file", "args": args}]


def _fixture_content_only():
    return [
        _entry("USER_INPUT", "hello", source="USER_EXPLICIT"),
        _entry("CONVERSATION_HISTORY", source="SYSTEM"),
        _entry("PLANNER_RESPONSE", "plain answer"),
    ]


def _fixture_tool_then_content():
    return [
        _entry("USER_INPUT", "search then answer", source="USER_EXPLICIT"),
        _entry("CONVERSATION_HISTORY", source="SYSTEM"),
        _entry(
            "PLANNER_RESPONSE",
            tool_calls=[{"name": "grep_search", "args": {"Query": '"pm.py"'}}],
        ),
        _entry("GREP_SEARCH", "No results found"),
        _entry("CHECKPOINT", "truncated"),
        _entry("PLANNER_RESPONSE", "final after tool"),
    ]


def _fixture_tool_only_terminal(path="/Users/zvisegal/devlope/TaideAi/src/taide/v2/pm.py"):
    return [
        _entry("USER_INPUT", "read the file", source="USER_EXPLICIT"),
        _entry("CONVERSATION_HISTORY", source="SYSTEM"),
        _entry("PLANNER_RESPONSE", tool_calls=_view_file_call(path, start=1, end=800)),
    ]


def test_read_response_returns_last_planner_response_with_content(brain_dir):
    _write_transcript(
        brain_dir,
        "c1",
        [
            _entry("RUN_COMMAND", "step"),
            _entry("PLANNER_RESPONSE", "first"),
            _entry("PLANNER_RESPONSE", "final"),
        ],
    )
    assert server._read_response("c1") == "final"


def test_read_response_ignores_contentless_and_malformed_lines(brain_dir):
    _write_transcript(
        brain_dir,
        "c2",
        [
            "{ broken json",
            "",
            _entry("PLANNER_RESPONSE"),  # no content
            _entry("PLANNER_RESPONSE", "answer"),
        ],
    )
    assert server._read_response("c2") == "answer"


def test_read_response_no_completed_response_raises(brain_dir):
    _write_transcript(
        brain_dir,
        "c3",
        [
            _entry("PLANNER_RESPONSE", "x", status="RUNNING"),
        ],
    )
    with pytest.raises(RuntimeError, match="No completed MODEL response"):
        server._read_response("c3")


def test_read_response_missing_transcript_no_db_mentions_sqlite(brain_dir):
    (brain_dir / "c4").mkdir()
    with pytest.raises(RuntimeError, match="SQLite"):
        server._read_response("c4")


def test_read_response_missing_transcript_with_db_points_at_db(brain_dir):
    conv_dir = brain_dir / "c5"
    conv_dir.mkdir()
    (conv_dir / "conversation.db").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"conversation\.db"):
        server._read_response("c5")


# --------------------------------------------------------------------------
# transcript parser — three real-world shapes
# --------------------------------------------------------------------------


def test_parse_content_only_fixture(brain_dir):
    path = _write_transcript(brain_dir, "shape-content", _fixture_content_only())
    content, pending = server._parse_planner_transcript(path.read_text(encoding="utf-8"))
    assert content == "plain answer"
    assert pending == []
    assert server._read_response("shape-content") == "plain answer"


def test_parse_tool_call_then_content_fixture(brain_dir):
    path = _write_transcript(brain_dir, "shape-tool-then", _fixture_tool_then_content())
    content, pending = server._parse_planner_transcript(path.read_text(encoding="utf-8"))
    assert content == "final after tool"
    assert pending == []
    assert server._read_response("shape-tool-then") == "final after tool"


def test_parse_tool_call_only_terminal_fixture(brain_dir):
    target = "/Users/zvisegal/devlope/TaideAi/src/taide/v2/pm.py"
    path = _write_transcript(
        brain_dir, "shape-tool-only", _fixture_tool_only_terminal(target)
    )
    content, pending = server._parse_planner_transcript(path.read_text(encoding="utf-8"))
    assert content is None
    assert pending and pending[0]["name"] == "view_file"
    with pytest.raises(RuntimeError, match="unexecuted tool call") as exc:
        server._read_response("shape-tool-only")
    msg = str(exc.value)
    assert "view_file" in msg
    assert target in msg
    assert "failed silently or timed out" not in msg


def test_parse_does_not_return_earlier_content_when_terminal_is_tool_only():
    lines = [
        _entry("PLANNER_RESPONSE", "stale earlier answer"),
        _entry("PLANNER_RESPONSE", tool_calls=_view_file_call("/tmp/x.py")),
    ]
    content, pending = server._parse_planner_transcript("\n".join(lines))
    assert content is None
    assert pending[0]["name"] == "view_file"


def test_format_tool_call_strips_embedded_quotes_and_names_range():
    tc = _view_file_call("/Users/zvisegal/devlope/TaideAi/src/taide/v2/pm.py", 1, 800)[0]
    formatted = server._format_tool_call(tc)
    assert formatted == (
        "view_file /Users/zvisegal/devlope/TaideAi/src/taide/v2/pm.py lines 1-800"
    )


def test_soft_deny_excerpt_prefers_jetski_stderr():
    stderr = (
        'jetski: no output produced — a tool required the "read_file" permission '
        "that headless mode cannot prompt for, so it was auto-denied. "
        "Add an allow-rule under permissions.allow in settings.json "
        "(e.g. read_file(<target>)). Alternatively, re-run with "
        "--dangerously-skip-permissions to auto-approve all tools."
    )
    excerpt = server._soft_deny_excerpt(stderr)
    assert excerpt is not None
    assert "headless mode cannot prompt" in excerpt
    assert server._is_print_mode_soft_deny(stderr)


# --------------------------------------------------------------------------
# _run_agy tool-loop
# --------------------------------------------------------------------------


def _ok_proc(args, stderr=""):
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=stderr)


def test_run_agy_content_only_does_not_continue(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    ws = str(tmp_path)
    last_conv_file.write_text(json.dumps({ws: "c-plain"}), encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        _write_transcript(brain_dir, "c-plain", _fixture_content_only())
        return _ok_proc(args)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: "")

    assert server._run_agy("hi", ws, continue_conv=False, timeout_s=30) == "plain answer"
    assert len(calls) == 1
    assert "--conversation" not in calls[0]
    # P0: workspace must be passed via --add-dir (agy ignores cwd alone).
    assert "--add-dir" in calls[0]
    add_idx = calls[0].index("--add-dir")
    assert calls[0][add_idx + 1] == os.path.abspath(ws)
    assert "--dangerously-skip-permissions" not in calls[0]
    assert "--sandbox" not in calls[0]


def test_run_agy_stale_cli_log_does_not_block_continue(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    """A previous session's soft-deny in cli.log must not abort this run."""
    ws = str(tmp_path)
    last_conv_file.write_text(json.dumps({ws: "c-fresh"}), encoding="utf-8")
    calls = []
    stale_log = (
        "Print mode: conversation=00000000-old-session, sending message\n"
        'Print mode: soft-denying tool confirmation "ReadFile" at step 3\n'
        "permission check failed for read_file \"/tmp/old.py\""
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            _write_transcript(brain_dir, "c-fresh", _fixture_tool_only_terminal())
        else:
            _write_transcript(brain_dir, "c-fresh", _fixture_tool_then_content())
        return _ok_proc(args)  # this process stderr is clean

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: stale_log)

    assert (
        server._run_agy("read the file", ws, continue_conv=False, timeout_s=30)
        == "final after tool"
    )
    assert len(calls) == 2


def test_cli_log_for_conv_requires_this_conversation_id(monkeypatch):
    monkeypatch.setattr(
        server,
        "_read_cli_log_text",
        lambda: 'soft-denying tool confirmation "ReadFile" conv=other-id',
    )
    assert server._cli_log_for_conv("this-id") == ""
    monkeypatch.setattr(
        server,
        "_read_cli_log_text",
        lambda: "permission check failed for read_file x conv=this-id",
    )
    assert "this-id" in server._cli_log_for_conv("this-id")


def test_run_agy_continues_tool_only_until_content(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    ws = str(tmp_path)
    last_conv_file.write_text(json.dumps({ws: "c-loop"}), encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            _write_transcript(brain_dir, "c-loop", _fixture_tool_only_terminal())
        else:
            _write_transcript(brain_dir, "c-loop", _fixture_tool_then_content())
        return _ok_proc(args)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: "")

    assert (
        server._run_agy("read the file", ws, continue_conv=False, timeout_s=30)
        == "final after tool"
    )
    assert len(calls) == 2
    assert "--conversation" in calls[1]
    assert "c-loop" in calls[1]
    assert server._TOOL_CONTINUE_PROMPT in calls[1]


def test_run_agy_soft_deny_does_not_continue(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    ws = str(tmp_path)
    last_conv_file.write_text(json.dumps({ws: "c-deny"}), encoding="utf-8")
    calls = []
    stderr = (
        'jetski: no output produced — a tool required the "read_file" permission '
        "that headless mode cannot prompt for, so it was auto-denied."
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        _write_transcript(brain_dir, "c-deny", _fixture_tool_only_terminal())
        return _ok_proc(args, stderr=stderr)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: "")

    with pytest.raises(RuntimeError, match="unexecuted tool call") as exc:
        server._run_agy("read pm.py", ws, continue_conv=False, timeout_s=30)
    assert len(calls) == 1
    msg = str(exc.value)
    assert "view_file" in msg
    assert "pm.py" in msg
    assert "soft-deny" in msg.lower() or "headless mode cannot prompt" in msg
    assert "failed silently or timed out" not in msg
    assert "--dangerously-skip-permissions" not in calls[0]


def test_run_agy_cli_log_soft_deny_empty_stderr_only_one_invocation(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    """Soft-deny only in this-conv cli.log must skip resumes (not burn 3*timeout)."""
    ws = str(tmp_path)
    conv_id = "c-log-deny"
    last_conv_file.write_text(json.dumps({ws: conv_id}), encoding="utf-8")
    calls = []
    cli_log = (
        f"Print mode: conversation={conv_id}, sending message\n"
        'Print mode: soft-denying tool confirmation "ReadFile" at step 2\n'
        'permission check failed for read_file "/tmp/x.py"'
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        _write_transcript(brain_dir, conv_id, _fixture_tool_only_terminal())
        return _ok_proc(args, stderr="")  # deny is log-only

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: cli_log)

    with pytest.raises(RuntimeError, match="unexecuted tool call") as exc:
        server._run_agy("read the file", ws, continue_conv=False, timeout_s=30)
    assert len(calls) == 1
    msg = str(exc.value)
    assert "view_file" in msg
    assert "soft-deny" in msg.lower() or "permission check failed" in msg.lower()
    assert "failed silently or timed out" not in msg
    assert "did not advance" not in msg


def test_run_agy_exhausts_continuations_then_names_tool(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    """Each resume advances the transcript but stays tool-only until the cap."""
    ws = str(tmp_path)
    last_conv_file.write_text(json.dumps({ws: "c-stuck"}), encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        n = len(calls)
        # Grow transcript so the progress fingerprint changes, but keep a
        # tool-only terminal planner (still pending after each resume).
        lines = [
            _entry("USER_INPUT", "read the file", source="USER_EXPLICIT"),
            _entry("CONVERSATION_HISTORY", source="SYSTEM"),
            *[_entry("CHECKPOINT", f"resume-step-{i}") for i in range(n)],
            _entry(
                "PLANNER_RESPONSE",
                tool_calls=_view_file_call(
                    "/Users/zvisegal/devlope/TaideAi/src/taide/v2/pm.py",
                    start=1,
                    end=800 + n,
                ),
            ),
        ]
        _write_transcript(brain_dir, "c-stuck", lines)
        return _ok_proc(args)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: "")

    with pytest.raises(RuntimeError, match="unexecuted tool call") as exc:
        server._run_agy("read the file", ws, continue_conv=False, timeout_s=30)
    assert len(calls) == 1 + server._MAX_TOOL_CONTINUATIONS
    assert "view_file" in str(exc.value)
    assert "failed silently or timed out" not in str(exc.value)
    assert "did not advance" not in str(exc.value)


def test_run_agy_stalled_transcript_fails_fast(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    """Identical transcript after resume must not claim 'unexecuted tool call'."""
    ws = str(tmp_path)
    last_conv_file.write_text(json.dumps({ws: "c-stall"}), encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        # Same bytes every time — fingerprint does not advance.
        _write_transcript(brain_dir, "c-stall", _fixture_tool_only_terminal())
        return _ok_proc(args)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    monkeypatch.setattr(server, "_read_cli_log_text", lambda: "")

    with pytest.raises(RuntimeError, match="did not advance") as exc:
        server._run_agy("read the file", ws, continue_conv=False, timeout_s=30)
    assert len(calls) == 2  # initial + one resume, then fail fast
    msg = str(exc.value)
    assert "unexecuted tool call" not in msg
    assert "c-stall" in msg


def test_invoke_agy_raises_on_conversation_fork(
    brain_dir, last_conv_file, tmp_path, monkeypatch
):
    """Pinned resume that records a new last_conversations id must not stick on pin."""
    ws = str(tmp_path)
    pin = "c-old-pin"
    forked = "c-new-fork"
    last_conv_file.write_text(json.dumps({ws: pin}), encoding="utf-8")
    _write_transcript(brain_dir, pin, _fixture_tool_only_terminal())
    _write_transcript(brain_dir, forked, _fixture_content_only())

    def fake_run(args, **kwargs):
        # Simulate agy writing a NEW conversation id for this workspace.
        last_conv_file.write_text(json.dumps({ws: forked}), encoding="utf-8")
        return _ok_proc(args)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="forked the conversation") as exc:
        server._invoke_agy(
            "continue",
            ws,
            30,
            continue_conv=True,
            pinned_conv=pin,
        )
    msg = str(exc.value)
    assert pin in msg
    assert forked in msg
    assert "brain/" in msg or "Refusing" in msg


# --------------------------------------------------------------------------
# _parse_agy_version
# --------------------------------------------------------------------------


def test_parse_agy_version_bare():
    assert server._parse_agy_version("1.0.4") == (1, 0, 4)


def test_parse_agy_version_trailing_newline():
    assert server._parse_agy_version("1.0.4\n") == (1, 0, 4)


def test_parse_agy_version_with_prefix_and_build():
    assert server._parse_agy_version("agy version 1.2.0 (build abc)") == (1, 2, 0)


def test_parse_agy_version_garbage_returns_none():
    assert server._parse_agy_version("no version here") is None


def test_parse_agy_version_empty_returns_none():
    assert server._parse_agy_version("") is None


# --------------------------------------------------------------------------
# _compat_warning
# --------------------------------------------------------------------------


def test_compat_warning_none_for_verified_version():
    assert server._compat_warning(server.VERIFIED_AGY_VERSION) is None


def test_compat_warning_none_for_older_version():
    assert server._compat_warning((1, 0, 3)) is None


def test_compat_warning_warns_for_newer_version():
    msg = server._compat_warning((1, 1, 0))
    assert msg is not None
    assert "1.1.0" in msg  # the detected version
    assert "1.0.4" in msg  # the verified baseline it's compared to


def test_compat_warning_none_when_version_unknown():
    assert server._compat_warning(None) is None


# --------------------------------------------------------------------------
# _debug_enabled
# --------------------------------------------------------------------------


def test_debug_enabled_false_when_unset(monkeypatch):
    monkeypatch.delenv("AGY_BRIDGE_DEBUG", raising=False)
    assert server._debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_debug_enabled_true_for_truthy(monkeypatch, value):
    monkeypatch.setenv("AGY_BRIDGE_DEBUG", value)
    assert server._debug_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_debug_enabled_false_for_falsy(monkeypatch, value):
    monkeypatch.setenv("AGY_BRIDGE_DEBUG", value)
    assert server._debug_enabled() is False


# --------------------------------------------------------------------------
# _startup_checks  (composition of the tested helpers; agy version injected)
# --------------------------------------------------------------------------


def test_startup_checks_warns_on_newer_agy(monkeypatch, caplog):
    monkeypatch.setattr(server, "_get_agy_version", lambda: "2.0.0")
    caplog.set_level("WARNING", logger="agy_bridge")
    server._startup_checks()
    assert "newer" in caplog.text


def test_startup_checks_silent_on_verified_agy(monkeypatch, caplog):
    monkeypatch.setattr(server, "_get_agy_version", lambda: "1.0.4")
    caplog.set_level("WARNING", logger="agy_bridge")
    server._startup_checks()
    assert caplog.text == ""


def test_startup_checks_silent_when_agy_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(server, "_get_agy_version", lambda: None)
    caplog.set_level("WARNING", logger="agy_bridge")
    server._startup_checks()
    assert caplog.text == ""
