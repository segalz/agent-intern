"""Antigravity CLI (agy) bridge — fastmcp server.

Exposes Antigravity CLI as MCP tools so Claude Code (or any MCP host) can
use it as a sub-agent. Solves the headless print-mode bug in agy 1.0.x
(verified broken through 1.0.1; stdout not re-tested on 1.0.4) by running
`agy -p` and reading the response from agy's own transcript files instead of
relying on stdout. State-file layout and transcript schema re-verified on
agy 1.0.4.

Auth: piggybacks on whatever credential store `agy` itself uses on the host
OS (Windows Credential Manager, macOS Keychain, libsecret on Linux). User
must have logged in interactively at least once via the Antigravity IDE or
`agy -i`. Uses the same AI Pro quota. The bridge itself only does cross-
platform filesystem reads under `~/.gemini/antigravity-cli/`.

Model: as of agy 1.1.14, print mode honors `--model` and `--effort`
(low|medium|high), and the default comes from
`~/.gemini/antigravity-cli/settings.json` (this host currently selects
"Gemini 3.7 Flash (Low)"). Older releases baked a fixed print-mode model;
this bridge does not currently expose model/effort MCP parameters — it
uses whatever agy would use for a plain `agy -p` on the host.

Compat (agy 1.0.4): state-file paths, last_conversations.json, and the
transcript schema are unchanged. agy 1.0.4 added a SQLite (.db) conversation
format that it says "will be the CLI's conversation format" — once that
becomes the default, the JSONL transcript this bridge parses may disappear
and _read_response will need a SQLite reader (it now raises a clear,
SQLite-aware error when the transcript is missing).

Print-mode tool loop (agy 1.1.3+ / reproduced on 1.1.14): `-p` no longer
hangs on a confirmation prompt. A tool that needs approval is soft-denied
and the process still exits 0, leaving a terminal MODEL PLANNER_RESPONSE
with `tool_calls` and no `content`. That is not a timeout and not a
view_file size or line-range limit: on this host, view_file of a 5-line
in-workspace file (117 B) and of pm.py (~3008 lines / 111 KB) both
soft-denied in ~5.5s with the same stderr notice, while grep_search in
the same mode ran because it did not require confirmation. The allow-rule
print mode names is `permissions.allow` in agy's settings.json
(e.g. `read_file(/abs/path)`). This bridge will not add
--dangerously-skip-permissions or --sandbox (see SECURITY). When a run
ends on an unexecuted tool call, we surface the tool/args and that
soft-deny reason instead of claiming the model failed silently.

SECURITY — read this: `agy -p` runs the model as an autonomous agent that
auto-executes its tools (read/write files, run shell commands, reach the
network) with NO approval gate and NO opt-out. We verified empirically on
agy 1.0.4 / Windows that print mode runs out-of-workspace writes and network
fetches even WITHOUT --dangerously-skip-permissions (that flag is a no-op
for -p), and that --sandbox does not constrain filesystem or network egress
there. The bridge passes `--add-dir <abs workspace>` so agy treats that path as
the agent workspace (cwd alone is ignored by agy), but `workspace` is
still NOT a security boundary: every call effectively runs arbitrary code
with your privileges. Only invoke this bridge with trusted prompts on
trusted content (untrusted input here is the classic prompt-injection
"lethal trifecta"). For real isolation, run the whole bridge inside a
container or VM.
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("agy")

# Logs go to stderr (stdout is the MCP protocol channel). Quiet by default;
# set AGY_BRIDGE_DEBUG=1 for per-call diagnostics. See _configure_logging.
log = logging.getLogger("agy_bridge")

AGY_DATA = Path.home() / ".gemini" / "antigravity-cli"
LAST_CONVERSATIONS = AGY_DATA / "cache" / "last_conversations.json"
BRAIN_DIR = AGY_DATA / "brain"

# Serializes agy invocations within this process. Concurrent runs would race
# on last_conversations.json (agy rewrites it on every call), so a second
# request could pick up the first request's conversation id.
_AGY_LOCK = threading.Lock()

# Brief pause after agy exits so transcript.jsonl is flushed to disk.
_FS_SETTLE_S = 0.3

# Bounded resumes when print mode ends on a tool-call-only PLANNER_RESPONSE.
# Each resume is another `agy -p` under the same _AGY_LOCK. Soft-denied
# permission prompts skip this loop — retries cannot approve the tool.
_MAX_TOOL_CONTINUATIONS = 3
_TOOL_CONTINUE_PROMPT = (
    "Continue. Execute any pending tool calls and then give your final answer."
)

_PATH_ARG_KEYS = (
    "AbsolutePath",
    "absolute_path",
    "Path",
    "path",
    "SearchPath",
    "file_path",
    "FilePath",
    "target_file",
)

# Latest agy version the bridge's state-file assumptions were verified against.
# Newer agy releases may change paths/schemas (the SQLite migration is the known
# risk), so we warn at startup if the installed agy is newer than this.
VERIFIED_AGY_VERSION = (1, 0, 4)


def _parse_agy_version(text: str) -> Optional[tuple[int, int, int]]:
    """Extract a (major, minor, patch) tuple from `agy --version` output."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _compat_warning(version: Optional[tuple[int, int, int]]) -> Optional[str]:
    """Return a warning if the installed agy is newer than we've verified.

    None if the version is unknown, equal to, or older than VERIFIED_AGY_VERSION.
    """
    if version is None or version <= VERIFIED_AGY_VERSION:
        return None
    detected = ".".join(map(str, version))
    verified = ".".join(map(str, VERIFIED_AGY_VERSION))
    return (
        f"agy {detected} is newer than the {verified} this bridge was verified "
        "against. If responses look wrong or empty, agy may have changed its "
        "state-file layout (the SQLite conversation format is the known risk). "
        "Pin a known-good agy version if needed."
    )


def _debug_enabled() -> bool:
    """True if AGY_BRIDGE_DEBUG is set to a truthy value (1/true/yes/on)."""
    return os.environ.get("AGY_BRIDGE_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_agy_version() -> Optional[str]:
    """Return `agy --version` output, or None if agy can't be run."""
    try:
        proc = subprocess.run(
            ["agy", "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def _startup_checks() -> None:
    """Warn (once, at startup) if the installed agy is newer than verified."""
    warning = _compat_warning(_parse_agy_version(_get_agy_version() or ""))
    if warning:
        log.warning(warning)


def _configure_logging() -> None:
    """Route bridge logs to stderr; DEBUG when AGY_BRIDGE_DEBUG is set."""
    handler = logging.StreamHandler()  # defaults to stderr
    handler.setFormatter(logging.Formatter("[agy-bridge] %(levelname)s: %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.DEBUG if _debug_enabled() else logging.WARNING)
    log.propagate = False


def _normalize_workspace(ws: Optional[str]) -> str:
    return os.path.abspath(ws) if ws else os.getcwd()


def _read_last_conv_id(workspace: str) -> Optional[str]:
    if not LAST_CONVERSATIONS.exists():
        return None
    try:
        data = json.loads(LAST_CONVERSATIONS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if workspace in data:
        return data[workspace]
    for k, v in data.items():
        if k.lower() == workspace.lower():
            return v
    return None


def _find_newest_conv_after(start_time: float) -> Optional[str]:
    if not BRAIN_DIR.exists():
        return None
    best = None
    best_mtime = start_time - 2
    for child in BRAIN_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best = child.name
            best_mtime = mtime
    return best


def _transcript_path(conv_id: str) -> Path:
    return BRAIN_DIR / conv_id / ".system_generated" / "logs" / "transcript.jsonl"


def _missing_transcript_error(transcript: Path, conv_id: str) -> RuntimeError:
    # agy 1.0.4 added a SQLite (.db) conversation format and announced it
    # "will be the CLI's conversation format". This is where the bridge
    # breaks first if a future release stops writing JSONL transcripts.
    conv_dir = BRAIN_DIR / conv_id
    db_files = sorted(str(p) for p in conv_dir.glob("**/*.db")) if conv_dir.exists() else []
    if db_files:
        hint = (
            f" Found SQLite store(s) instead: {db_files}. agy appears to have "
            "migrated this conversation to its 1.0.4 SQLite format; the bridge's "
            "JSONL transcript reader needs updating to read from the .db file."
        )
    else:
        hint = (
            " No JSONL transcript in the conversation dir. If you upgraded agy, it "
            "may have switched to the SQLite (.db) conversation format added in 1.0.4."
        )
    return RuntimeError(f"Transcript not found: {transcript}.{hint}")


def _is_completed_planner(entry: dict) -> bool:
    return (
        entry.get("source") == "MODEL"
        and entry.get("status") == "DONE"
        and entry.get("type") == "PLANNER_RESPONSE"
    )


def _last_completed_planner(text: str) -> Optional[dict]:
    """Return the last completed MODEL PLANNER_RESPONSE entry, or None."""
    last: Optional[dict] = None
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and _is_completed_planner(entry):
            last = entry
    return last


def _parse_planner_transcript(text: str) -> tuple[Optional[str], list]:
    """Parse JSONL transcript text into (final_content, terminal_tool_calls).

    The last completed MODEL PLANNER_RESPONSE is the turn that counts:
    - if it has content, that content is the final answer (earlier tool steps
      and earlier planner contents are ignored);
    - if it has tool_calls and no content, those calls are unexecuted/pending
      and earlier content must not be returned as the answer;
    - otherwise there is no completed model response.
    """
    last = _last_completed_planner(text)
    if last is None:
        return None, []
    content = last.get("content")
    if content:
        return content, []
    tool_calls = last.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    return None, tool_calls


def _transcript_progress_fingerprint(text: str) -> tuple:
    """Fingerprint terminal planner + size so a stalled resume is detectable.

    `_FS_SETTLE_S` is only a best-effort sleep; identical fingerprints after a
    resume mean the transcript did not advance (flush lag or no-op turn).
    """
    last = _last_completed_planner(text)
    terminal = json.dumps(last, sort_keys=True, ensure_ascii=False) if last else ""
    return (len(text), terminal)


def _unwrap_quoted(value: object) -> str:
    s = str(value).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _format_tool_call(tc: object) -> str:
    if not isinstance(tc, dict):
        return str(tc)
    name = tc.get("name") or "unknown"
    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
    path = None
    for key in _PATH_ARG_KEYS:
        if args.get(key) not in (None, ""):
            path = _unwrap_quoted(args[key])
            break
    if path:
        start, end = args.get("StartLine"), args.get("EndLine")
        if start is not None and end is not None:
            return f"{name} {path} lines {start}-{end}"
        return f"{name} {path}"
    extras = {
        k: _unwrap_quoted(v)
        for k, v in args.items()
        if k not in {"toolAction", "toolSummary"}
    }
    if extras:
        return f"{name} {extras}"
    return str(name)


def _format_tool_calls(tool_calls: list) -> str:
    if not tool_calls:
        return "unknown"
    return "; ".join(_format_tool_call(tc) for tc in tool_calls)


def _is_print_mode_soft_deny(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            "soft-deny",
            "soft-denying",
            "headless mode cannot prompt",
            "auto-denied",
            "user denied permission",
            "permission check failed",
        )
    )


def _soft_deny_excerpt(stderr: str, log_text: str = "") -> Optional[str]:
    combined = "\n".join(part for part in (stderr, log_text) if part)
    if not combined or not _is_print_mode_soft_deny(combined):
        return None
    parts: list[str] = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("jetski:") or "headless mode cannot prompt" in stripped.lower():
            parts.append(stripped)
            break
    for line in log_text.splitlines():
        lower = line.lower()
        if "permission check failed for" in lower:
            idx = lower.find("permission check failed")
            parts.append(line[idx:].strip())
            break
    if parts:
        return " ".join(parts)
    return combined.strip()[:500]


def _read_cli_log_text() -> str:
    path = AGY_DATA / "cli.log"
    try:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _cli_log_for_conv(conv_id: str) -> str:
    """Return cli.log text only if it mentions this conversation.

    The symlink may still point at a previous session; using it unfiltered
    would attribute another run's soft-deny to this one.
    """
    text = _read_cli_log_text()
    if conv_id and conv_id in text:
        return text
    return ""


def _unexecuted_tool_error(
    tool_calls: list,
    transcript: Path,
    *,
    stderr: str = "",
    log_text: str = "",
) -> RuntimeError:
    named = _format_tool_calls(tool_calls)
    msg = (
        f"agy ended on an unexecuted tool call: {named}. "
        "The print-mode run returned before the tool executed."
    )
    deny = _soft_deny_excerpt(stderr, log_text)
    if deny:
        msg += (
            f" Print-mode permission soft-deny (agy 1.1.3+): {deny} "
            "This is not a timeout and not a view_file size/line-range limit. "
            "Grant a matching permissions.allow rule in agy's settings.json. "
            "The bridge does not pass --dangerously-skip-permissions (see SECURITY)."
        )
    else:
        msg += f" Transcript: {transcript}."
    return RuntimeError(msg)


def _load_transcript_text(conv_id: str) -> tuple[str, Path]:
    transcript = _transcript_path(conv_id)
    if not transcript.exists():
        raise _missing_transcript_error(transcript, conv_id)
    return transcript.read_text(encoding="utf-8"), transcript


def _read_response(conv_id: str) -> str:
    text, transcript = _load_transcript_text(conv_id)
    content, pending = _parse_planner_transcript(text)
    if content:
        # Last completed planner response is the final answer (tool steps come earlier).
        return content
    if pending:
        raise _unexecuted_tool_error(pending, transcript)
    raise RuntimeError(
        f"No completed MODEL response in transcript {transcript}. "
        "agy may have failed silently or timed out."
    )


def _soft_deny_blocks_continuation(stderr: str, conv_id: str) -> bool:
    """True when this run already soft-denied a tool and resumes cannot help.

    Checks this process's stderr first. Also consults cli.log, but only when
    `_cli_log_for_conv` confirms the log mentions THIS conv_id — a stale
    symlink/log from another session must never abort the loop.
    """
    if _is_print_mode_soft_deny(stderr):
        return True
    log_text = _cli_log_for_conv(conv_id)
    return bool(log_text) and _is_print_mode_soft_deny(log_text)


def _invoke_agy(
    prompt: str,
    workspace: str,
    timeout_s: int,
    *,
    continue_conv: bool,
    pinned_conv: Optional[str] = None,
) -> tuple[subprocess.CompletedProcess, str]:
    """Run one `agy -p` and resolve the conversation id. Caller holds _AGY_LOCK."""
    # Note: agy's `-p` mode auto-executes all tools/commands with no approval
    # gate, so we deliberately do NOT pass --dangerously-skip-permissions (it is
    # a no-op for -p) or --sandbox (verified not to constrain FS/network on
    # Windows). There is no agy flag that makes print mode safe; see the module
    # docstring's SECURITY note.
    # Empirically (agy 1.1.14): subprocess cwd=workspace alone is ignored —
    # tool Cwd stays under ~/.gemini/antigravity-cli. `--add-dir <abs>` makes
    # agy treat that path as the workspace (verified via run_command Cwd).
    workspace = os.path.abspath(workspace)
    args = [
        "agy",
        "--add-dir",
        workspace,
        "--print-timeout",
        f"{timeout_s}s",
    ]

    pin = pinned_conv
    if continue_conv:
        # Pin to the exact conversation rooted at this workspace instead of `-c`
        # ("most recent"), which could resume a conversation started elsewhere in
        # between. Fall back to -c only when we have no id on record yet.
        if pin is None:
            pin = _read_last_conv_id(workspace)
        if pin:
            args.extend(["--conversation", pin])
        else:
            args.append("-c")
    args.extend(["-p", prompt])

    start = time.time()
    log.debug(
        "running agy: continue=%s pinned=%s workspace=%s timeout=%ss prompt_chars=%d",
        continue_conv,
        pin,
        workspace,
        timeout_s,
        len(prompt),
    )
    proc = subprocess.run(
        args,
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s + 30,
    )
    log.debug("agy exited %s in %.1fs", proc.returncode, time.time() - start)
    if proc.returncode != 0:
        raise RuntimeError(
            f"agy exited {proc.returncode}\n"
            f"stderr: {proc.stderr[-1000:]}\n"
            f"stdout: {proc.stdout[-500:]}"
        )

    time.sleep(_FS_SETTLE_S)

    recorded = _read_last_conv_id(workspace)
    if pin and recorded and recorded != pin:
        raise RuntimeError(
            f"agy forked the conversation on resume: pinned={pin} but "
            f"last_conversations.json now records {recorded} for "
            f"workspace={workspace}. Refusing to keep reading brain/{pin}/ "
            "while agy advanced a different conversation id."
        )
    conv_id = pin or recorded or _find_newest_conv_after(start)
    log.debug("resolved conv_id=%s", conv_id)
    if conv_id is None:
        raise RuntimeError(
            f"No conversation found after agy run (workspace={workspace}). "
            f"Check {LAST_CONVERSATIONS} and {BRAIN_DIR}."
        )
    return proc, conv_id


def _run_agy(prompt: str, workspace: str, continue_conv: bool, timeout_s: int) -> str:
    with _AGY_LOCK:
        proc, conv_id = _invoke_agy(
            prompt,
            workspace,
            timeout_s,
            continue_conv=continue_conv,
        )
        text, transcript = _load_transcript_text(conv_id)
        content, pending = _parse_planner_transcript(text)
        if content:
            return content

        # Resume the pinned conversation until a content-bearing planner
        # appears, unless this run already soft-denied a tool (stderr and/or
        # this-conv cli.log). Retries cannot approve a confirmation prompt.
        stderr = proc.stderr or ""
        progress_fp = _transcript_progress_fingerprint(text)
        attempts = 0
        while pending and attempts < _MAX_TOOL_CONTINUATIONS:
            if _soft_deny_blocks_continuation(stderr, conv_id):
                break
            attempts += 1
            log.debug(
                "terminal tool_calls=%s; continuing %s (%d/%d)",
                _format_tool_calls(pending),
                conv_id,
                attempts,
                _MAX_TOOL_CONTINUATIONS,
            )
            proc, conv_id = _invoke_agy(
                _TOOL_CONTINUE_PROMPT,
                workspace,
                timeout_s,
                continue_conv=True,
                pinned_conv=conv_id,
            )
            text, transcript = _load_transcript_text(conv_id)
            stderr = proc.stderr or ""
            new_fp = _transcript_progress_fingerprint(text)
            if new_fp == progress_fp:
                # Resume soft-deny often leaves the transcript unchanged; prefer
                # the soft-deny diagnosis over a stall error.
                if _soft_deny_blocks_continuation(stderr, conv_id):
                    content, pending = _parse_planner_transcript(text)
                    break
                raise RuntimeError(
                    f"agy transcript did not advance after resume "
                    f"(conversation={conv_id}). The resumed turn's "
                    "PLANNER_RESPONSE was not flushed or the run made no "
                    f"progress. Transcript: {transcript}."
                )
            progress_fp = new_fp
            content, pending = _parse_planner_transcript(text)
            if content:
                return content

        if pending:
            raise _unexecuted_tool_error(
                pending,
                transcript,
                stderr=stderr,
                log_text=_cli_log_for_conv(conv_id),
            )
        raise RuntimeError(
            f"No completed MODEL response in transcript {transcript}. "
            "agy may have failed silently or timed out."
        )


@mcp.tool()
def agy_ask(prompt: str, workspace: Optional[str] = None, timeout_s: int = 180) -> str:
    """Ask Antigravity (agy CLI) a question in a NEW conversation.

    Uses your existing AI Pro authentication (silent-auth via the host OS
    credential store). Returns the model's final response as text.

    Model/effort follow agy's host defaults (`--model` / `--effort` and
    settings.json). This tool does not expose those flags; on agy 1.1.14 the
    host default is typically whatever is selected in
    ~/.gemini/antigravity-cli/settings.json (e.g. Gemini 3.7 Flash Low).

    Args:
        prompt: Question or instruction for Antigravity.
        workspace: Project directory passed to agy via --add-dir (absolute).
                   Defaults to cwd. Choose an existing project dir for
                   context-aware responses.
        timeout_s: Max seconds to wait for agy to complete. Default 180.
    """
    ws = _normalize_workspace(workspace)
    return _run_agy(prompt, ws, continue_conv=False, timeout_s=timeout_s)


@mcp.tool()
def agy_continue(prompt: str, workspace: Optional[str] = None, timeout_s: int = 180) -> str:
    """Continue the Antigravity conversation rooted at this workspace.

    Resumes the exact conversation id recorded for `workspace` (via agy's
    --conversation flag), not agy's global "most recent", so it stays correct
    even if agy was used elsewhere in between. Same model/effort defaults as
    agy_ask (host settings / agy CLI defaults; not overridden here).

    Args:
        prompt: Follow-up message.
        workspace: Project directory passed to agy via --add-dir (absolute);
                   must match the prior conversation. Defaults to cwd.
        timeout_s: Max seconds to wait for agy to complete. Default 180.
    """
    ws = _normalize_workspace(workspace)
    return _run_agy(prompt, ws, continue_conv=True, timeout_s=timeout_s)


if __name__ == "__main__":
    _configure_logging()
    _startup_checks()
    mcp.run()
