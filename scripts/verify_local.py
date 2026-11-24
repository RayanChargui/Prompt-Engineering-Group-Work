#!/usr/bin/env python3
"""Run acceptance checks against real local Ollama, files, SQLite, and Docker."""
import argparse
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medtech.agent import Agent, Config, ParseError, chat, parse, unsupported_claim
from medtech.backends import LocalBackend, SandboxBackend
from medtech.store import Store, StoreError


class CaptureUI:
    """Approve tools in isolated test workspaces and retain the final answer."""

    def __init__(self) -> None:
        self.answers = []

    def say(self, text: str) -> None:
        self.answers.append(text)

    def warn(self, text: str) -> None:
        print(f"  warning: {text}", flush=True)

    def action(self, *args: object) -> None:
        pass

    def confirm(self, *args: object) -> bool:
        return True

    def result(self, ok: bool, text: str) -> None:
        if not ok:
            print(f"  tool failure: {text}", flush=True)


def require(condition: bool, message: str) -> None:
    """Keep checks active even when Python is run with -O."""
    if not condition:
        raise RuntimeError(message)


def worker(action: str, root: Path) -> None:
    """Use separate processes to demonstrate SQLite persistence across restarts."""
    if action in ("resume", "new"):
        require(root is not None and (root / "session-id").is_file(),
                "Prerequisite failed: the initial name-memory check did not save a session.")
    store = Store()
    backend = LocalBackend(str(root / "local"))
    ui = CaptureUI()
    try:
        if action == "remember":
            sid = store.new_session("local", backend.host_root)
            agent = Agent(Config(), backend, store, sid, ui, auto=True)
            agent.turn("My name is Rayan. Please remember my name.")
            require(any(row["key"] == "name" and row["value"] == "Rayan" for row in store.memories()),
                    "Real model did not extract name=Rayan")
            (root / "session-id").write_text(str(sid), encoding="utf-8")
        elif action == "resume":
            sid = int((root / "session-id").read_text(encoding="utf-8"))
            original = list(store.messages(sid))
            require(any("My name is Rayan" in row["content"] for row in original), "User message missing after restart")
            agent = Agent(Config(), backend, store, sid, ui, auto=True)
            require(len(agent.hist) == len(original), "Resume did not restore the complete persisted history")
            agent.turn("What is my name? Answer briefly.")
            require(ui.answers and "Rayan" in ui.answers[-1], "Resumed real model did not recall Rayan")
        elif action == "new":
            sid = store.new_session("local", backend.host_root)
            agent = Agent(Config(), backend, store, sid, ui, auto=True)
            agent.turn("What is my name? Answer briefly.")
            require(ui.answers and "Rayan" in ui.answers[-1], "Real model did not use memory in a new session")
        else:
            raise RuntimeError("Unknown worker action")
    finally:
        store.close()


def run_worker(action: str, root: Path) -> None:
    """Execute real model turns in a fresh Python interpreter."""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", action,
                             "--root", str(root)], capture_output=True, text=True, timeout=1800)
    require(result.returncode == 0, (result.stdout + result.stderr).strip())


def real_creation(backend: object, db: Path) -> None:
    """Ask the real model to act, then inspect both the execution environment and host."""
    store = Store(str(db))
    ui = CaptureUI()
    try:
        sid = store.new_session(backend.name, backend.host_root)
        agent = Agent(Config(), backend, store, sid, ui, auto=True)
        agent.turn("Create folder test_dir with hello.txt inside it, containing exactly the two characters hi. "
                   "Use the filesystem tools, then report the verified absolute file path.")
        result = backend.run("read_file", {"path": backend.root + "/test_dir/hello.txt"})
        require(result["success"] and result["stdout"] == "hi", f"Environment read-back failed: {result}")
        host_file = Path(backend.host_root) / "test_dir" / "hello.txt"
        require(host_file.read_text(encoding="utf-8") == "hi", f"Host file missing or incorrect: {host_file}")
        verified_path = backend.root + "/test_dir/hello.txt"
        require(bool(ui.answers) and verified_path in ui.answers[-1],
                "The agent did not give an accepted final completion containing the verified absolute file path")
        require(any(task["status"] == "done" and task["tool"] == "write_file" for task in store.tasks(sid)),
                "No successful write_file audit record")
    finally:
        store.close()


def ui_check(root: Path) -> None:
    """Launch the real terminal UI headlessly and open a blank draft without Ollama."""
    from medtech.tui import MedTechApp

    async def exercise() -> None:
        store = Store(str(root / "ui.db"))
        try:
            backend = LocalBackend(str(root / "ui-workspace"))
            sid = store.new_session("local", backend.host_root)
            store.add_message(sid, "user", "Saved UI smoke-test message")
            original = list(store.messages(sid))
            app = MedTechApp(Config(), backend, store, sid=sid)
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                require(app.is_running, "Textual application did not start")
                require(len(app.query("#sidebar")) == 1, "Conversation sidebar is missing")
                require(len(app.query("#message-input")) == 1, "Multiline editor is missing")
                await pilot.press("ctrl+n")
                await pilot.pause()
                require(app.sid is None, "New-chat keyboard action did not open a blank draft")
                require(not app.query_one("#message-input").text, "New-chat editor is not empty")
                require(len(store.sessions()) == 1, "An empty draft was persisted as a sidebar conversation")
                require([tuple(row) for row in store.messages(sid)] == [tuple(row) for row in original],
                        "Opening a new draft changed the previous saved conversation")
        finally:
            store.close()

    asyncio.run(exercise())


def cli_checks(root: Path) -> None:
    """Exercise public chat and memory commands against a separate real database."""
    from medtech.cli import command
    store = Store(str(root / "commands.db"))
    try:
        backend = LocalBackend(str(root / "commands"))
        sid = store.new_session("local", backend.host_root)
        agent = Agent(Config(), backend, store, sid, CaptureUI())
        with contextlib.redirect_stdout(io.StringIO()):
            command(f'/rename {sid} "Acceptance chat"', agent, store)
            require(store.session(sid)["title"] == "Acceptance chat", "/rename failed")
            command("/history", agent, store)
            command("/chats", agent, store)
            command("/new", agent, store)
            require(agent.sid != sid, "/new failed")
            command(f"/resume {sid}", agent, store)
            require(agent.sid == sid, "/resume failed")
            command("/remember name=Rayan", agent, store)
            command("/memory", agent, store)
            fact = store.memories()[0]
            command(f"/forget {fact['id']}", agent, store)
            require(not store.memories(), "/forget failed")
            command("/remember name=Rayan", agent, store)
            command("/memory clear", agent, store)
            require(not store.memories(), "/memory clear failed")
            command(f"/delete {sid}", agent, store)
            require(store.session(sid) is None, "/delete failed")
    finally:
        store.close()


def safety_checks(root: Path) -> None:
    """Check malformed responses, unsupported claims, and actual operation errors."""
    try:
        parse('{"say":"done","action":{"tool":"mkdir","args":')
    except ParseError:
        pass
    else:
        raise RuntimeError("Partial tool call was accepted")
    require(unsupported_claim("I created the folder", []), "Unsupported creation claim was accepted")
    backend = LocalBackend(str(root / "failure"))
    result = backend.run("write_file", {"path": "/etc/medtech-forbidden-check", "content": "hi"})
    require(not result["success"] and result["exit_code"] != 0 and bool(result["stderr"]),
            "A rejected write was reported as success")
    corrupt = root / "corrupted.db"
    corrupt.write_bytes(b"this is not a SQLite database")
    try:
        Store(str(corrupt))
    except StoreError as exc:
        require("database" in str(exc).lower(), "Corruption error was not actionable")
    else:
        raise RuntimeError("Corrupt database was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["remember", "resume", "new"], help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            require(args.root is not None, "Worker requires --root.")
            worker(args.worker, args.root)
        except Exception as exc:
            print(f"{args.worker} check failed: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    failures = []

    def check(name: str, function: object) -> None:
        try:
            function()
        except Exception as exc:
            failures.append(name)
            print(f"FAIL {name}: {exc}", flush=True)
        else:
            print(f"PASS {name}", flush=True)

    with tempfile.TemporaryDirectory(prefix="medtech-acceptance-") as directory:
        root = Path(directory)
        previous = os.environ.get("MEDTECH_HOME")
        os.environ["MEDTECH_HOME"] = str(root / "data")
        sandbox = SandboxBackend(str(root / "sandbox"), name="medtech-verify-" + uuid.uuid4().hex[:12])
        try:
            cfg = Config()
            print(f"Using real Ollama at {cfg.url}, model {cfg.model}; temporary data at {root}", flush=True)
            check("headless Textual UI launch and new-chat action", lambda: ui_check(root))
            check("Ollama responds with valid agent JSON", lambda: parse(chat(cfg, [
                {"role": "user", "content": 'Reply exactly {"say":"ready","action":null}'}])))
            check("session saved and durable user fact extracted", lambda: run_worker("remember", root))
            check("resume restores history after process restart", lambda: run_worker("resume", root))
            check("new session recalls name after process restart", lambda: run_worker("new", root))
            check("chat and memory CLI commands", lambda: cli_checks(root))
            check("malformed calls, unsupported claims, failures, and corrupted DB", lambda: safety_checks(root))
            check("real model creates local folder and file", lambda: real_creation(
                LocalBackend(str(root / "local-creation")), root / "local-creation.db"))

            def docker_check() -> None:
                sandbox.ensure()
                real_creation(sandbox, root / "sandbox.db")
                # A second backend must see the same persistent mount and files.
                reopened = SandboxBackend(sandbox.host_root, name=sandbox.box)
                reopened.ensure()
                result = reopened.run("read_file", {"path": "/workspace/test_dir/hello.txt"})
                require(result["success"] and result["stdout"] == "hi", "Sandbox data did not persist")

            check("real model creates Docker folder/file visible on host and after reopen", docker_check)
        finally:
            # The randomly named container belongs to this script, never an app/user container.
            try:
                result = subprocess.run(["docker", "inspect", sandbox.box], capture_output=True, timeout=20)
                if result.returncode == 0:
                    sandbox.reset()
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"Cleanup note: {exc}", flush=True)
            if previous is None:
                os.environ.pop("MEDTECH_HOME", None)
            else:
                os.environ["MEDTECH_HOME"] = previous
    print(f"{9 - len(failures)} passed; {len(failures)} failed. No required checks are skipped.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
