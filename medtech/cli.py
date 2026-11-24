"""Terminal UI for local tools, persistent conversations, and user memory."""
import argparse
import json
import shlex
import sys
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .agent import Agent, Config, LLMError
from .backends import BackendError, LocalBackend, SandboxBackend
from .logo import draw_logo
from .store import Store, StoreError

c = Console()
ACCENT, SOFT, DIM = "#3fc1e8", "#e8fbff", "#2b6f87"
HELP = f"""[bold {ACCENT}]/new[/] new chat    [bold {ACCENT}]/history[/] or [bold {ACCENT}]/chats[/] list chats
[bold {ACCENT}]/resume <id>[/] resume chat    [bold {ACCENT}]/history <id>[/] show messages
[bold {ACCENT}]/delete <id>[/] delete chat    [bold {ACCENT}]/rename <id> <title>[/] rename chat
[bold {ACCENT}]/memory[/] list facts    [bold {ACCENT}]/remember <fact>[/] save fact
[bold {ACCENT}]/forget <id or text>[/] remove fact    [bold {ACCENT}]/memory clear[/] remove all facts
[bold {ACCENT}]/auto on|off[/] auto-approve safe writes (deletes and overwrites always ask)
[bold {ACCENT}]/permissions full|workspace[/] local access scope
[bold {ACCENT}]/debug on|off[/] log tool diagnostics to file
[bold {ACCENT}]/tasks [id|all][/] executed actions    [bold {ACCENT}]/help[/] help    [bold {ACCENT}]exit[/] leave"""

# ---------- big title ----------
GLYPHS = {
    "M": ["#...#", "##.##", "#.#.#", "#...#", "#...#"],
    "E": ["#####", "#....", "####.", "#....", "#####"],
    "D": ["####.", "#...#", "#...#", "#...#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#.."],
    "C": [".####", "#....", "#....", "#....", ".####"],
    "H": ["#...#", "#...#", "#####", "#...#", "#...#"],
    "A": [".###.", "#...#", "#####", "#...#", "#...#"],
    "G": [".####", "#....", "#..##", "#...#", ".###."],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
}
STOPS = [(0.0, (0, 127, 164)), (0.5, (63, 193, 232)), (1.0, (232, 251, 255))]


def _grad(i, n):
    t = i / max(n - 1, 1)
    for (t0, a), (t1, b) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return "#%02x%02x%02x" % tuple(int(a[j] + (b[j] - a[j]) * k) for j in range(3))
    return "#e8fbff"


def big(words, gap=3):
    """words: list of strings -> one big block-letter line (list of 5 strings)."""
    rows = []
    for r in range(5):
        parts = []
        for w in words:
            parts.append(" ".join("".join("█" if ch == "#" else " " for ch in GLYPHS[l][r]) for l in w))
        rows.append((" " * gap).join(parts))
    return rows


def title_block(width):
    one = big(["MEDTECH", "AGENT"])
    if len(one[0]) <= width:
        groups = [one]
    else:
        groups = [big(["MEDTECH"]), big(["AGENT"])]
    if max(len(g[0]) for g in groups) > width:
        return Text("MEDTECH AGENT", style=f"bold {ACCENT}", justify="center")
    out = []
    for gi, rows in enumerate(groups):
        for line in rows:
            t = Text()
            for i, ch in enumerate(line):
                t.append(ch, style=_grad(i, len(line)))
            out.append(Align.center(t))
        if gi < len(groups) - 1:
            out.append(Text(""))
    return Group(*out)


# ---------- buttons ----------
BUTTONS = [("1", "Chats"), ("2", "Tasks"), ("3", "Memory"), ("4", "Help")]

def button_bar():
    t = Text()
    for i, (k, label) in enumerate(BUTTONS):
        t.append(f" {k} ", style="bold #0b1d26 on #3fc1e8")
        t.append(f" {label} ", style=f"{SOFT} on #16394a")
        t.append("  ")
    return t


def info_box(cfg, mode, backend, sid, agent):
    g = Table.grid(padding=(0, 2))
    g.add_column(style=DIM, justify="right")
    g.add_column()
    g.add_row("Model", f"[bold {SOFT}]{escape(cfg.model)}[/]  [dim]chat[/]")
    if mode == "sandbox":
        g.add_row("Environment", f"[bold {SOFT}]Docker sandbox[/]  [dim]isolated, no network, resource-limited[/]")
    else:
        g.add_row("Environment", f"[bold {SOFT}]Local machine[/]  [dim]{escape(backend.root)}[/]")
    g.add_row("Session", f"[bold {SOFT}]{sid}[/]")
    g.add_row("Safety", "[dim]writes auto-approved, deletes ask[/]" if agent.auto else "[dim]every write and delete asks first[/]")
    return g


def show_banner(cfg, mode, backend, sid, agent):
    c.print()
    c.print(Panel(Group(title_block(max(20, c.size.width - 8)), Text(""),
                        Align.center(Text("Local AI agent  ·  files  ·  memory", style=DIM)),
                        Text(""), Align.center(info_box(cfg, mode, backend, sid, agent))),
                  box=box.ROUNDED, border_style=ACCENT, padding=(1, 3),
                  title=f"[bold {ACCENT}] MedTech [/]", subtitle=f"[{DIM}] v0.2 [/]"))
    c.print(Align.center(button_bar()))
    c.print(Align.center(Text("type a number or /help for commands  ·  exit to leave", style=DIM)))
    c.print()


def ts(x):
    return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else ""


def _table(title):
    return Table(title=title, box=box.SIMPLE_HEAD, border_style=ACCENT, header_style=f"bold {ACCENT}")


# ---------- history views ----------
def show_sessions(store):
    t = _table("Sessions")
    for col in ("id", "mode", "updated", "title", "msgs", "tasks"):
        t.add_column(col)
    for s in store.sessions():
        t.add_row(str(s["id"]), s["mode"], ts(s["updated_at"]), escape(s["title"] or "New chat"),
                  str(s["n_msgs"]), str(s["n_tasks"]))
    c.print(t)


def show_history(store, sid):
    s = store.session(sid)
    if not s:
        return c.print(f"[red]no session {sid}[/]")
    c.print(f"[{ACCENT}]Session {sid}  ·  {s['mode']}  ·  {ts(s['started_at'])}[/]")
    for m in store.messages(sid):
        who = f"[bold {SOFT}]you[/]" if m["role"] == "user" else f"[bold {ACCENT}]medtech[/]"
        c.print(f"{who} [dim]{ts(m['ts'])}[/]  {escape(m['content'])}")


def show_tasks(store, sid=None):
    t = _table("Tasks" + (f" (session {sid})" if sid else " (all sessions)"))
    for col in ("id", "session", "mode", "time", "tool", "args", "risk", "status"):
        t.add_column(col)
    for r in store.tasks(sid):
        t.add_row(str(r["id"]), str(r["session_id"]), r["mode"], ts(r["ts"]), r["tool"],
                  escape(r["args"][:60]), r["risk"], r["status"])
    c.print(t)


# ---------- UI object handed to the agent ----------
class UI:
    def say(self, text):
        c.print(f"[{ACCENT}]{escape(text)}[/]")

    def warn(self, text):
        c.print(f"[yellow]! {escape(text)}[/]")

    def action(self, tool, args, d):
        col = {"read": ACCENT, "write": "yellow", "danger": "red"}[d.risk]
        c.print(f"[dim {col}]> {tool} {escape(json.dumps(args, ensure_ascii=False)[:200])}  [{d.risk}][/]")

    def confirm(self, tool, args, d):
        extra = f" ({d.why})" if d.why else ""
        return Confirm.ask(f"[{SOFT}]Allow this {d.risk} action?{escape(extra)}[/]", default=False)

    def result(self, ok, text):
        c.print(text if ok else f"[red]{escape(text)}[/]", markup=False if ok else True, highlight=False)


# ---------- main ----------
def build_parser():
    p = argparse.ArgumentParser(prog="medtech", description="MedTech terminal agent")
    p.add_argument("--mode", choices=["sandbox", "local"], help="skip the question")
    p.add_argument("--workspace", default="~/medtech-workspace", help="initial local directory; also the default root in workspace permission mode")
    p.add_argument("--permissions", choices=["full", "workspace"], help="local filesystem scope (config default: full)")
    p.add_argument("--model", help="Ollama chat model (default qwen2.5:7b or MEDTECH_MODEL)")
    p.add_argument("--num-ctx", type=int, help="model context size, 8192..32768 (MEDTECH_NUM_CTX or 16384)")
    p.add_argument("--plain", action="store_true", help="use the line-oriented interface instead of the Textual UI")
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=int, nargs="?", const=-1, metavar="ID", help="resume ID, or latest chat if omitted")
    resume.add_argument("--continue", dest="continue_latest", action="store_true", help="resume latest chat")
    p.add_argument("--debug", action="store_true", help="log tool arguments and raw results to MEDTECH_HOME/debug.log")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("sessions", help="list all sessions")
    h = sub.add_parser("history", help="show a session's chat")
    h.add_argument("id", type=int, nargs="?", help="omit to list chats")
    t = sub.add_parser("tasks", help="show executed actions")
    t.add_argument("id", nargs="?", default="all")
    return p


def ask_mode():
    g = Table.grid(padding=(0, 2))
    g.add_column(justify="right")
    g.add_column()
    g.add_row(Text(" 1 ", style="bold #0b1d26 on #3fc1e8"),
              Text.from_markup(f"[bold {SOFT}]Sandbox[/]  [dim]isolated Docker container, no network, resource-limited[/]"))
    g.add_row(Text(" 2 ", style="bold #0b1d26 on #3fc1e8"),
              Text.from_markup(f"[bold {SOFT}]Local[/]    [dim]your own computer, using the configured filesystem permissions[/]"))
    c.print(Panel(g, box=box.ROUNDED, border_style=ACCENT, padding=(1, 3),
                  title=f"[bold {ACCENT}] Select environment [/]"))
    while True:
        s = Prompt.ask("Where do you want to work? (sandbox / local / exit)", default="sandbox").strip().lower()
        if s in ("exit", "quit", "q"):
            return None
        if s in ("sandbox", "s", "1"):
            return "sandbox"
        if s in ("local", "l", "2"):
            return "local"
        c.print("[yellow]type sandbox, local or exit[/]")


def make_backend(mode: str, workspace: str = None, cfg=None):
    """Open the recorded workspace without changing the user's filesystem location."""
    cfg = cfg or Config()
    if mode == "sandbox":
        backend = SandboxBackend(workspace=workspace, extra_mounts=cfg.sandbox.extra_mounts)
        with c.status("starting sandbox ..."):
            backend.ensure()
        return backend
    initial = (cfg.permissions.root if cfg.permissions.mode == "workspace" else None) or workspace or "~/medtech-workspace"
    return LocalBackend(initial, permissions_mode=cfg.permissions.mode,
                        confirm_dangerous=cfg.permissions.confirm_dangerous)


def new_chat(agent: Agent, store: Store) -> None:
    """Create a fresh conversation in the active execution environment."""
    workspace = getattr(agent.backend, "host_root", agent.backend.root)
    sid = store.new_session(agent.backend.name, workspace)
    agent.resume(sid)
    c.print(f"[dim]New chat {sid}[/]")


def resume_chat(agent: Agent, store: Store, sid: int) -> None:
    """Restore a conversation and its original backend before replacing active state."""
    session = store.session(sid)
    if session is None:
        raise ValueError(f"no session {sid}")
    backend = make_backend(session["mode"], session["workspace"], getattr(agent, "cfg", None))
    agent.backend = backend
    agent.resume(sid)
    c.print(f"[dim]Resumed chat {sid} in {escape(backend.where())}[/]")


def show_memories(store: Store) -> None:
    """Display editable durable facts independently of conversation history."""
    t = _table("Memory")
    for col in ("id", "fact", "value"):
        t.add_column(col)
    for row in store.memories():
        t.add_row(str(row["id"]), escape(row["key"]), escape(row["value"]))
    c.print(t)


def command(line: str, agent: Agent, store: Store) -> None:
    """Handle a slash command; invalid arguments never terminate the chat."""
    try:
        parts = shlex.split(line)
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]
        arg = " ".join(args)
        if cmd == "/help":
            c.print(HELP)
        elif cmd in ("/sessions", "/chats", "/history") and not args:
            show_sessions(store)
        elif cmd == "/history" and len(args) == 1:
            show_history(store, int(args[0]))
        elif cmd == "/new" and not args:
            new_chat(agent, store)
        elif cmd == "/resume" and len(args) == 1:
            resume_chat(agent, store, int(args[0]))
        elif cmd == "/delete" and len(args) == 1:
            sid = int(args[0])
            if not store.delete_session(sid):
                raise ValueError(f"no session {sid}")
            if agent.sid == sid:
                new_chat(agent, store)
            c.print(f"[dim]Deleted chat {sid}[/]")
        elif cmd == "/rename" and len(args) >= 2:
            sid = int(args[0])
            if store.session(sid) is None:
                raise ValueError(f"no session {sid}")
            store.rename_session(sid, " ".join(args[1:]))
            c.print(f"[dim]Renamed chat {sid}[/]")
        elif cmd == "/memory" and not args:
            show_memories(store)
        elif cmd == "/memory" and arg == "clear":
            store.clear_memories()
            c.print("[dim]Memory cleared[/]")
        elif cmd == "/remember" and arg:
            store.remember(arg)
            c.print("[dim]Remembered[/]")
        elif cmd == "/forget" and arg:
            count = store.forget_memory(arg)
            c.print(f"[dim]Forgot {count} fact(s)[/]")
        elif cmd == "/tasks":
            show_tasks(store, None if arg == "all" else int(arg) if arg else agent.sid)
        elif cmd == "/permissions":
            if agent.backend.name != "local":
                c.print("Docker stays restricted to its configured bind mounts.")
            else:
                if arg in ("full", "workspace"):
                    agent.cfg.permissions.mode = arg
                    agent.backend.set_permissions(arg, agent.cfg.permissions.root)
                    agent.cfg.save_preferences()
                elif len(args) == 2 and args[0] == "confirm" and args[1] in ("on", "off"):
                    agent.cfg.permissions.confirm_dangerous = args[1] == "on"
                    agent.backend.confirm_dangerous = agent.cfg.permissions.confirm_dangerous
                    agent.cfg.save_preferences()
                elif arg:
                    raise ValueError("Use /permissions full|workspace or /permissions confirm on|off")
                c.print(f"Permissions: {agent.backend.permissions_mode}; cwd: {agent.backend.where()}")
        elif cmd == "/debug" and arg in ("", "on", "off"):
            if arg:
                agent.debug = arg == "on"
            c.print(f"debug logging: {'on' if agent.debug else 'off'} (MEDTECH_HOME/debug.log)")
        elif cmd == "/auto" and arg in ("on", "off"):
            agent.auto = arg == "on"
            c.print(f"auto-approve safe writes: {'on' if agent.auto else 'off'}")
        else:
            c.print("[yellow]unknown command or invalid arguments, try /help[/]")
    except (ValueError, BackendError, OSError) as exc:
        c.print(f"[red]{escape(str(exc))}[/]")


def button(key: str, agent: Agent, store: Store) -> None:
    """Map the terminal shortcut buttons to the same slash commands."""
    command({"1": "/chats", "2": "/tasks", "3": "/memory", "4": "/help"}[key], agent, store)


def _main(argv=None):
    a = build_parser().parse_args(argv)
    store = Store()
    try:
        if a.cmd == "sessions":
            return show_sessions(store)
        if a.cmd == "history":
            return show_history(store, a.id) if a.id is not None else show_sessions(store)
        if a.cmd == "tasks":
            return show_tasks(store, None if a.id == "all" else int(a.id))
        use_tui = not a.plain and sys.stdin.isatty() and sys.stdout.isatty()
        if c.is_terminal and not use_tui:
            draw_logo()
        cfg = Config()
        if a.permissions:
            cfg.permissions.mode = a.permissions
        if a.model:
            cfg.model = a.model
        if a.num_ctx is not None:
            cfg.num_ctx = a.num_ctx
            cfg.__post_init__()
        resumed = None
        if a.resume is not None or a.continue_latest:
            resumed = store.latest_session() if a.resume in (None, -1) else store.session(a.resume)
            if resumed is None:
                return c.print("[red]No matching chat to resume.[/]")
        mode = resumed["mode"] if resumed else (a.mode or ask_mode())
        if mode is None:
            return c.print("[dim]Goodbye.[/]")
        workspace = resumed["workspace"] if resumed else a.workspace if mode == "local" else None
        backend = make_backend(mode, workspace, cfg)
        if use_tui:
            from .tui import MedTechApp
            return MedTechApp(cfg, backend, store, sid=resumed["id"] if resumed else None,
                              auto=(mode == "sandbox"), debug=a.debug).run()
        sid = resumed["id"] if resumed else store.new_session(mode, getattr(backend, "host_root", backend.root))
        agent = Agent(cfg, backend, store, sid, UI(), auto=(mode == "sandbox"), debug=a.debug)
        show_banner(cfg, mode, backend, sid, agent)
        while True:
            try:
                q = Prompt.ask(f"[bold {ACCENT}]{agent.backend.name}[/] [dim]{escape(agent.backend.where())}[/] [bold {ACCENT}]>[/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "/exit"):
                break
            if q.lower() in ("m", "/menu"):
                c.print(Align.center(button_bar()))
                continue
            try:
                if q in dict(BUTTONS):
                    button(q, agent, store)
                elif q.startswith("/"):
                    command(q, agent, store)
                else:
                    agent.turn(q)
            except LLMError as exc:
                c.print(f"[red]LLM error: {escape(str(exc))}[/]")
            except KeyboardInterrupt:
                c.print("[yellow]interrupted; messages already saved[/]")
        c.print(f"[dim]Goodbye. Session {agent.sid} saved  ·  `medtech history {agent.sid}`[/]")
    finally:
        store.close()


def main(argv=None):
    try:
        _main(argv)
    except (StoreError, BackendError, OSError, ValueError) as exc:
        c.print(f"[red]Error: {escape(str(exc))}[/]")
        return 1
    except (KeyboardInterrupt, EOFError):
        c.print("\n[dim]Goodbye.[/]")
