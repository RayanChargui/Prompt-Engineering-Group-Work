"""medtech - terminal agent. Pick sandbox (Docker) or local, chat, RAG over your documents."""
import argparse
import json
import shlex
import threading
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .agent import Agent, Config, LLMError, chat, make_embedder
from .backends import BackendError, LocalBackend, SandboxBackend
from .logo import draw_logo
from .rag import RAG
from .store import Store

c = Console()
ACCENT, SOFT, DIM = "#3fc1e8", "#e8fbff", "#2b6f87"
HELP = f"""[bold {ACCENT}]Chat[/]      say what you want: files, folders, ip config, questions about your documents
[bold {ACCENT}]/ingest <path>[/]   add a PDF / text file or a folder to the knowledge base
[bold {ACCENT}]/docs[/]  list documents      [bold {ACCENT}]/forget <id>[/]  remove one      [bold {ACCENT}]/rag on|off[/]  use documents in answers
[bold {ACCENT}]/auto on|off[/]   auto-approve safe writes (deletes and overwrites always ask)
[bold {ACCENT}]/sessions[/]  all sessions      [bold {ACCENT}]/history [id][/]  one chat      [bold {ACCENT}]/tasks [id|all][/]  executed actions
[bold {ACCENT}]/menu[/] or [bold {ACCENT}]m[/]  show the buttons      [bold {ACCENT}]/help[/]      [bold {ACCENT}]exit[/]  leave"""

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
BUTTONS = [("1", "Sessions"), ("2", "Tasks"), ("3", "Add documents"),
           ("4", "Knowledge base"), ("5", "Toggle RAG"), ("6", "Help")]


def button_bar():
    t = Text()
    for i, (k, label) in enumerate(BUTTONS):
        t.append(f" {k} ", style="bold #0b1d26 on #3fc1e8")
        t.append(f" {label} ", style=f"{SOFT} on #16394a")
        t.append("  ")
    return t


def info_box(cfg, mode, backend, sid, rag, agent):
    g = Table.grid(padding=(0, 2))
    g.add_column(style=DIM, justify="right")
    g.add_column()
    g.add_row("Model", f"[bold {SOFT}]{escape(cfg.model)}[/]  [dim]chat[/]   [bold {SOFT}]{escape(cfg.embed_model)}[/]  [dim]embeddings[/]")
    if mode == "sandbox":
        g.add_row("Environment", f"[bold {SOFT}]Docker sandbox[/]  [dim]isolated, no network, resource-limited[/]")
    else:
        g.add_row("Environment", f"[bold {SOFT}]Local machine[/]  [dim]{escape(backend.root)}[/]")
    g.add_row("Session", f"[bold {SOFT}]{sid}[/]")
    g.add_row("Knowledge base", f"[bold {SOFT}]{len(rag.docs())}[/] documents  [dim]RAG {'on' if agent.rag_on else 'off'}[/]")
    g.add_row("Safety", "[dim]writes auto-approved, deletes ask[/]" if agent.auto else "[dim]every write and delete asks first[/]")
    return g


def show_banner(cfg, mode, backend, sid, rag, agent):
    c.print()
    c.print(Panel(Group(title_block(max(20, c.size.width - 8)), Text(""),
                        Align.center(Text("Local AI agent  ·  files  ·  documents  ·  RAG", style=DIM)),
                        Text(""), Align.center(info_box(cfg, mode, backend, sid, rag, agent))),
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
    for col in ("id", "mode", "started", "first message", "msgs", "tasks"):
        t.add_column(col)
    for s in store.sessions():
        t.add_row(str(s["id"]), s["mode"], ts(s["started_at"]), escape(s["title"] or ""),
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

    def sources(self, hits):
        c.print("[dim]sources: " + ", ".join(f"{h.source}" + (f" p.{h.page}" if h.page else "") +
                                            f" ({h.score:.2f})" for h in hits) + "[/]")

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
    p.add_argument("--workspace", default="~/medtech-workspace", help="folder the agent may touch in local mode")
    p.add_argument("--model", help="Ollama chat model")
    p.add_argument("--resume", type=int, metavar="ID", help="continue a previous session")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("sessions", help="list all sessions")
    h = sub.add_parser("history", help="show a session's chat")
    h.add_argument("id", type=int)
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
              Text.from_markup(f"[bold {SOFT}]Local[/]    [dim]your own computer, restricted to one workspace folder[/]"))
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


def _main(argv=None):
    a = build_parser().parse_args(argv)
    store = Store()
    if a.cmd == "sessions":
        return show_sessions(store)
    if a.cmd == "history":
        return show_history(store, a.id)
    if a.cmd == "tasks":
        return show_tasks(store, None if a.id == "all" else int(a.id))

    if c.is_terminal:
        draw_logo()  # animation first
    cfg = Config()
    if a.model:
        cfg.model = a.model
    hist, resumed = [], None
    if a.resume:
        resumed = store.session(a.resume)
        if not resumed:
            return c.print(f"[red]no session {a.resume}[/]")
    mode = resumed["mode"] if resumed else (a.mode or ask_mode())
    if mode is None:
        return c.print("[dim]Goodbye.[/]")

    try:
        if mode == "sandbox":
            backend = SandboxBackend()
            with c.status("starting sandbox ..."):
                backend.ensure()
        else:
            backend = LocalBackend(a.workspace)
    except BackendError as e:
        return c.print(f"[red]{e}[/]")

    if resumed:
        sid = resumed["id"]
        hist = [{"role": m["role"], "content": m["content"] if m["role"] == "user"
                 else json.dumps({"say": m["content"], "action": None})} for m in store.messages(sid)]
    else:
        sid = store.new_session(mode, backend.root)

    rag = RAG(store, make_embedder(cfg))
    ui = UI()
    agent = Agent(cfg, backend, store, sid, rag, ui, auto=(mode == "sandbox"), hist=hist)
    threading.Thread(target=lambda: _warm(cfg), daemon=True).start()
    show_banner(cfg, mode, backend, sid, rag, agent)

    while True:
        try:
            q = Prompt.ask(f"[bold {ACCENT}]{mode}[/] [dim]{escape(backend.where())}[/] [bold {ACCENT}]>[/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "/exit"):
            break
        if q.lower() in ("m", "/menu"):
            c.print(Align.center(button_bar()))
            continue
        if q in dict(BUTTONS):
            try:
                button(q, agent, store, rag)
            except KeyboardInterrupt:
                c.print("[yellow]cancelled[/]")
            continue
        if q.startswith("/"):
            try:
                command(q, agent, store, rag, backend)
            except KeyboardInterrupt:
                c.print("[yellow]cancelled[/]")
            continue
        try:
            agent.turn(q)
        except LLMError as e:
            c.print(f"[red]LLM error: {escape(str(e))}[/]")
            agent.hist.pop()
        except KeyboardInterrupt:
            c.print("[yellow]interrupted[/]")
    c.print(f"[dim]Goodbye. Session {sid} saved  ·  `medtech history {sid}`[/]")


def main(argv=None):
    try:
        _main(argv)
    except (KeyboardInterrupt, EOFError):
        c.print("\n[dim]Goodbye.[/]")


def _warm(cfg):
    try:
        chat(cfg, [{"role": "user", "content": 'reply {"say":"hi","action":null}'}])
    except Exception:  # noqa: BLE001 - warm-up is best effort
        pass


def ingest(rag, path):
    path = path.strip().strip('"\'')
    try:
        with c.status("embedding documents ..."):
            res = rag.ingest(path)
        for name, status in res:
            c.print(f"[{ACCENT}]+[/] {escape(name)}  [dim]{status}[/]")
        if not res:
            c.print("[yellow]nothing to ingest (supported: pdf, txt, md, csv, json, log, py)[/]")
    except Exception as e:  # noqa: BLE001
        c.print(f"[red]ingest failed: {escape(str(e))}[/]  [dim](embedding model: `ollama pull nomic-embed-text`)[/]")


def show_docs(rag):
    t = _table("Knowledge base")
    for col in ("id", "name", "chunks", "path"):
        t.add_column(col)
    for d in rag.docs():
        t.add_row(str(d["id"]), d["name"], str(d["n"]), escape(d["path"]))
    c.print(t)


def button(key, agent, store, rag):
    if key == "1":
        show_sessions(store)
    elif key == "2":
        show_tasks(store, agent.sid)
    elif key == "3":
        p = Prompt.ask("Path to a PDF, file or folder (empty to cancel)", default="")
        if p.strip():
            ingest(rag, p)
    elif key == "4":
        show_docs(rag)
    elif key == "5":
        agent.rag_on = not agent.rag_on
        c.print(f"RAG: {'on' if agent.rag_on else 'off'}")
    elif key == "6":
        c.print(HELP)


def command(line, agent, store, rag, backend):
    try:
        parts = shlex.split(line, posix=False)
    except ValueError:
        parts = line.split()
    cmd, args = parts[0].lower(), [x.strip('"\'') for x in parts[1:]]
    arg = " ".join(args)
    if cmd == "/help":
        c.print(HELP)
    elif cmd == "/sessions":
        show_sessions(store)
    elif cmd == "/history":
        show_history(store, int(arg) if arg.isdigit() else agent.sid)
    elif cmd == "/tasks":
        show_tasks(store, None if arg == "all" else int(arg) if arg.isdigit() else agent.sid)
    elif cmd == "/auto":
        agent.auto = arg == "on"
        c.print(f"auto-approve safe writes: {'on' if agent.auto else 'off'}")
    elif cmd == "/rag":
        agent.rag_on = arg != "off"
        c.print(f"RAG: {'on' if agent.rag_on else 'off'}")
    elif cmd == "/docs":
        show_docs(rag)
    elif cmd == "/forget" and arg.isdigit():
        rag.forget(int(arg))
        c.print("removed")
    elif cmd == "/ingest" and arg:
        ingest(rag, arg)
    else:
        c.print("[yellow]unknown command, try /help[/]")
