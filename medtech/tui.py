"""Responsive Textual chat UI over the independently testable agent core."""
import asyncio
import json
import re
import traceback
import shlex
import threading
from datetime import datetime, timedelta
from typing import Optional

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.strip import Strip
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Footer, Input, Label, ListItem, ListView, LoadingIndicator, Markdown, Static, Switch, TextArea

from .agent import Agent, Config, LLMError, chat
from .backends import LocalBackend, SandboxBackend
from .memory import extract_memories
from .presentation import brief_error, concise_text, tool_outcome
from .themes import apply_theme, theme_names
from .store import Store, StoreError
from .tui_dialogs import ChatMenuDialog, ConfirmToolDialog, DeleteDialog, MemoryDialog, RenameDialog, ThemeDialog


class MessageInput(TextArea):
    """Show an empty-editor hint without putting placeholder text in user input."""
    def render_line(self, y: int) -> Strip:
        if not self.text and y == 0:
            hint = "Message MedTech...  Ctrl+Enter to send" if self.app.cfg.ui.ascii else "Message MedTech…  Ctrl+Enter to send"
            return Strip([Segment(hint, self.rich_style + Style(dim=True))]).adjust_cell_length(self.scrollable_content_region.width)
        return super().render_line(y)


class ChatItem(ListItem):
    """A sidebar entry carrying a persistent session id."""
    def __init__(self, sid: int, title: str, current: bool = False, ascii_only: bool = False):
        self.sid, self.full_title, self.ascii_only = sid, title, ascii_only
        super().__init__(Static(Text(title, no_wrap=True, overflow="ellipsis")),
                         id=f"chat-{sid}", classes="current" if current else "")
        self.tooltip = title

    def on_resize(self, event: events.Resize) -> None:
        width = max(1, event.size.width - 3)
        title = Text(self.full_title, no_wrap=True)
        if self.ascii_only and title.cell_len > width:
            title.truncate(max(0, width - 3), overflow="crop")
            title.append("." * min(3, width))
        else:
            title.truncate(width, overflow="ellipsis")
        self.query_one(Static).update(title)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 3:
            event.stop()
            self.app.context_chat(self.sid)


class WorkerUI:
    """Marshal agent callbacks to the Textual thread, including modal approvals."""
    def __init__(self, app: "MedTechApp"):
        self.app = app

    def dispatch(self, function, *args):
        if self.app._ui_closing or self.app.cancel_event.is_set():
            raise InterruptedError("Turn cancelled.")
        return self.app.call_from_thread(function, *args)

    def say(self, text: str) -> None:
        self.dispatch(self.app.finish_draft, text)

    def delta(self, text: str) -> None:
        self.dispatch(self.app.append_delta, text)

    def discard_draft(self) -> None:
        if not self.app._ui_closing:
            self.app.call_from_thread(self.app.discard_draft)

    def status(self, text: str) -> None:
        self.dispatch(self.app.set_status, text)

    def saved(self) -> None:
        self.dispatch(self.app.refresh_history)

    message_saved = saved

    def warn(self, text: str) -> None:
        self.dispatch(self.app.show_warning, text)

    def action(self, tool: str, args: dict, decision: object) -> None:
        self.dispatch(self.app.set_status, f"Running {tool}…")
        self.dispatch(self.app.add_tool_call, tool, args)

    def result(self, ok: bool, text: str) -> None:
        self.dispatch(self.app.add_tool_result, ok, text)

    def confirm(self, tool: str, args: dict, decision: object) -> bool:
        answer = threading.Event()
        state = {"allowed": False}
        self.app._pending_approvals.append(answer)

        def show() -> None:
            def resolved(value) -> None:
                state["allowed"] = value is True
                answer.set()
            details = json.dumps(args, indent=2, ensure_ascii=False)
            why = getattr(decision, "why", "")
            self.app.set_status("Waiting for your approval")
            self.app.push_screen(ConfirmToolDialog(tool, details + ("\n" + why if why else "")), resolved)

        self.dispatch(show)
        while not answer.wait(0.1):
            if self.app._ui_closing or self.app.cancel_event.is_set():
                if answer in self.app._pending_approvals:
                    self.app._pending_approvals.remove(answer)
                return False
        if answer in self.app._pending_approvals:
            self.app._pending_approvals.remove(answer)
        return state["allowed"] and not self.app.cancel_event.is_set()


class MedTechApp(App):
    """Persistent sidebar, full transcript, and threaded local-agent turns."""
    TITLE = "MedTech"
    CSS_PATH = 'styles.tcss'
    BINDINGS = [
        Binding("ctrl+d", "toggle_details", "Details", priority=True),
        Binding("ctrl+t", "theme", "Theme", priority=True),
        Binding("ctrl+n", "new_chat", "New chat", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", priority=True),
        Binding("ctrl+f", "search", "Find chat", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+enter", "send", "Send", priority=True),
        Binding("f2", "rename_chat", "Rename", priority=True),
        Binding("ctrl+o", "chat_menu", "Chat actions", priority=True),
        Binding("delete", "delete_chat", "Delete chat", show=False),
        Binding("escape", "cancel_turn", "Cancel turn", show=False),
    ]

    def __init__(self, cfg: Config, backend: object, store: Store, sid: Optional[int] = None,
                 auto: bool = False, llm=None, debug: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.cfg, self.backend, self.store = cfg, backend, store
        self.cfg.ui.theme = apply_theme(self, self.cfg.ui.theme)
        self.sid, self.auto, self.llm, self.debug_enabled = sid, auto, llm, debug
        self.details = bool(cfg.ui.details)
        self._status_text = "Ready"
        self._turn_outcome = False
        self._last_tool = ("", {})
        self._turn_traceback = ""
        self.busy = False
        self.cancel_event = threading.Event()
        self.shutdown_event = threading.Event()
        self._ui_closing = False
        self._pending_approvals = []
        self._draft_text = ""
        self._draft_widget = None
        self._refreshing = False
        self._history_refresh_again = False
        self._sidebar_user_hidden = False
        self._last_width = None
        self._memory_active = 0
        self._memory_queue = []
        self._memory_epoch = 0
        self._turn_error = None

    def chat_widget(self, selector: str, expect_type: Optional[type] = None) -> Widget:
        """Address the chat screen even while a modal is the active screen."""
        return self.screen_stack[0].query_one(selector, expect_type)

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("MEDTECH" if self.cfg.ui.ascii else "◈  MEDTECH", id="brand")
                yield Button("+ New chat" if self.cfg.ui.ascii else "＋ New chat", id="new-chat", variant="primary")
                yield Input(placeholder="Search conversations..." if self.cfg.ui.ascii else "Search conversations…", id="chat-search")
                yield ListView(id="chat-list")
                yield Button("Memory", id="memory")
                yield Button("Themes", id="themes")
                yield Static("", id="sidebar-footer", markup=False)
            with Vertical(id="main"):
                yield Static("New chat", id="chat-title", markup=False)
                yield VerticalScroll(id="transcript")
                with Horizontal(id="composer"):
                    yield MessageInput(id="message-input", soft_wrap=True)
                    yield Button("Send", id="send", variant="primary")
                with Horizontal(id="status-area"):
                    yield LoadingIndicator(id="thinking")
                    yield Static("Ready", id="status", markup=False)
                    yield Label("Details", id="details-label")
                    yield Switch(value=self.details, id="details-switch", animate=False)
                yield Static("", id="context-bar", markup=False)
        yield Footer()

    async def on_mount(self) -> None:
        self.chat_widget("#thinking", LoadingIndicator).display = False
        await self.refresh_history()
        await self.render_transcript()
        self.update_footer()
        self.chat_widget("#message-input", TextArea).focus()
        self._apply_responsive(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        if self.is_mounted:
            self._apply_responsive(event.size.width)

    def _apply_responsive(self, width: int) -> None:
        self.screen_stack[0].set_class(width < 80, "narrow")
        if self._last_width is None or (width < 80) != (self._last_width < 80):
            self.chat_widget("#sidebar").set_class(width < 80 or self._sidebar_user_hidden, "hidden")
        self._last_width = width

    def chrome(self, text: str) -> str:
        """Use portable punctuation for interface labels in ASCII mode."""
        if self.cfg.ui.ascii:
            for old, new in (("…", "..."), ("·", "|"), ("—", "-"), ("◈", "*"), ("＋", "+")):
                text = text.replace(old, new)
        return text

    def update_footer(self) -> None:
        mode = "Docker" if self.backend.name == "sandbox" else "Local"
        permission = self.cfg.permissions.mode if self.backend.name == "local" else "workspace"
        detail = "Details on" if self.details else "Details off"
        cwd = self.backend.where()
        self.chat_widget("#sidebar-footer", Static).update(self.chrome(f"{self.cfg.model}\n{mode} · {permission}\n{cwd}\n{detail}"))
        context = self.chat_widget("#context-bar", Static)
        context.update(Text(f"{self.cfg.model} | {mode} | {permission} | {detail}\n{cwd}", no_wrap=True, overflow="ellipsis"))
        context.tooltip = cwd

    def set_status(self, text: str) -> None:
        self._status_text = text
        self.chat_widget("#status", Static).update(self.chrome(text.splitlines()[0][:160]))
        self.chat_widget("#thinking", LoadingIndicator).display = self.busy
        self.update_footer()

    def save_preferences(self) -> None:
        """Persist only explicit UI/settings changes, with an actionable error."""
        try:
            self.cfg.save_preferences()
        except (OSError, ValueError) as exc:
            self.set_status(f"Could not save preferences: {exc}")

    async def set_details(self, value: bool) -> None:
        if value == self.details:
            return
        self.details = value
        self.cfg.ui.details = value
        self.chat_widget("#details-switch", Switch).value = value
        self.save_preferences()
        await self.render_transcript()
        if self._turn_error:
            await self.show_error(self._turn_error, self._turn_traceback)
        self.set_status("Details on" if value else "Details off · Ctrl+D for diagnostics")

    @on(Switch.Changed, "#details-switch")
    async def details_changed(self, event: Switch.Changed) -> None:
        await self.set_details(event.value)

    async def action_toggle_details(self) -> None:
        await self.set_details(not self.details)

    def action_theme(self) -> None:
        async def selected(value) -> None:
            if value:
                name, ascii_mode = value
                self.cfg.ui.theme = apply_theme(self, name)
                self.cfg.ui.ascii = ascii_mode
                self.chat_widget("#brand", Static).update("MEDTECH" if ascii_mode else "◈  MEDTECH")
                self.chat_widget("#new-chat", Button).label = "+ New chat" if ascii_mode else "＋ New chat"
                self.chat_widget("#chat-search", Input).placeholder = "Search conversations..." if ascii_mode else "Search conversations…"
                self.save_preferences()
                await self.render_transcript()
                self.set_status("Theme saved: " + self.cfg.ui.theme)
        self.push_screen(ThemeDialog(theme_names(), self.cfg.ui.theme, self.cfg.ui.ascii), selected)

    @on(Button.Pressed, "#themes")
    def theme_pressed(self) -> None:
        self.action_theme()

    @staticmethod
    def redundant_final(text: str, had_outcome: bool) -> bool:
        """Hide pure tool-summary repetition while retaining mixed answers/code."""
        if not had_outcome or len(text) > 700 or "```" in text:
            return False
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n", text) if part.strip()]
        action = re.compile(r"\b(creat(?:e|ed|ing)|made|mkdir|writ(?:e|ten|ing)|wrote|sav(?:e|ed)|cop(?:y|ied)|"
                            r"mov(?:e|ed)|delet(?:e|ed)|remov(?:e|ed)|failed|denied|permission|successful|done|completed)\b", re.I)
        summary = re.compile(r"(?:[/\\]|\b(?:file|folder|directory|path|permission)\b|^\s*(?:I\b|we\b|created\b|wrote\b|saved\b|copied\b|moved\b|deleted\b|done\b|completed\b|couldn.t\b|could not\b|failed\b))", re.I)
        return bool(sentences) and all(action.search(part) and summary.search(part) for part in sentences)

    async def add_outcome(self, tool: str, args: dict, result: dict) -> bool:
        text = tool_outcome(tool, args, result, ascii=self.cfg.ui.ascii)
        if text:
            await self.chat_widget("#transcript", VerticalScroll).mount(Static(text, markup=False, classes="outcome " + ("success" if result.get("success") else "failure")))
            self.chat_widget("#transcript", VerticalScroll).scroll_end(animate=False)
        return bool(text) and (tool in {"mkdir", "write_file", "move", "copy", "delete"} or not result.get("success"))

    async def show_error(self, error: str, trace: str = "") -> None:
        if self.details and trace:
            await self.chat_widget("#transcript", VerticalScroll).mount(Collapsible(Static(trace, markup=False), title="Exception details", collapsed=False))
        marker = "FAIL" if self.cfg.ui.ascii else "✗"
        cause = brief_error(error)
        await self.show_notice(f"{marker} {cause}" + ("" if self.details else " (Ctrl+D for details)"))

    async def show_warning(self, text: str) -> None:
        if self.details:
            await self.show_notice(text)
        else:
            self.set_status(text.splitlines()[0][:100] + " · Ctrl+D for details")

    def available(self) -> bool:
        if self.busy:
            self.set_status("A response is running. Cancel it with Escape before changing chats.")
            return False
        return True

    async def refresh_history(self) -> None:
        if not self.is_mounted:
            return
        if self._refreshing:
            self._history_refresh_again = True
            return
        self._refreshing = True
        try:
            listing = self.chat_widget("#chat-list", ListView)
            query = self.chat_widget("#chat-search", Input).value.casefold().strip()
            rows = [row for row in self.store.sessions(limit=1_000_000) if row["n_msgs"] > 0
                    and (not query or query in (row["title"] or "").casefold())]
            today = datetime.now().date()
            groups = {}
            for row in rows:
                age = (today - datetime.fromtimestamp(row["updated_at"]).date()).days
                group = "Today" if age <= 0 else "Yesterday" if age == 1 else "Previous 7 days" if age <= 7 else "Older"
                groups.setdefault(group, []).append(row)
            await listing.clear()
            selected = None
            for group in ("Today", "Yesterday", "Previous 7 days", "Older"):
                if group not in groups:
                    continue
                await listing.append(ListItem(Label(group), disabled=True, classes="date-group"))
                for row in groups[group]:
                    await listing.append(ChatItem(row["id"], row["title"] or "Untitled", row["id"] == self.sid, self.cfg.ui.ascii))
                    if row["id"] == self.sid:
                        selected = len(listing.children) - 1
            if selected is not None:
                listing.index = selected
            if self.sid is not None:
                row = self.store.session(self.sid)
                self.chat_widget("#chat-title", Static).update(row["title"] or "New chat" if row else "New chat")
        finally:
            self._refreshing = False
        if self._history_refresh_again:
            self._history_refresh_again = False
            await self.refresh_history()

    async def add_bubble(self, role: str, text: str) -> None:
        bubble = Vertical(Static("You" if role == "user" else "MedTech", classes="speaker"),
                          Markdown(text), classes=f"bubble {role}")
        transcript = self.chat_widget("#transcript", VerticalScroll)
        for welcome in transcript.query("#welcome"):
            await welcome.remove()
        await transcript.mount(bubble)
        transcript.scroll_end(animate=False)

    async def show_notice(self, text: str) -> None:
        await self.chat_widget("#transcript", VerticalScroll).mount(Static(text, classes="notice", markup=False))
        self.set_status(text)

    async def render_transcript(self) -> None:
        transcript = self.chat_widget("#transcript", VerticalScroll)
        await transcript.remove_children()
        self._draft_widget, self._draft_text = None, ""
        if self.sid is None:
            await transcript.mount(Vertical(
                Static("Your local assistant", id="welcome-title", markup=False),
                Static("Ask for a result. Tools verify the work.", classes="welcome-muted", markup=False),
                Static("Try a short request", classes="welcome-section", markup=False),
                Static("Create a folder called test in the current directory.\nRead a text file and explain what it contains.\nRemember that I prefer concise answers.", id="welcome-prompts", markup=False),
                Static("Ctrl+Enter  Send     Ctrl+N  New chat\nCtrl+D  Details     Ctrl+T  Theme", classes="welcome-muted", markup=False),
                id="welcome"))
            self.set_status("Ready")
            return
        had_outcome = False
        for row in self.store.messages(self.sid):
            role, content = row["role"], row["content"]
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except (ValueError, TypeError):
                metadata = {}
            if role == "tool":
                if self.details:
                    await transcript.mount(Collapsible(Static(content, markup=False), title="Tool result", collapsed=True))
                else:
                    try:
                        result = json.loads(content)
                    except (ValueError, TypeError):
                        result = {"success": False, "stderr": "Unreadable saved tool result"}
                    had_outcome = await self.add_outcome(metadata.get("tool", ""), metadata.get("args", {}), result) or had_outcome
            elif metadata.get("tool_call"):
                if self.details:
                    await transcript.mount(Collapsible(Static(content, markup=False), title="Tool call · full model response", collapsed=True))
            elif metadata.get("internal") or metadata.get("accepted") is False:
                if self.details:
                    await transcript.mount(Collapsible(Static(content, markup=False), title=self.chrome("Diagnostic · rejected/internal"), collapsed=True))
            elif role == "assistant":
                try:
                    decoded = json.loads(content)
                    text = decoded["say"] if isinstance(decoded, dict) and "say" in decoded else content
                except (ValueError, TypeError):
                    text = content
                if not self.details:
                    text = concise_text(text)
                    if self.redundant_final(text, had_outcome):
                        text = ""
                if text:
                    await self.add_bubble("assistant", text)
            else:
                had_outcome = False
                await self.add_bubble("user", content)
        transcript.scroll_end(animate=False)

    async def append_delta(self, text: str) -> None:
        self._draft_text += text
        if not self.details:
            return
        if self._draft_widget is None:
            self._draft_widget = Vertical(Static(self.chrome("Generating — unverified"), classes="draft-label"),
                                          Markdown("", id="draft-content"), id="draft", classes="bubble assistant")
            await self.chat_widget("#transcript", VerticalScroll).mount(self._draft_widget)
        await self._draft_widget.query_one(Markdown).update(self._draft_text)
        self.chat_widget("#transcript", VerticalScroll).scroll_end(animate=False)

    async def discard_draft(self) -> None:
        if self._draft_widget is not None:
            await self._draft_widget.remove()
        self._draft_widget, self._draft_text = None, ""

    async def finish_draft(self, text: str) -> None:
        await self.discard_draft()
        if not self.details:
            text = concise_text(text)
            if self.redundant_final(text, self._turn_outcome):
                return
        if text:
            await self.add_bubble("assistant", text)

    async def add_tool_call(self, tool: str, args: dict) -> None:
        self._last_tool = (tool, args)
        await self.discard_draft()
        if self.details:
            await self.chat_widget("#transcript", VerticalScroll).mount(
                Collapsible(Static(json.dumps(args, indent=2, ensure_ascii=False), markup=False), title=self.chrome("Tool call · " + tool), collapsed=True))

    async def add_tool_result(self, ok: bool, text: str) -> None:
        await self.discard_draft()
        if self.details:
            await self.chat_widget("#transcript", VerticalScroll).mount(
                Collapsible(Static(text, markup=False), title=self.chrome("Tool result · " + ("success" if ok else "failed")), collapsed=True))
        else:
            tool, args = self._last_tool
            rows = self.store.messages(self.sid) if self.sid is not None else []
            if rows and rows[-1]["role"] == "tool":
                metadata = json.loads(rows[-1]["metadata"] or "{}")
                tool, args = metadata.get("tool", tool), metadata.get("args", args)
            try:
                result = json.loads(text)
            except (ValueError, TypeError):
                result = {"success": ok, "stderr": text if not ok else "", "stdout": text if ok else ""}
            self._turn_outcome = await self.add_outcome(tool, args, result) or self._turn_outcome
        self.update_footer()

    @on(Input.Changed, "#chat-search")
    async def search_changed(self) -> None:
        await self.refresh_history()

    @on(ListView.Selected, "#chat-list")
    async def chat_selected(self, event: ListView.Selected) -> None:
        if not self._refreshing and isinstance(event.item, ChatItem):
            await self.load_session(event.item.sid)

    async def load_session(self, sid: int) -> None:
        if not self.available() or sid == self.sid:
            return
        row = self.store.session(sid)
        if row is None:
            self.set_status(f"Conversation {sid} no longer exists.")
            return
        self.busy = True
        self.chat_widget("#send", Button).disabled = True
        try:
            if row["mode"] == "sandbox":
                backend = SandboxBackend(workspace=row["workspace"], extra_mounts=self.cfg.sandbox.extra_mounts)
                self.set_status("Starting sandbox…")
                # Docker setup does not access SQLite and may block on image startup.
                await asyncio.to_thread(backend.ensure)
            else:
                backend = LocalBackend(self.cfg.permissions.root or row["workspace"], permissions_mode=self.cfg.permissions.mode,
                                       confirm_dangerous=self.cfg.permissions.confirm_dangerous)
            self.backend, self.sid = backend, sid
            self.update_footer()
            await self.refresh_history()
            await self.render_transcript()
            self.set_status(f"Resumed conversation {sid}")
        except Exception as exc:
            self.set_status(str(exc))
        finally:
            self.busy = False
            self.chat_widget("#thinking", LoadingIndicator).display = False
            self.chat_widget("#send", Button).disabled = False

    async def action_new_chat(self) -> None:
        if not self.available():
            return
        self.sid = None  # A blank draft is not saved or listed until its first message.
        await self.refresh_history()
        await self.render_transcript()
        self.chat_widget("#chat-title", Static).update("New chat")
        self.chat_widget("#message-input", TextArea).clear()
        self.chat_widget("#message-input", TextArea).focus()
        self.set_status("New chat · Enter adds a line · Ctrl+Enter sends")

    def action_toggle_sidebar(self) -> None:
        sidebar = self.chat_widget("#sidebar")
        sidebar.toggle_class("hidden")
        self._sidebar_user_hidden = sidebar.has_class("hidden")

    def action_search(self) -> None:
        self.chat_widget("#sidebar").remove_class("hidden")
        self._sidebar_user_hidden = False
        self.chat_widget("#chat-search", Input).focus()

    def selected_sid(self) -> Optional[int]:
        listing = self.chat_widget("#chat-list", ListView)
        if listing.has_focus and isinstance(listing.highlighted_child, ChatItem):
            return listing.highlighted_child.sid
        return self.sid

    def context_chat(self, sid: Optional[int] = None) -> None:
        if not self.available():
            return
        target = sid if sid is not None else self.selected_sid()
        if target is None:
            self.set_status("Send a message before editing this chat.")
            return
        def chosen(value) -> None:
            if value == "rename":
                self.rename_chat(target)
            elif value == "delete":
                self.delete_chat(target)
        self.push_screen(ChatMenuDialog(), chosen)

    def action_chat_menu(self) -> None:
        self.context_chat()

    def rename_chat(self, sid: Optional[int]) -> None:
        if not self.available() or sid is None:
            return
        row = self.store.session(sid)
        if row is None:
            return
        async def renamed(title) -> None:
            if title:
                self.store.rename_session(sid, title)
                await self.refresh_history()
                self.set_status("Conversation renamed")
        self.push_screen(RenameDialog(row["title"] or ""), renamed)

    def action_rename_chat(self) -> None:
        self.rename_chat(self.selected_sid())

    def delete_chat(self, sid: Optional[int]) -> None:
        if not self.available() or sid is None:
            return
        row = self.store.session(sid)
        if row is None:
            return
        async def deleted(confirmed) -> None:
            if confirmed:
                self.store.delete_session(sid)
                if sid == self.sid:
                    self.sid = None
                    await self.render_transcript()
                    self.chat_widget("#chat-title", Static).update("New chat")
                await self.refresh_history()
                self.set_status("Conversation deleted; workspace files unchanged")
        self.push_screen(DeleteDialog(row["title"] or "Untitled"), deleted)

    def action_delete_chat(self) -> None:
        if self.chat_widget("#chat-list", ListView).has_focus:
            self.delete_chat(self.selected_sid())

    @on(Button.Pressed, "#new-chat")
    async def new_pressed(self) -> None:
        await self.action_new_chat()

    @on(Button.Pressed, "#memory")
    def memory_pressed(self) -> None:
        self.push_screen(MemoryDialog(self.store, on_mutation=self.memory_mutated))

    @on(Button.Pressed, "#send")
    async def send_pressed(self) -> None:
        await self.action_send()

    async def action_send(self) -> None:
        editor = self.chat_widget("#message-input", TextArea)
        text = editor.text.strip()
        if text:
            await self.send_message(text)

    async def send_message(self, text: str) -> None:
        if text.strip().lower() in ("exit", "quit", "/exit"):
            self.action_quit()
            return
        if text.strip().lower() == "m":
            await self.slash_command("/menu")
            return
        if not self.available():
            return
        if text.startswith("/"):
            self.chat_widget("#message-input", TextArea).clear()
            await self.slash_command(text)
            return
        if self.store.path == ":memory:":
            self.set_status("Threaded chat needs a file-backed Store; configure MEDTECH_HOME.")
            return
        if self.sid is None:
            self.sid = self.store.new_session(self.backend.name, getattr(self.backend, "host_root", self.backend.root))
        self.cancel_event.clear()
        self._turn_error = None
        self._turn_traceback = ""
        self._turn_outcome = False
        self.busy = True
        self.chat_widget("#send", Button).disabled = True
        self.chat_widget("#message-input", TextArea).clear()
        await self.add_bubble("user", text)
        self.set_status("Thinking…")
        self.run_turn(text, self.sid)

    @work(thread=True, group="turn")
    def run_turn(self, text: str, sid: int) -> None:
        worker_store = None
        try:
            worker_store = Store(self.store.path)
            agent = Agent(self.cfg, self.backend, worker_store, sid, WorkerUI(self), auto=self.auto,
                          llm=self.llm, debug=self.debug_enabled, cancel_event=self.cancel_event,
                          memory_extractor=lambda value: self.call_from_thread(self.start_memory, value))
            agent.turn(text)
        except (Exception, InterruptedError) as exc:
            if not self._ui_closing:
                self.call_from_thread(self.turn_failed, str(exc), traceback.format_exc())
        finally:
            if worker_store is not None:
                worker_store.close()
            if not self._ui_closing:
                self.call_from_thread(self.turn_finished)

    def turn_failed(self, text: str, trace: str = "") -> None:
        self._turn_error = text
        self._turn_traceback = trace

    async def turn_finished(self) -> None:
        await self.discard_draft()
        # Re-read the entire database transcript, including calls and diagnostics.
        await self.refresh_history()
        await self.render_transcript()
        self.chat_widget("#send", Button).disabled = False
        if self._turn_error:
            await self.show_error(self._turn_error, self._turn_traceback)
        else:
            self.set_status("Ready" + (" · Updating memory…" if self._memory_active else ""))
        self.chat_widget("#message-input", TextArea).focus()
        self.busy = False
        self.chat_widget("#thinking", LoadingIndicator).display = False
        self.update_footer()

    def start_memory(self, text: str) -> None:
        if not self._ui_closing:
            self._memory_active += 1
            self._memory_queue.append((text, self._memory_epoch))
            if self._memory_active == 1:
                self.extract_memory(self._memory_queue.pop(0))

    @work(thread=True, group="memory")
    def extract_memory(self, job: tuple) -> None:
        text, epoch = job
        memory_store = None
        error, facts = None, []
        try:
            # Validate a candidate in isolated storage; only the UI thread may
            # merge it into durable memory after checking manual-edit precedence.
            memory_store = Store(":memory:")
            client = self.llm or (lambda messages: chat(self.cfg, messages, memory=True, cancel_event=self.shutdown_event))
            if extract_memories(memory_store, text, client):
                facts = [{"key": row["key"], "value": row["value"]} for row in memory_store.memories()]
        except Exception as exc:
            error = str(exc)
        finally:
            if memory_store is not None:
                memory_store.close()
            if not self._ui_closing:
                self.call_from_thread(self.memory_finished, error, epoch, facts)

    def memory_mutated(self) -> None:
        """Explicit user edits supersede every already queued extraction."""
        self._memory_epoch += 1

    async def memory_finished(self, error: Optional[str], epoch: int, facts: list) -> None:
        if not error and epoch == self._memory_epoch and facts:
            try:
                self.store.remember_many(facts)
            except StoreError as exc:
                error = str(exc)
        self._memory_active = max(0, self._memory_active - 1)
        if self._memory_queue and not self._ui_closing:
            self.extract_memory(self._memory_queue.pop(0))
        if isinstance(self.screen, MemoryDialog):
            await self.screen.refresh_facts()
        if error:
            self.set_status("Memory update failed: " + error)
        elif not self.busy:
            self.set_status("Ready" + (" · Updating memory…" if self._memory_active else ""))

    def action_cancel_turn(self) -> None:
        if self.busy:
            self.cancel_event.set()
            for pending in self._pending_approvals:
                pending.set()
            self.set_status("Cancelling… saved messages remain on disk")

    def action_quit(self) -> None:
        self._ui_closing = True
        self.shutdown_event.set()
        self.cancel_event.set()
        for pending in self._pending_approvals:
            pending.set()
        self.exit()

    def on_unmount(self) -> None:
        self._ui_closing = True
        self.shutdown_event.set()
        self.cancel_event.set()
        for pending in self._pending_approvals:
            pending.set()

    async def slash_command(self, line: str) -> None:
        """Preserve the existing slash vocabulary without importing the Rich frontend."""
        try:
            parts = shlex.split(line)
            command, args = parts[0].lower(), parts[1:]
            arg = " ".join(args)
            if command == "/details" and arg in ("", "on", "off"):
                await self.set_details(not self.details if not arg else arg == "on")
            elif command == "/theme":
                if arg:
                    self.cfg.ui.theme = apply_theme(self, arg)
                    self.save_preferences()
                    self.set_status("Theme saved: " + self.cfg.ui.theme)
                else:
                    self.action_theme()
            elif command == "/permissions":
                if not self.available():
                    return
                if arg in ("full", "workspace"):
                    self.cfg.permissions.mode = arg
                    if self.backend.name == "local":
                        self.backend.set_permissions(arg, root=self.cfg.permissions.root or self.backend.host_root)
                    self.save_preferences()
                elif len(args) == 2 and args[0] == "confirm" and args[1] in ("on", "off"):
                    self.cfg.permissions.confirm_dangerous = args[1] == "on"
                    if self.backend.name == "local":
                        self.backend.confirm_dangerous = self.cfg.permissions.confirm_dangerous
                    self.save_preferences()
                elif arg:
                    raise ValueError("Use /permissions full|workspace or /permissions confirm on|off")
                permission = self.cfg.permissions.mode if self.backend.name == "local" else "workspace (Docker)"
                self.set_status(f"Permissions: {permission}; dangerous confirmation {'on' if self.cfg.permissions.confirm_dangerous else 'off'}")
                self.update_footer()
            elif command in ("/exit", "/quit"):
                self.action_quit()
            elif command == "/new":
                await self.action_new_chat()
            elif command in ("/history", "/sessions", "/chats") and not args:
                self.action_search()
                rows = [row for row in self.store.sessions() if row["n_msgs"]]
                await self.show_notice("Saved conversations (ID · title · updated):\n" +
                                       ("\n".join(f"{row['id']} · {row['title']} · " +
                                                   datetime.fromtimestamp(row['updated_at']).strftime("%Y-%m-%d")
                                                   for row in rows) or "No saved conversations yet."))
            elif command == "/resume" and len(args) == 1:
                await self.load_session(int(arg))
            elif command == "/history" and len(args) == 1:
                row = self.store.session(int(arg))
                if row:
                    await self.show_notice("\n".join(m["role"] + ": " + m["content"] for m in self.store.messages(int(arg))))
                else:
                    self.set_status("No matching conversation")
            elif command == "/rename" and len(args) >= 2:
                if not self.store.rename_session(int(args[0]), " ".join(args[1:])):
                    raise ValueError("No matching conversation")
                await self.refresh_history()
            elif command == "/delete" and len(args) == 1:
                self.delete_chat(int(arg))
            elif command == "/memory" and not arg:
                self.push_screen(MemoryDialog(self.store, on_mutation=self.memory_mutated))
            elif command == "/memory" and arg == "clear":
                self.store.clear_memories()
                self.memory_mutated()
                self.set_status("Memory cleared")
            elif command == "/remember" and arg:
                self.store.remember(arg)
                self.memory_mutated()
                self.set_status("Fact saved")
            elif command == "/forget" and arg:
                count = self.store.forget_memory(arg)
                self.memory_mutated()
                self.set_status(f"Forgot {count} fact(s)")
            elif command == "/auto" and arg in ("on", "off"):
                self.auto = arg == "on"
                self.set_status(f"Ordinary write auto-approval {arg}")
            elif command == "/debug" and arg in ("", "on", "off"):
                if arg:
                    self.debug_enabled = arg == "on"
                self.set_status(f"Debug file logging {'on' if self.debug_enabled else 'off'} · MEDTECH_HOME/debug.log")
            elif command == "/tasks":
                rows = self.store.tasks(None if arg == "all" else int(arg) if arg else self.sid)
                await self.show_notice("\n".join(f"{row['tool']}: {row['status']} {row['output']}" for row in rows) or "No recorded tasks")
            elif command in ("/help", "/menu"):
                await self.show_notice("Ctrl+N new · Ctrl+B sidebar · Ctrl+F search · F2 rename · Ctrl+O chat actions · Ctrl+Enter send · Escape cancel · Ctrl+Q quit\n"
                                       "/new /history /chats /sessions /resume ID /rename ID TITLE /delete ID /memory /remember FACT /forget ID-or-text /memory clear /tasks [ID|all] /auto on|off /permissions full|workspace /details on|off /theme [NAME] /debug on|off /exit")
            else:
                self.set_status("Unknown command or invalid arguments. Try /help.")
        except (ValueError, StoreError, OSError) as exc:
            self.set_status(str(exc))
