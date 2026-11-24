"""Small modal dialogs for conversation and memory management."""
from typing import Callable, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from .store import Store


class Dialog(ModalScreen):
    """Centered, keyboard-dismissable modal base."""
    BINDINGS = [("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    Dialog { align: center middle; background: $background 65%; }
    Dialog > Vertical { width: 64; max-width: 95%; height: auto; max-height: 85%; border: round $accent; padding: 1 2; background: $surface; }
    Dialog Horizontal { height: auto; margin-top: 1; }
    Dialog Button { margin-right: 1; }
    Dialog Label { margin-bottom: 1; }
    """

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class RenameDialog(Dialog):
    """Edit a title without altering the conversation content."""
    def __init__(self, title: str):
        super().__init__()
        self.title_value = title

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Rename conversation")
            yield Input(self.title_value, id="rename-title", max_length=200)
            with Horizontal():
                yield Button("Save", variant="primary", id="rename-save")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Button.Pressed, "#rename-save")
    @on(Input.Submitted, "#rename-title")
    def save(self) -> None:
        title = self.query_one(Input).value.strip()
        if title:
            self.dismiss(title)


class DeleteDialog(Dialog):
    """Require explicit confirmation before permanently deleting a chat."""
    def __init__(self, title: str):
        super().__init__()
        self.title_value = title

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Delete conversation?")
            yield Static(self.title_value, markup=False)
            yield Static("Messages and task history will be deleted. Workspace files and memory stay.")
            with Horizontal():
                yield Button("Delete", variant="error", id="delete-confirm")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#delete-confirm")
    def delete(self) -> None:
        self.dismiss(True)


class ChatMenuDialog(Dialog):
    """Accessible keyboard alternative to the conversation context menu."""
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Conversation actions")
            yield Button("Rename  F2", id="chat-rename")
            yield Button("Delete", variant="error", id="chat-delete")
            yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#chat-rename")
    def rename(self) -> None:
        self.dismiss("rename")

    @on(Button.Pressed, "#chat-delete")
    def delete(self) -> None:
        self.dismiss("delete")


class ConfirmToolDialog(Dialog):
    """Resolve a worker's approval request on the UI thread."""
    def __init__(self, tool: str, details: str):
        super().__init__()
        self.tool, self.details = tool, details

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Allow {self.tool}?")
            yield Static(self.details, markup=False)
            with Horizontal():
                yield Button("Allow", variant="warning", id="tool-allow")
                yield Button("Deny", id="cancel")

    @on(Button.Pressed, "#tool-allow")
    def allow(self) -> None:
        self.dismiss(True)


class MemoryItem(ListItem):
    """One fact with an explicit per-fact delete control."""
    def __init__(self, memory_id: int, key: str, value: str):
        self.memory_id = memory_id
        super().__init__(Horizontal(Static(f"{key}: {value}", markup=False),
                                    Button("Forget", id=f"forget-{memory_id}", variant="error")))


class MemoryDialog(Dialog):
    """Browse and remove durable facts without leaving the current chat."""
    DEFAULT_CSS = """
    MemoryDialog > Vertical { width: 78; height: 75%; }
    MemoryDialog ListView { height: 1fr; }
    MemoryDialog ListItem { height: auto; padding: 1; }
    MemoryDialog ListItem Horizontal { margin: 0; }
    MemoryDialog ListItem Static { width: 1fr; }
    MemoryDialog ListItem Button { min-width: 10; width: 10; }
    """
    def __init__(self, store: Store, on_mutation: Optional[Callable[[], None]] = None):
        super().__init__()
        self.store = store
        self.on_mutation = on_mutation

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Known facts about you")
            yield ListView(id="memory-list")
            yield Static("Use /remember key=value to add or update a fact.")
            yield Button("Close", id="memory-close")

    async def on_mount(self) -> None:
        await self.refresh_facts()

    async def refresh_facts(self) -> None:
        facts = self.query_one("#memory-list", ListView)
        await facts.clear()
        rows = self.store.memories()
        await facts.extend([MemoryItem(row["id"], row["key"], row["value"]) for row in rows])
        if not rows:
            await facts.append(ListItem(Static("No saved memories."), disabled=True))

    @on(Button.Pressed)
    async def pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "memory-close":
            self.dismiss(None)
        elif button_id.startswith("forget-"):
            self.store.forget_memory(button_id[7:])
            if self.on_mutation is not None:
                self.on_mutation()
            await self.refresh_facts()
