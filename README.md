# medtech 0.2

Terminal agent: Ollama LLM + two execution modes + local RAG + full history.

## Setup
    ollama pull qwen2.5-coder:7b
    ollama pull nomic-embed-text      # embeddings for RAG
    pip install -e .
    medtech                           # asks: sandbox (Docker) or local

Flags: `--mode sandbox|local`, `--workspace PATH` (local jail, default ~/medtech-workspace), `--model NAME`, `--resume ID`.

## Use
- Chat: "make a folder reports and a file notes.txt inside", "show my ip config", "what's the warranty of the Nova X?"
- `/ingest <pdf|folder>`, `/docs`, `/forget <id>`, `/rag on|off`
- `/sessions`, `/history [id]`, `/tasks [id|all]` (also `medtech sessions|history ID|tasks [ID]`)
- `/auto on|off` auto-approve safe writes (destructive actions always ask)

History lives in `~/.medtech/medtech.db` (sessions from both modes, chats, tasks, RAG index).

## Security design
1. No free-form shell: the model can only call 11 typed tools with strict schemas (`policy.py`).
2. Jail: every path is resolved (symlinks, `..`) and must stay in the workspace; `.env`, `.ssh`... blocked (`ops.py`).
3. Risk levels: read = auto, write = confirm (auto in sandbox), delete/overwrite = always confirm.
4. Taint tracking: once documents or file contents are in the model's context, writes need confirmation.
5. Untrusted text is wrapped in `<untrusted>` tags, sanitised, scanned for injection phrases; flagged RAG chunks are dropped.
6. Sandbox: Docker with no network, all capabilities dropped, read-only rootfs, 512 MB / 1 CPU / 128 pids, only a named volume writable.
7. Every action (including blocked/denied) is logged in `tasks`.

## Tests
    python tests/test_core.py
