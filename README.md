# MedTech

A terminal assistant using a local Ollama model, typed filesystem tools, SQLite
conversations, and durable user facts. Runtime dependencies: Python 3.9+ and
`rich`. No cloud account or paid API is needed.

## Setup and launch

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ollama serve
# In another terminal, pull the exact model tag installed on your machine:
ollama pull qwen3.5:7b
medtech --mode local --workspace ~/medtech-workspace
```

The requested default model is `qwen3.5:7b`. Ollama tag availability varies; check
`ollama list` and explicitly override with `--model TAG` or `MEDTECH_MODEL=TAG`
when needed. The app does not silently substitute models. `OLLAMA_URL` defaults
to `http://127.0.0.1:11434`. Once the model and optional Docker image are installed,
the assistant can run offline. Startup does not contact Ollama until a user turn.

Use `medtech --mode sandbox` for Docker, or omit `--mode` to choose interactively.
Docker must be installed and its daemon running. The sandbox uses
`python:3.12-alpine`; run `docker pull python:3.12-alpine` before going offline.

## Conversations and memory

A launch starts a new chat by default. User input is committed before a model
request; model replies, proposed tool calls, results, and rejected output are
committed as they arrive. Ctrl+C cannot undo already committed messages. A
conversation title comes from the first user message, without another model call.

| Command | Behavior |
| --- | --- |
| `/new` | Start a fresh chat in the current workspace |
| `/history`, `/chats`, `/sessions` | List chats, most recently updated first |
| `/history ID` | Show the persisted transcript |
| `/resume ID` | Restore a chat and its original execution workspace |
| `/rename ID TITLE` | Rename a chat |
| `/delete ID` | Delete a chat and its message/tool history |
| `/memory` | List durable user facts and their IDs |
| `/remember FACT` | Save a fact explicitly; `name=Rayan` sets a stable property |
| `/forget ID` or `/forget TEXT` | Remove a fact by ID, exact property, or exact value |
| `/memory clear` | Remove all durable facts |
| `/tasks [ID\|all]` | Inspect the execution audit log |
| `/auto on\|off` | Auto-approve ordinary writes; destructive actions still ask |
| `/help`, `exit` | Show help or leave |

Startup forms: `medtech --resume ID`, `medtech --resume` (latest chat), and
`medtech --continue` (latest chat). Noninteractive views:
`medtech sessions`, `medtech history ID`, and `medtech tasks [ID|all]`.

Data lives in `~/.medtech/medtech.db`; set `MEDTECH_HOME` to use a different data
directory. SQLite schema migrations preserve conversation records. Back up this
file before upgrading. A corrupt or unwritable database produces a clear error;
the application does not replace it with an empty database.

Facts are stored separately from chat messages in that database. A lightweight
local-model extraction request runs for each user turn, accepting only a validated
JSON list of property/value pairs. The same property replaces its earlier value;
malformed extraction leaves memory unchanged. Facts are included as delimited
reference data in every model context, including brand-new chats. `/remember`
also works without Ollama. Extraction quality depends on the model; use
`/memory`, `/remember`, and `/forget` to review or correct its choices. Plain
free-form manual facts deduplicate identical text; use stable `key=value` facts
when a newer value should replace an older one. Deleting a chat does not erase
cross-session facts; forgetting a fact does not erase its original chat message.

Full transcripts remain on disk. Model context has a conservative byte-based
budget within the configured context window, a bounded synopsis of older
messages, and recent messages. Oversized message text is explicitly shortened
for model input, preserving the saved original. The synopsis is extractive and
may omit old details; durable facts belong in memory.

## Execution and paths

The agent proposes one JSON action per model reply. A strict parser and security
policy validate it before a backend runs one of eleven typed tools. There is no
arbitrary shell tool. Results include `success`, `exit_code`, `stdout`, `stderr`,
`cwd`, and `duration`, plus verified filesystem paths for writes. Results return
to the model and are saved in SQLite. The tools perform read-back/stat checks
inside the same execution environment before reporting created files or folders.
Malformed responses trigger corrective retries. Unsupported completion claims
are rejected with a bounded retry limit; action narration is not displayed as a
completed operation. The English-language claim guard is defensive and cannot
prove arbitrary natural-language statements true; inspect the tool evidence.

Local paths resolve relative to the backend's retained current directory, starting
at the real absolute `--workspace` path. The path jail resolves symlinks and `..`;
private credential paths and leaving the workspace are blocked. `cd` changes
subsequent typed-tool resolution, not your terminal's shell directory.

The sandbox runs one persistent container per host workspace. Its mapping is:

```text
host:      ~/medtech-files/             (default shared directory)
container: /workspace/
example:   ~/medtech-files/test_dir/hello.txt
        -> /workspace/test_dir/hello.txt
```

The container has a fixed `/workspace` workdir. Its bind mount retains files
between tool calls and container restarts. On Unix it runs as the invoking
UID:GID so host-created files are accessible to that user. Existing containers
are reused only after their mount and isolation settings are validated; an
incompatible container is reported without deleting it. A path-derived container
name avoids unrelated workspaces sharing a single global sandbox.

The container has no network, all capabilities dropped, a read-only root filesystem,
a writable temporary directory, and limits of 512 MB RAM, one CPU, and 128 processes.
Only the shared bind mount is persistent and writable. File output reports
absolute paths within the environment that actually executed it. Sandbox paths
refer to `/workspace`, not an unrelated host directory. Docker bind mounts require
a locally accessible daemon filesystem; remote daemons need matching shared paths.

Reads are automatic; local writes require confirmation unless `/auto on` is set.
Sandbox ordinary writes are automatic. Deletes, overwrites, and writes following
untrusted file contents still require confirmation. File data is delimited as
untrusted input and never grants extra tool permissions.

`medtech --debug ...` prints each proposed tool call, arguments, and raw structured
result to stderr. Debug output and SQLite history may contain file contents or
personal data; keep them private.

## Validation

Run the deterministic acceptance suite (no Ollama required):

```sh
python -m pytest -q
python -m pytest -q -m 'not docker'   # deliberately omit daemon-dependent tests
```

Tests use a scripted model interface, actual temporary local files, and a mocked
Docker transport. Tests marked `docker` auto-skip when the daemon is unavailable;
when it is available they execute against the real sandbox image. Image download
or runtime failures are reported rather than counted as passes.

On your own machine, start Ollama, install the selected model, start Docker, and run:

```sh
scripts/verify_local.sh
# Or choose an exact installed tag:
MEDTECH_MODEL=YOUR_TAG python scripts/verify_local.py
```

The verification script prints PASS/FAIL for each required check and exits nonzero
on failure. It uses the real model for persistence, name recall after a process
restart and in a new chat, and folder/file creation in local and Docker modes.
It checks the actual file contents from the execution environment and host,
reopens the sandbox to check persistence, and also exercises chat/memory commands,
malformed-call rejection, unsupported-claim detection, operation failure, and
corrupt database handling. It uses a temporary data directory and unique test
container; your chats and normal sandbox are untouched. Missing Ollama, model,
or Docker is a failure in this local verification script, not a skipped success.
This may take several minutes on CPU. The script does not install model weights
or change your Docker configuration.
