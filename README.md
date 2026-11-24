# MedTech — local terminal AI assistant

MedTech is a Python terminal assistant powered by a model running through
[Ollama](https://ollama.com/). You can chat, manage files inside a chosen workspace,
keep conversations across restarts, and save facts the assistant can use in new chats.
File operations run either on your computer or in a persistent Docker sandbox.

The application needs no cloud account, API key, subscription, or paid service.
With a **local Ollama endpoint** and the model/image already downloaded, it can run
offline. The model generates responses; MedTech controls which tools actually run.

## Contents

- [What the AI can do](#what-the-ai-can-do)
- [Requirements](#requirements)
- [Installation and first launch](#installation-and-first-launch)
- [Everyday use](#everyday-use)
- [OS terminal command reference](#os-terminal-command-reference)
- [Commands inside MedTech](#commands-inside-medtech)
- [Conversations and memory](#conversations-and-memory)
- [Local mode, Docker mode, and paths](#local-mode-docker-mode-and-paths)
- [Permissions and execution results](#permissions-and-execution-results)
- [Configuration](#configuration)
- [Data, privacy, backup, and updates](#data-privacy-backup-and-updates)
- [Troubleshooting](#troubleshooting)
- [Tests and real-service verification](#tests-and-real-service-verification)
- [How it works and project map](#how-it-works-and-project-map)

## What the AI can do

Ask in ordinary language; you do not need to write JSON or memorize tool names.

| Capability | Example to type inside MedTech |
| --- | --- |
| General conversation, explanations, and drafting | `Explain Python decorators with a small example.` |
| Create folders and text files | `Create folder test_dir with hello.txt containing exactly hi.` |
| Read text files | `Read test_dir/hello.txt.` |
| List files and navigate | `List this workspace, then change into test_dir.` |
| Search filenames or text | `Find files whose names contain notes.` |
| Move and copy files or directories | `Copy notes.txt to notes-backup.txt.` |
| Delete files or directories, with confirmation | `Delete notes-backup.txt.` |
| Inspect the execution environment | `Show the OS version and free disk space.` |
| Remember user facts | `My name is Rayan. I prefer explanations in French.` |
| Return to earlier conversations | Use `/history` and `/resume ID`. |

The available tools are deliberately limited. The assistant **cannot execute
arbitrary shell commands, run Python programs, install packages, browse the web,
or control other applications**. It can draft code as text or save it in a file;
saving code does not execute it. For live filesystem questions, ask it to inspect
the workspace rather than guess. General knowledge comes from the installed model
and may be incomplete or outdated.

Only files inside the active workspace are accessible through its file tools.
System information describes the environment executing the tool: in sandbox mode,
that is the container, not necessarily your host computer.

## Requirements

| Component | Needed for |
| --- | --- |
| Python **3.9 or newer**, with `venv` and `pip` | Running MedTech |
| Ollama and an installed compatible chat model | AI replies and automatic memory extraction |
| `rich` | Terminal interface; installed automatically with MedTech |
| Docker with a reachable daemon | Sandbox mode and real Docker tests only |
| `pytest` | Development tests; installed through the `dev` extra |

Install Python from [python.org](https://www.python.org/downloads/), Ollama from
[ollama.com/download](https://ollama.com/download), and Docker from
[Docker's installation guide](https://docs.docker.com/get-started/get-docker/).
On Linux, your distribution may package Python's `venv` support separately.

The configured default model is **`qwen3.5:7b`**, matching the project's requested
Qwen 3.5 7B setup. This is a model tag, not a bundled download. Tag availability
must be checked on your machine; use an exact available/installed tag with
`--model` or `MEDTECH_MODEL` if it differs. MedTech does not silently substitute a
model or download weights for you.

Model weights and inference need several GB of disk space and memory; the actual
requirements depend on the tag, quantization, and context size. GPU acceleration
is managed by Ollama. CPU inference can be slow, and each user turn can involve
several model requests. Docker's 512 MB limit applies to the **tool container**;
Ollama runs separately and needs its own resources.

## Installation and first launch

### 1. Open the repository and create a Python environment

Run these in your **OS terminal**, outside MedTech. If you already have the checkout,
start with `cd Prompt-Engineering-Group-Work` in its parent directory.

```sh
git clone https://github.com/RayanChargui/Prompt-Engineering-Group-Work.git
cd Prompt-Engineering-Group-Work
python3 -m venv .venv
```

On Linux/macOS, activate and install:

```sh
. .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, create/activate and install with:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If PowerShell blocks activation, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m medtech --help
```

For the test suite as well, install the development extra instead:

```sh
python -m pip install -e '.[dev]'
```

`-e` installs the checkout in editable mode, so Python source changes take effect
without reinstalling. Reinstall after dependency or package metadata changes.
In a new terminal, activate this environment again before using `medtech`.

### 2. Start Ollama and install your model

Ollama may already be running as a desktop app or background service. If it is
not running, open a separate terminal and start it:

```sh
ollama serve
```

Leave that process running. In another terminal:

```sh
ollama list
ollama pull qwen3.5:7b
ollama run qwen3.5:7b
```

The pull command requires an available tag and Internet access. If it reports
that the tag does not exist, select a compatible tag available through Ollama and
use that same tag when launching MedTech. `ollama run` is an optional direct model
check; use Ollama's `/bye` command to leave that chat. Its commands are separate
from MedTech's commands below.

### 3. Launch MedTech

From the activated Python environment:

```sh
medtech --mode local --workspace ~/medtech-workspace
```

Or, with a different installed model tag:

```sh
medtech --mode local --model YOUR_INSTALLED_TAG --workspace ~/medtech-workspace
```

`python -m medtech` is an equivalent entry point if the `medtech` executable is
not on your PATH. On PowerShell, an explicit workspace can be written as
`--workspace "$HOME\medtech-workspace"`.

A missing workspace directory is created automatically. This is where file tools
work; it is independent of the repository containing the application source.

For sandbox mode, start Docker first, then run:

```sh
docker info
docker pull python:3.12-alpine
medtech --mode sandbox
```

To choose the execution mode interactively, just run `medtech`. The mode prompt
accepts `sandbox`/`s`/`1`, `local`/`l`/`2`, or `exit`/`quit`/`q`.

## Everyday use

There are **two different command locations**:

- **OS terminal:** installation, `ollama`, `docker`, `medtech --help`, and launch commands.
- **Inside MedTech's chat prompt:** ordinary-language requests and `/` commands.

Typing `mkdir reports` in the OS terminal runs your shell's command. Typing it
inside MedTech sends text to the model; it is not a direct shell command. Prefer
clear requests such as `Create a folder named reports.` There is no `/mkdir` or
`/shell` shortcut.

Here is a first session. Enter one line at a time and wait for the response:

```text
My name is Rayan.
/memory
Create folder test_dir with hello.txt inside it, containing exactly hi.
Read test_dir/hello.txt and list the folder.
/tasks
/rename 1 First experiment
/history
exit
```

Replace `1` with the actual session ID shown by your app. Local file creation asks
for permission; review the proposed tool and arguments, then answer `y` to allow
or `n` to refuse. The default is refusal. Denying an action stops the current turn.

To inspect the resulting file yourself in a Linux/macOS OS terminal:

```sh
cat ~/medtech-workspace/test_dir/hello.txt
```

On PowerShell:

```powershell
Get-Content "$HOME\medtech-workspace\test_dir\hello.txt"
```

For reliable results, give exact paths, exact file contents, and a short task.
Break large jobs into smaller requests. Existing files are not silently overwritten;
ask explicitly if you want overwrite or append behavior, then confirm it.

`exit`, `quit`, and `/exit` leave MedTech. Ctrl+C during a turn interrupts it while
keeping messages already saved; at the input prompt Ctrl+C exits. There is no
separate save command. An interrupted turn is not automatically rerun on resume.

## OS terminal command reference

Run these **outside** the chat. Global flags go before subcommands. `ID` and
`YOUR_INSTALLED_TAG` are placeholders to replace, not literal values.

| Command | Purpose |
| --- | --- |
| `medtech` | Start a new chat and choose local or sandbox mode |
| `medtech --help` | Show supported startup flags and subcommands |
| `medtech --mode local --workspace PATH` | Start a local chat in that directory |
| `medtech --mode sandbox` | Start a Docker-backed chat |
| `medtech --model YOUR_INSTALLED_TAG` | Override the model for this launch |
| `medtech --resume ID` | Resume the specified chat |
| `medtech --resume` | Resume the most recently updated chat |
| `medtech --continue` | Same latest-chat behavior |
| `medtech --debug --mode local` | Launch with detailed tool diagnostics on stderr |
| `medtech sessions` | List up to 50 most recently updated chats, then exit |
| `medtech history` | List chats, then exit |
| `medtech history ID` | Print the saved transcript, then exit |
| `medtech tasks` | Print the latest 200 tasks across chats, then exit |
| `medtech tasks ID` | Print task records for one chat, then exit |
| `medtech tasks all` | Explicit form of the all-chat task view |

| Startup flag | Default / details |
| --- | --- |
| `--mode {local,sandbox}` | Prompt if omitted for a new chat |
| `--workspace PATH` | `~/medtech-workspace`; **local mode only** |
| `--model TAG` | `MEDTECH_MODEL`, otherwise `qwen3.5:7b` |
| `--resume [ID]` | Optional numeric ID; omitted ID selects latest |
| `--continue` | Select latest; mutually exclusive with `--resume` |
| `--debug` | Off; logs tool calls, full arguments, and raw results |
| `-h`, `--help` | Print help |

Resume uses the saved chat's mode and workspace, overriding supplied `--mode` or
`--workspace`. Neither `--resume` nor `--continue` changes the selected model;
use `--model` if you want a particular model when resuming.

Useful Ollama/Docker commands, also run in the OS terminal:

```sh
ollama list                         # installed model tags
ollama ps                           # models currently loaded by Ollama
ollama pull YOUR_INSTALLED_TAG       # download/update a model
ollama run YOUR_INSTALLED_TAG        # test that model directly
ollama stop YOUR_INSTALLED_TAG       # unload it from Ollama memory
docker info                         # check daemon access
docker images                       # list cached images
docker ps -a --filter "name=medtech-sandbox-"
```

`exit` from MedTech does not stop Ollama or the persistent Docker container.
To release container resources, first identify its exact name using the last
command, then run `docker stop CONTAINER_NAME`. A later sandbox launch starts a
compatible stopped container again; bind-mounted files remain on disk.

## Commands inside MedTech

Use these at the **MedTech prompt**, without a `medtech` prefix. For multiword
arguments, plain text usually works; quotes can group names. Slash commands use
shell-style quoting, so an apostrophe in a fact can be enclosed in double quotes.

| Command | Behavior / example |
| --- | --- |
| `/help` | Show the built-in command help |
| `/new` | Open a new chat in the current execution environment |
| `/history`, `/chats`, `/sessions` | List up to 50 chats, most recently updated first |
| `/history ID` | Show that chat's persisted messages and tool data |
| `/resume ID` | Switch to an existing chat and its recorded workspace |
| `/rename ID TITLE` | Rename it, e.g. `/rename 12 Python project` |
| `/delete ID` | Delete that chat and its message/task records |
| `/memory` | List durable facts with their memory IDs |
| `/remember FACT` | Explicitly save a fact, e.g. `/remember name=Rayan` |
| `/forget ID` | Delete the memory with that memory ID |
| `/forget TEXT` | Delete a memory by exact property or exact value |
| `/memory clear` | Delete all durable facts |
| `/tasks` | Show tasks for the current chat |
| `/tasks ID` | Show tasks for another chat |
| `/tasks all` | Show the latest 200 tasks across chats |
| `/auto on` | Auto-approve ordinary writes, subject to the rules below |
| `/auto off` | Ask before ordinary writes |
| `/menu`, `m` | Display the shortcut buttons |
| `exit`, `quit`, `/exit` | Leave MedTech |

At the chat prompt, shortcut `1` lists chats, `2` shows current tasks, `3` lists
memory, and `4` shows help. These mappings differ from the startup mode prompt.

**Chat IDs and memory IDs are separate.** `/delete 3` deletes chat 3; `/forget 3`
deletes memory 3. Neither command deletes files created in a workspace. Deleting
the active chat immediately opens a new one. There is no undo command for chat
or memory deletion.

## Conversations and memory

### Conversation history

Every ordinary launch creates a new session. A short title is taken from the
first user message. Sessions include timestamps, mode, workspace, messages,
proposed tool calls, results, and audit records. Each message is committed to disk
as it is recorded; user input is committed before the first model request.
Rejected model output and corrective messages are retained for diagnosis too, so
the history view can include raw JSON and internal retry messages.

Use `/history` to find an ID and `/resume ID` to return to it, or exit and run
`medtech --continue`. Latest means **most recently updated**, not necessarily the
highest ID. Deletion retires a session ID rather than reusing it for a new chat.

Resuming reloads the full transcript from disk, but only a bounded context is
sent to the model. The context includes the system instructions, selected durable
facts, an extractive synopsis of older entries, and recent messages. Large entries
are shortened for model input; their original versions remain in SQLite. The
synopsis may omit older details and is not a perfect record of everything said.

A fresh backend on resume starts at the workspace root: previous `cd` state is
not restored after restart or `/resume`. `/new` reuses the active backend and its
current directory. Ask `Show the current directory` if unsure.

### Durable facts across chats

After you send ordinary chat text, a separate lightweight request to the same
local model extracts explicitly stated durable facts. This happens before the
main reply. Extraction accepts only a validated JSON schema, with at most 10
facts per response and values up to 500 characters; malformed output or a model
failure leaves memory unchanged. New values for the same property replace old
ones. This is saved reference data, not model training or a change to its weights.

Use explicit facts when you need predictable updates without a model request:

```text
/remember name=Rayan
/remember preference_language=French
/remember project=Local terminal assistant
/memory
/new
What is my name and what language do I prefer?
```

You can change a property with `/remember name=Ray`. Free-form manual facts also
work, but only identical text deduplicates reliably; use stable `key=value`
properties for facts that can change. Automatic extraction depends on the model,
so inspect `/memory` rather than assuming it remembered correctly.

Remove a fact by ID, exact property, or exact value:

```text
/forget name
/forget French
/memory clear
```

Forgetting is not a substring search. Numeric text is interpreted as a memory ID.
Memory is shared by all sessions using the same data directory. Deleting a chat
does not erase its saved facts; forgetting a fact does not erase the original
message in a chat. Old chat text can still influence a resumed conversation.
A new chat avoids that old transcript. The prompt's memory budget prioritizes
recently updated facts, so an unusually large memory collection may not all fit.

Manual memory commands and history views work without Ollama. AI replies and
automatic fact extraction require Ollama.

## Local mode, Docker mode, and paths

| Detail | Local | Sandbox |
| --- | --- | --- |
| Tool environment | Your computer, under your user account | Linux Docker container |
| Workspace on host | `--workspace PATH`, default `~/medtech-workspace` | `~/medtech-files` |
| Workspace seen by tools | Actual absolute host directory | `/workspace` |
| Ordinary write approval on fresh launch | Ask | Automatic |
| Docker required | No | Yes |
| File persistence | Host files remain | Host bind-mounted files remain |

### Local paths

Paths resolve relative to the backend's current directory. The application uses
real absolute paths, resolving `..` and symlinks, and denies paths outside the
workspace. An absolute path is allowed only if it remains inside that workspace.
The `~` shorthand works for the startup `--workspace` option, but **not** for paths
passed to file tools. Within chat, use workspace-relative paths such as
`reports/notes.txt` instead of `~/reports/notes.txt`.

A tool `cd` affects subsequent tools, not the directory of your OS shell. Starting
`medtech` from the repository does not automatically make that repository the
file workspace. To work there intentionally, launch with `--workspace .`.

### Sandbox mapping and lifecycle

```text
Host computer                         Container
~/medtech-files/                  ->  /workspace/
~/medtech-files/test_dir/hello.txt ->  /workspace/test_dir/hello.txt
```

The container path in a tool result refers to this mapping. Look in
`~/medtech-files` on your host, not your local-mode workspace. The CLI currently
ignores `--workspace` in sandbox mode; custom sandbox bind paths are available only
through the Python backend API.

MedTech uses a long-lived container with a path-derived name such as
`medtech-sandbox-<hash>`. Each tool call uses `docker exec`, with an explicit working
directory. Compatible stopped containers are restarted; existing mount, user,
image, and isolation settings are checked before reuse. Files persist between
calls, sessions, and container restarts through the host bind mount.

The container runs with no network, all capabilities dropped, no extra privileges,
a read-only root filesystem, writable temporary space, and limits of 512 MB RAM,
one CPU, and 128 processes. On Unix it runs with your host UID:GID. The shared
workspace is writable and persistent; temporary container files are not a backup.
Ollama runs outside the container, so the container's disabled network does not
prevent the Python application from contacting your local Ollama service.

Docker Desktop must allow access to the shared host directory. A remote Docker
daemon needs that directory available on the daemon's machine as well. Native
Windows path/permission behavior can differ; WSL with a local Docker integration
is another option. The automated Docker checks should be run on your setup.

## Permissions and execution results

The model can propose only these eleven typed tools; users normally request them
in natural language rather than calling them directly:

| Tool | Purpose and important options |
| --- | --- |
| `pwd` | Show the actual current directory |
| `ls` | List a directory; optional `path` |
| `cd` | Change the tool directory with `path` |
| `read_file` | Read a text file at `path` |
| `search` | Search a `query`, optional `path`, and `kind` of `name` or `content` |
| `sysinfo` | `kind`: `os`, `network`, `whoami`, `disk`, or `date` |
| `write_file` | `path`, text `content`, and `mode`: `create`, `overwrite`, or `append` |
| `mkdir` | Create a directory at `path`, including missing parents |
| `move` | Move `src` to `dst`; an existing directory destination receives the source basename |
| `copy` | Copy `src` to `dst`, with the same destination rule |
| `delete` | Delete `path`; `recursive` is needed for a nonempty directory |

The rules are enforced by application code:

- Reads are automatic. A fresh local launch asks before ordinary writes; a fresh
  sandbox launch automatically allows them.
- `/auto on` changes ordinary write approval only. **Delete, overwrite, and append
  always ask**, and an existing destination is not silently replaced by copy/move.
- After reading file contents or search results in a turn, subsequent writes in
  that turn require confirmation because the input could contain instructions.
- `/auto` is runtime state, not a saved per-chat preference. `/new` and interactive
  `/resume` retain the current setting, even when resuming another execution mode.
  Use `/auto off` when you want to require ordinary write approval again.
- Leaving the workspace and accessing protected credential names/directories
  such as `.env`, `.ssh`, and `.aws` is denied by the path policy.

Tools return a structured result with `success`, `exit_code`, `stdout`, `stderr`,
`cwd`, and `duration` in seconds. Filesystem mutations also return `verified`
evidence, such as an absolute path and whether it exists. Writes read the file
back; directory operations check filesystem state in the same environment.
A zero exit code indicates success; nonzero results carry the actual error.
The task log distinguishes `done`, `failed`, `blocked`, and `denied`.

Action narration is not displayed as completed work before a tool runs. Malformed
model replies and unsupported completion claims trigger corrective retries. The
loop stops after at most 12 model-response steps or 3 corrective failures. If it
stops, successful earlier operations remain; there is no automatic rollback.
Inspect `/tasks` and the actual filesystem before retrying a partially completed
job. The English-language completion guard is defensive and cannot prove every
possible natural-language claim true.

Some tool output is deliberately bounded: text reads cover up to 20,000 bytes,
file writes accept up to 100,000 characters, directory listings show up to 200
entries, and searches return up to 50 hits. Binary files are not supported by the
text reader. Fixed network-info commands may be missing in a minimal container.

## Configuration

Set environment variables **before launching** MedTech. The app does not load a
`.env` file. There are no API keys to configure for local Ollama.

| Variable | Default | Use |
| --- | --- | --- |
| `MEDTECH_MODEL` | `qwen3.5:7b` | Exact Ollama model tag; `--model` takes precedence |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint |
| `MEDTECH_HOME` | `~/.medtech` | Directory holding the SQLite database |

Linux/macOS example:

```sh
export MEDTECH_MODEL=YOUR_INSTALLED_TAG
export OLLAMA_URL=http://127.0.0.1:11434
export MEDTECH_HOME="$HOME/.medtech"
medtech --mode local
```

PowerShell example:

```powershell
$env:MEDTECH_MODEL = "YOUR_INSTALLED_TAG"
$env:OLLAMA_URL = "http://127.0.0.1:11434"
$env:MEDTECH_HOME = "$HOME\.medtech"
python -m medtech --mode local
```

These examples set variables for the current terminal session and its child
processes. A different `MEDTECH_HOME` gives you separate chats and memory; it does
not move existing data or change the file workspace.

Keep `OLLAMA_URL` local for local-only processing. If you explicitly point it at
another machine, prompts, selected memories, and tool output go to that endpoint.
The application has no built-in authentication configuration for a hosted service.

Model requests currently use temperature `0`, a 16,384-token context setting,
and up to 1,000 generated tokens per response. Context/output settings are Python
`Config` fields, not CLI flags or additional environment variables. Budgeting uses
a conservative byte estimate without a separate tokenizer. Large responses may
need smaller tasks or a developer adjustment to `Config`.

## Data, privacy, backup, and updates

| Item | Default location |
| --- | --- |
| Chat messages, facts, summaries, and task audit | `~/.medtech/medtech.db` |
| Local workspace files | `~/medtech-workspace/` |
| Sandbox workspace files visible on the host | `~/medtech-files/` |
| Ollama model weights | Managed separately by Ollama |
| Python dependencies | The virtual environment you installed into |

The SQLite database is not encrypted by MedTech. It can contain user messages,
file contents returned by tools, arguments, and rejected model output. Automatic
extraction is instructed to omit secrets, but this is not a guarantee that a
message or fact is safe to retain. Review `/memory` and avoid submitting secrets.
Workspace path restrictions do not encrypt logs or protect files from other
programs running as your user.

To back up the database, exit MedTech and other processes using that database,
then copy it to a backup location. For the default Linux/macOS location:

```sh
cp "$HOME/.medtech/medtech.db" ./medtech-backup.db
```

For PowerShell:

```powershell
Copy-Item "$HOME\.medtech\medtech.db" .\medtech-backup.db
```

Use the actual directory if you set `MEDTECH_HOME`, and choose a backup filename
you want to keep. Back up workspace files separately: the database is not a copy
of all files created or edited by the assistant. To restore, stop the app,
preserve the current database, and copy a known-good backup to `medtech.db` in
the configured data directory. The app reports corrupt/incompatible databases;
it does not replace them with empty ones.

Schema migrations run when the database opens. Before updating, back up your data
and commit or otherwise preserve your own repository changes. Then update the
checkout using your normal Git workflow and refresh dependencies:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

Use `python -m pip install -e .` if you do not want development dependencies.
Updating Python code does not update model weights; `ollama pull TAG` handles that
separately. Uninstalling the Python package does not erase your database,
workspaces, Docker image, or Ollama downloads.

## Troubleshooting

| Symptom | What to check or do |
| --- | --- |
| `medtech: command not found` | Activate the same virtual environment used to install it, or run `python -m medtech`. |
| `No module named medtech` / missing `rich` | From the checkout, run `python -m pip install -e .` with the intended interpreter. |
| `venv` or `ensurepip` is unavailable | Install your OS's Python virtual-environment support, then recreate `.venv`. |
| Cannot reach Ollama / connection refused | Start the Ollama app/service or `ollama serve`; check `OLLAMA_URL`. |
| Ollama says the address is already in use | Check whether Ollama is already running instead of launching another server. |
| Model not found / HTTP error | Run `ollama list`; pull an available tag and pass it with `--model TAG`. |
| Slow response or timeout | Test with `ollama run TAG`, inspect `ollama ps`, use a model your hardware supports, and make requests smaller. A turn may need multiple calls. |
| Invalid JSON / retry or step limit reached | The model did not produce a valid supported response, or completion lacked evidence. Inspect `/tasks`; retry with a shorter explicit request. Earlier writes may already exist. |
| It says a task failed | Inspect `stderr`, `exit_code`, and the absolute path. Check permissions and whether the destination already exists. |
| Cannot find a created file | Check the tool's verified path and active mode. Local defaults to `~/medtech-workspace`; sandbox files appear under `~/medtech-files`. |
| Outside workspace / protected path denied | Choose a path inside the workspace. A broader local workspace must be chosen at launch in a new chat; blocked credential paths remain blocked. |
| Writes still ask after `/auto on` | Deletes, append/overwrite, and writes following untrusted file reads still need confirmation. |
| Docker missing or daemon inaccessible | Start Docker Desktop/daemon and run `docker info` from the same terminal/user. Local mode works without Docker. |
| Docker image pull fails | Pre-pull `python:3.12-alpine` with network access; check registry access and disk space. |
| Sandbox bind mount / permission error | Check host `~/medtech-files` permissions, Docker Desktop sharing, and whether the daemon is local. |
| Incompatible existing container | Inspect the named container and its mounts. MedTech leaves it unchanged. After preserving needed data, remove/recreate only that identified container or use local mode. |
| Network-info tool unavailable | Minimal images may lack `ip`/`ifconfig`; a reported tool failure is expected until the required OS utility exists. |
| No chat found to resume | Run `medtech sessions` and check `MEDTECH_HOME`; a new data directory has no old sessions. |
| Memory missing or wrong | Inspect `/memory`; correct it with `/remember key=value`. Check that sessions use the same data directory. |
| Forgotten fact still appears in an old chat | The original conversation remains. Forgetting removes the durable fact, not its prior messages. |
| Database corrupt, locked, unwritable, or newer than supported | Stop competing processes, check directory permissions, and preserve the file. Restore a known-good backup or use the matching app version; do not discard the only copy. |

For tool diagnostics:

```sh
medtech --debug --mode local --workspace ~/medtech-workspace
# Capture stderr to a file if you need to inspect it:
medtech --debug --mode local 2>medtech-debug.log
```

Debug logging includes full proposed arguments and raw results, potentially with
file contents or personal data. Review and redact logs before sharing them.

## Tests and real-service verification

Install the development dependencies first. From the repository root:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

The suite tests agent logic with a scripted LLM, SQLite persistence and migrations,
CLI commands, real temporary local files, and mocked Docker transport behavior.
It does not require Ollama. Tests marked `docker` use the actual daemon and
sandbox image, and automatically skip if the daemon is unavailable. If the daemon
is present, image/runtime failures count as failures rather than skipped passes.

To intentionally select or omit Docker integration tests:

```sh
python -m pytest -q -m docker
python -m pytest -q -m 'not docker'
```

For end-to-end checks with **real Ollama and Docker**, first start both services,
install your chosen model, and cache the sandbox image. Then run:

```sh
python scripts/verify_local.py
# Linux/macOS shell wrapper:
scripts/verify_local.sh
```

To select an installed model on Linux/macOS:

```sh
MEDTECH_MODEL=YOUR_INSTALLED_TAG python scripts/verify_local.py
```

On PowerShell, set `$env:MEDTECH_MODEL = "YOUR_INSTALLED_TAG"`, then run the Python
command. The verifier uses `MEDTECH_MODEL`, not a `--model` option.

The verifier prints a PASS/FAIL per check and exits nonzero on failure. It covers
real model responses, saved sessions, name recall after process restart and in a
new chat, local and Docker folder/file creation, actual host/container read-back,
chat/memory commands, malformed calls, unsupported claims, operation failure, and
corrupt database handling. It uses temporary data and a uniquely named test
container, then cleans up its own resources. It does not install model weights
or change your normal chat data. Missing Ollama/model/Docker is a **failed check**
in this verifier, not a skipped success. CPU runs may take several minutes.

Implementation validation in the cloud development environment completed with
**66 pytest tests passed and none skipped**, including real Docker tests. Ollama
was unavailable there: the real-service verifier reported **2 passed and 6 failed**
due to missing Ollama or dependent prerequisites. Those results do not establish
that your selected model or platform works; run the verifier on your machine.

## How it works and project map

```text
Your message -> SQLite commit -> local model fact extraction
             -> model proposes JSON answer/action
             -> strict parser -> approval/security policy
             -> local tools OR docker exec in the persistent container
             -> structured result + filesystem verification -> SQLite
             -> result returned to model -> next action or checked final answer
```

The LLM suggests; the application decides whether execution is allowed. Tool
output and facts are marked as untrusted reference data in model context. The
same standard-library operation code runs locally and in Docker so path and
verification rules stay consistent. Model behavior remains probabilistic even
with temperature zero; the checks limit execution claims rather than making
all generated answers correct.

| File / directory | Responsibility |
| --- | --- |
| `medtech/__main__.py` | `python -m medtech` entry point |
| `medtech/cli.py` | Terminal UI, startup options, and slash commands |
| `medtech/agent.py` | Ollama HTTP client, config, context budgeting, strict parsing, and agent loop |
| `medtech/policy.py` | Tool schemas, approval rules, and untrusted-text handling |
| `medtech/ops.py` | Shared filesystem/system operations and post-execution checks |
| `medtech/backends.py` | Local execution and Docker lifecycle/mount/transport |
| `medtech/store.py` | SQLite schema versions, migrations, sessions, messages, tasks, and facts |
| `medtech/memory.py` | Validated fact extraction and bounded memory prompt |
| `medtech/logo.py` | Terminal startup animation |
| `tests/` | Offline acceptance tests and optional real Docker tests |
| `scripts/verify_local.py` | Real-service acceptance checks |
| `scripts/verify_local.sh` | Shell launcher for the verifier |
| `pyproject.toml` | Package metadata, dependencies, executable entry point, and pytest marker |

For development, `Agent` accepts an injectable callable LLM interface. A fake
client receives the message list and returns raw JSON text, allowing deterministic
malformed-response and hallucination tests without a running model. Runtime
uses the standard-library HTTP client against Ollama's `/api/chat` endpoint.
