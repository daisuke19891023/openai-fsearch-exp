# Code Agent Experiments

This repository explores search strategies for code-oriented agent workflows.

## Getting started

The project is managed with [uv](https://github.com/astral-sh/uv).
Synchronize the Python environment and run the default quality checks with:

```bash
uv sync
uv run nox -s lint
uv run nox -s typing
uv run nox -s test
```

## Installing required CLI tools

Experiments rely on `ripgrep`, `fd`, and `uv`. Use the bundled installer to fetch
prebuilt binaries for your platform (Linux and macOS are supported):

```bash
uv run python -m code_agent_experiments.setup.install_tools
```

By default the binaries are placed in `~/.local/bin`. If that directory is not
already on your `PATH`, export it in your shell profile, for example:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Re-run the script with `--force` to reinstall or `--tools ripgrep fd` to install
a subset of tools.

## Running the CLI

Execute a single-shot agent search against a repository with:

```bash
uv run cae query "Find TODO markers" --repo /path/to/repository
```

By default the command uses the Responses API backend. Switch to the
Agents SDK implementation and export traces to the OpenAI dashboard by
providing the `--driver agents` flag (ensure `OPENAI_API_KEY` is set in
your environment):

```bash
uv run cae query \
  --driver agents \
  --workflow-name code-agent-experiments.demo \
  "Audit TODO, FIXME, and NOTE markers across the repository" \
  --repo /path/to/repository
```

The Agents SDK runner registers the local `ripgrep` tool, enforces the
configured tool-call limit, and emits telemetry that appears under the
specified workflow name in the Traces UI. A single run that inspects
multiple patterns (as in the example above) will surface three or more
tool executions in the trace timeline. Disable trace export locally with
`--disable-trace` when experimentation data should remain on disk.
