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
