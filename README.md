# FINE — ML Portfolios (Module 2&3)

## Getting Started

This project uses **[uv](https://docs.astral.sh/uv/)** to manage Python dependencies. `uv` is a fast Python package manager that replaces `pip`, `venv`, and `pip-tools` in one tool.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with Homebrew
brew install uv
```

### 2. Install dependencies

From the project root, run:

```bash
uv sync
```

**What does `uv sync` do?**

- Creates a virtual environment (`.venv/`) if one doesn't exist.
- Reads `pyproject.toml` (the project's dependency list) and `uv.lock` (the exact pinned versions).
- Installs every dependency so your environment matches the lock file exactly.
- Think of it as `pip install -r requirements.txt` but deterministic and much faster.

> You only need to re-run `uv sync` when dependencies change (e.g. after a `git pull` that updates `pyproject.toml` or `uv.lock`).

### 3. Download the data

The dataset files are too large for GitHub and are hosted on [Google Drive](https://drive.google.com/drive/folders/1zCadGVIZvnRekehDiFLPfa1ogBCuqM_V). After cloning, run:

```bash
uv run python download_data.py
```

This downloads the entire `data/` folder from Google Drive automatically. If the folder already exists locally, it will skip the download.

### 4. Run a script

```bash
uv run python <script.py>
```

For example:

```bash
uv run python build_portofolio.py
uv run python lab1/nn_main.py
```

**What does `uv run` do?**

- Automatically activates the project's `.venv` and runs the command inside it.
- You never need to manually run `source .venv/bin/activate` — `uv run` handles it for you.
- Any command after `uv run` executes with the correct Python and all installed packages available.

### Quick Reference

| Command | What it does |
|---|---|
| `uv sync` | Install / update all dependencies from the lock file |
| `uv run python script.py` | Run a Python script using the project's environment |
| `uv add <package>` | Add a new dependency to `pyproject.toml` and install it |
| `uv remove <package>` | Remove a dependency |
| `uv lock` | Regenerate the lock file after manual `pyproject.toml` edits |
