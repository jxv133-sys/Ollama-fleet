# Ollama Fleet

Quickstart
----------

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo (no Ollama server required):

```bash
PYTHONPATH=. python3 -m ollama_fleet.main --goal "Demo run" --demo
```

3. Run the demo with GUI (optional):

```bash
pip install textual
PYTHONPATH=. python3 -m ollama_fleet.main --goal "Demo run" --demo --ui
```

4. To use a real Ollama server, create `config.toml` with your `base_url` and run without `--demo`.

More details: see `.kiro/specs/ollama-fleet/tasks.md` for the implementation plan.
