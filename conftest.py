"""Repo root on sys.path, plus a loader for scripts/ modules.

`scripts` cannot be imported as a package here: verl (editable-installed in the
box venv) already owns that name in sys.modules by the time tests import, so
`scripts.foo` resolves against verl's tree. Load by file path instead.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_script(name: str):
    """Import scripts/<name>.py as a standalone module."""
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
