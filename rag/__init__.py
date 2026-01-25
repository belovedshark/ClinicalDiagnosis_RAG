"""Package initializer for `rag`.

This module auto-loads a local `.env` file from the repository root (if
present) so all modules under `rag` can read environment variables during
import. It is intentionally lightweight and will not raise if python-dotenv
is not installed; a warning is printed instead.
"""
from pathlib import Path
import os

try:
    # Import lazily to avoid a hard dependency; if python-dotenv isn't
    # installed we simply skip loading .env and rely on externally-set vars.
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        # Do not override existing environment variables
        load_dotenv(env_path, override=False)
        print(f"[rag] Loaded environment variables from {env_path}")
    else:
        # No .env present — this is normal for production installs
        pass
except Exception:
    # Either python-dotenv isn't installed or some other issue occurred.
    # We don't want imports to fail because of this; print a helpful note.
    print("[rag] python-dotenv not available — ensure environment variables are set externally if needed")

__all__ = []
