"""Access the project's Gemini credential in Windows Credential Manager."""

import importlib
import os
from typing import Protocol

SERVICE_NAME = "yizkor-pipeline"
GEMINI_USERNAME = "GEMINI_API_KEY"


class _CredentialBackend(Protocol):
    """Minimal password operations required from the Credential Manager backend."""

    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str) -> None: ...


def _create_windows_keyring() -> _CredentialBackend:
    """Return keyring's Windows Credential Manager backend explicitly."""
    if os.name != "nt":
        raise RuntimeError("Gemini credentials are configured for Windows Credential Manager")

    try:
        windows_backend = importlib.import_module("keyring.backends.Windows")
    except ImportError as error:
        raise RuntimeError(
            "Credential Manager support is unavailable; install project dependencies "
            "with 'python -m pip install -r requirements.txt'"
        ) from error

    return windows_backend.WinVaultKeyring()


def get_stored_gemini_api_key() -> str | None:
    """Return the stored Gemini key, or None when no key has been saved."""
    return _create_windows_keyring().get_password(SERVICE_NAME, GEMINI_USERNAME)


def store_gemini_api_key(api_key: str) -> None:
    """Store a non-empty Gemini key in Windows Credential Manager."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Gemini API key must be a non-empty string")

    _create_windows_keyring().set_password(
        SERVICE_NAME,
        GEMINI_USERNAME,
        api_key.strip(),
    )
