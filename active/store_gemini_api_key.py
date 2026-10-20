"""Prompt for and save the Gemini API key to Windows Credential Manager."""

from getpass import getpass

from active.secret_store import store_gemini_api_key


def main() -> None:
    """Read the key without echoing it and save it to the local user vault."""
    api_key = getpass("Gemini API key: ")
    if not api_key.strip():
        raise SystemExit("No API key entered; nothing was saved.")

    confirmation = getpass("Re-enter Gemini API key: ")
    if api_key != confirmation:
        raise SystemExit("The keys did not match; nothing was saved.")

    try:
        store_gemini_api_key(api_key)
    except (RuntimeError, OSError) as error:
        raise SystemExit(f"Could not save Gemini API key: {error}") from error

    print("Gemini API key saved in Windows Credential Manager.")


if __name__ == "__main__":
	main()
